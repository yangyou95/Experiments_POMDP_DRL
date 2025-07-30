using Flux, Statistics, ProgressMeter, Random, Distributions
using Flux: params, gradient, logitcrossentropy, onehotbatch, logsoftmax
using Zygote
using CUDA
using RockSample
using Dates
using JLD2

# Include the necessary files
include("../Algorithms/new-PPO.jl")
include("../Envs/Env.jl")

# Setup
Random.seed!(1)
@assert CUDA.functional() "CUDA is not functional"

# GPU Selection
println("Available GPUs:")
for (i, dev) in enumerate(CUDA.devices())
    println("GPU $i: ", CUDA.name(dev))
end

# Choose GPU (modify this number to select different GPU)
selected_gpu = 2  # Change this to 0, 1, 2, or 3 to select different GPU
gpu_devices = collect(CUDA.devices())
CUDA.device!(gpu_devices[selected_gpu + 1])  # +1 because Julia is 1-indexed

println("=== Episode-Centric PPO Training (Fast GPU + Original Logging) ===")
println("CUDA functional: ", CUDA.functional())
println("Selected GPU: ", CUDA.name(CUDA.device()))

# Create environment and agent
pomdp = RockSamplePOMDP(7, 8)
pomdp_name = "RS78"
bool_full_observability = false

function create_env()
    return Env(pomdp, bool_full_observability)
end

# Test the environment to get dimensions
temp_env = create_env()
action_space = GetActionSpace(temp_env)
state_dim = GetObsDim(temp_env)

# This is needed by your Env/POMDP
function POMDPs.convert_o(T::Type{<:AbstractArray}, o::Int64, m::RockSamplePOMDP)
    vec = zeros(Float32, 3)
    vec[o] = 1.0f0
    return vec
end

println("State dimension: $state_dim")
println("Action space: $action_space")

# === HYPERPARAMETERS (matching original PPO-RNN.jl) ===
layer_size = 128  # Smaller networks like original
rnn_hidden_size = 128  # Smaller RNN like original
gamma = discount(pomdp)
sequence_length = 64
minibatch_size = 64
n_epochs = 4  # Fewer epochs like original

# === CRITICAL: Use original PPO-RNN.jl training approach ===
# The original uses episode-based training, not step-based
# This allows better exploration of long-term strategies

# Training parameters (matching original PPO-RNN.jl approach)
n_envs = 64
n_steps = 200  # Increased to allow more episodes to complete
num_updates = 2000  # More updates to match original training length
eval_interval = 10  # Less frequent evaluation like original

# Create the agent
agent = PPORNNAgent(action_space, state_dim;
    hidden_dim=layer_size,
    rnn_hidden_size=rnn_hidden_size,
    sequence_length=sequence_length,
    minibatch_size=minibatch_size,
    n_epochs=n_epochs,
    γ=gamma,
    ent_coef=0.02f0,
    lr_policy=2.5e-4,
    lr_value=2.5e-4,

)

println("\n=== Training Configuration ===")
println("Network Size: $layer_size")
println("RNN Hidden Size: $rnn_hidden_size")
println("Sequence Length: $sequence_length")
println("Parallel Environments: $n_envs")
println("Steps per Environment: $n_steps")
println("Total samples per update: $(n_envs * n_steps)")
println("Minibatch Size: $minibatch_size")
println("Epochs per update: $n_epochs")
println("Total Training Updates: $num_updates")
println("Entropy Coefficient: $(agent.ent_coef)")
println("Policy Learning Rate: $(agent.lr_policy)")
println("Value Learning Rate: $(agent.lr_value)")
println("Clip Gradients: $(agent.clip_grads)")
println("Clip Value: $(agent.clip_value)")
println("Gamma: $(agent.γ)")
println("Lambda: $(agent.λ)")
println("Expected samples/second: ~1000+")

# === CHANGES TO PREVENT MYOPIC CONVERGENCE ===
println("⚠️  Using smaller networks and fixed entropy to prevent myopic convergence")
println("   - Network size: 64 (vs 256) to match original PPO-RNN.jl")
println("   - Fixed entropy coefficient (no decay) to maintain exploration")
println("   - More training updates (2000) to allow complex strategy learning")

# === WARMUP FUNCTION TO REDUCE FIRST EPOCH SLOWDOWN ===
function warmup_gpu(agent::PPORNNAgent, env_constructor)
    println("Warming up GPU (compiling kernels)...")
    
    # Create a small test environment
    test_env = env_constructor()
    test_obs = reset!(test_env)
    
    # Warm up policy network
    test_input = zeros(Float32, agent.state_dim + agent.n_actions, 1, 1)
    test_input[agent.n_actions + 1:end, 1, 1] = Float32.(vec(test_obs))
    test_input_gpu = gpu(test_input)
    
    # Run a few forward passes to compile kernels
    for _ in 1:5
        Flux.reset!(agent.policy_net)
        Flux.reset!(agent.value_net)
        _ = agent.policy_net(test_input_gpu)
        _ = agent.value_net(test_input_gpu)
    end
    
    println("GPU warmup completed!")
end

# === EPISODE-CENTRIC REWARD COLLECTION FUNCTION ===
function collect_episode_rewards_vectorized(env_constructor, agent::PPORNNAgent, n_envs::Int, n_steps::Int)
    # Create environments
    envs = [env_constructor() for _ in 1:n_envs]
    
    # Initialize
    initial_obs_int = [reset!(env) for env in envs]
    initial_obs_vecs = [POMDPs.convert_o(Vector{Float32}, obs, envs[i].model) for (i, obs) in enumerate(initial_obs_int)]
    next_states = hcat(initial_obs_vecs...)
    
    next_dones = zeros(Bool, n_envs)
    prev_actions = ones(Int, n_envs)
    
    # Create network copies for each environment
    policy_nets = [deepcopy(agent.policy_net) |> gpu for _ in 1:n_envs]
    value_nets = [deepcopy(agent.value_net) |> gpu for _ in 1:n_envs]
    Flux.reset!.(policy_nets)
    Flux.reset!.(value_nets)
    
    # Data collection arrays
    b_action_obs_inputs = zeros(Float32, agent.state_dim + agent.n_actions, n_steps, n_envs)
    b_actions = zeros(Int32, n_steps, n_envs)
    b_log_probs = zeros(Float32, n_steps, n_envs)
    b_values = zeros(Float32, n_steps, n_envs)
    b_rewards = zeros(Float32, n_steps, n_envs)
    b_dones = zeros(Bool, n_steps, n_envs)
    
    # Episode tracking for episode-centric rewards
    episode_rewards = Float32[]
    current_episode_rewards = zeros(Float32, n_envs)
    episode_lengths = Int[]
    current_episode_lengths = zeros(Int, n_envs)
    
    # Main collection loop
    for t in 1:n_steps
        prev_actions_onehot = Float32.(onehotbatch(prev_actions, agent.action_space))
        action_obs_batch = vcat(prev_actions_onehot, next_states)
        
        b_action_obs_inputs[:, t, :] = action_obs_batch
        
        action_obs_gpu = gpu(action_obs_batch)
        logits = policy_nets[1](action_obs_gpu)  # Use first net as template
        values = value_nets[1](action_obs_gpu)
        
        probs = softmax(logits)
        probs_cpu = cpu(probs)
        dist = [Categorical(probs_cpu[:, i]) for i in 1:size(probs_cpu, 2)]
        actions = rand.(dist)
        log_probs = logpdf.(dist, actions)
        
        b_actions[t, :] = actions
        b_log_probs[t, :] = log_probs
        b_values[t, :] = cpu(vec(values))
        b_dones[t, :] = next_dones
        
        # Environment step
        next_states_vec = Vector{Vector{Float32}}(undef, n_envs)
        
        for i in 1:n_envs
            s_int, r, d = step!(envs[i], actions[i])
            b_rewards[t, i] = r
            
            # Track episode rewards (episode-centric)
            current_episode_rewards[i] += r
            current_episode_lengths[i] += 1
            
            s_vec = POMDPs.convert_o(Vector{Float32}, s_int, envs[i].model)
            
            if d
                # Episode completed - record it
                push!(episode_rewards, current_episode_rewards[i])
                push!(episode_lengths, current_episode_lengths[i])
                
                # Reset for next episode
                s_int_reset = reset!(envs[i])
                s_vec = POMDPs.convert_o(Vector{Float32}, s_int_reset, envs[i].model)
                current_episode_rewards[i] = 0.0f0
                current_episode_lengths[i] = 0
                Flux.reset!(policy_nets[i])
                Flux.reset!(value_nets[i])
            end
            next_states_vec[i] = s_vec
        end
        next_states = hcat(next_states_vec...)
        prev_actions = actions
    end
    
    # Process data for training (same as before)
    batch_size = n_steps * n_envs
    num_sequences = floor(Int, batch_size / agent.sequence_length)
    effective_size = num_sequences * agent.sequence_length
    
    function reshape_for_rnn(data)
        flat_data = if ndims(data) == 3
            reshape(data, size(data, 1), :)
        else
            reshape(data, 1, :)
        end
        flat_data_trunc = flat_data[:, 1:effective_size]
        return reshape(flat_data_trunc, size(flat_data, 1), agent.sequence_length, num_sequences)
    end
    
    b_actions_rnn = reshape_for_rnn(b_actions)
    b_log_probs_rnn = reshape_for_rnn(b_log_probs)
    b_values_rnn = reshape_for_rnn(b_values)
    b_action_obs_rnn = reshape_for_rnn(b_action_obs_inputs)
    
    # GAE calculation
    b_advantages = zeros(Float32, n_steps, n_envs)
    last_advantage = zeros(Float32, n_envs)
    
    last_prev_actions_onehot = Float32.(onehotbatch(prev_actions, agent.action_space))
    last_action_obs = vcat(last_prev_actions_onehot, next_states)
    last_values = cpu(vec(value_nets[1](gpu(last_action_obs))))
    
    for t in reverse(1:n_steps)
        mask = 1.0f0 .- b_dones[t, :]
        next_val = (t == n_steps) ? last_values : b_values[t+1, :]
        delta = b_rewards[t, :] .+ agent.γ .* next_val .* mask .- b_values[t, :]
        last_advantage = delta .+ agent.γ .* agent.λ .* last_advantage .* mask
        b_advantages[t, :] = last_advantage
    end
    
    b_returns = b_advantages .+ b_values
    b_advantages_rnn = reshape_for_rnn(b_advantages)
    b_returns_rnn = reshape_for_rnn(b_returns)
    
    # Normalize advantages
    b_advantages_squeezed = dropdims(b_advantages_rnn, dims=1)
    b_advantages_squeezed = (b_advantages_squeezed .- mean(b_advantages_squeezed)) ./ (std(b_advantages_squeezed) .+ 1f-8)
    b_advantages_rnn = reshape(b_advantages_squeezed, 1, agent.sequence_length, num_sequences)
    
    # Calculate episode-centric average reward (like original PPO-RNN.jl)
    episode_avg_reward = isempty(episode_rewards) ? 0.0f0 : mean(episode_rewards)
    
    # Also calculate step-centric reward for comparison
    step_avg_reward = sum(b_rewards) / (n_steps * n_envs)
    
    return (
        b_action_obs_rnn,
        b_actions_rnn,
        b_log_probs_rnn,
        b_advantages_rnn,
        b_returns_rnn
    ), episode_avg_reward, episode_rewards, episode_lengths, step_avg_reward
end

# === TRAINING LOOP ===
println("\nStarting episode-centric training...")

# Warm up GPU to reduce first epoch slowdown
warmup_gpu(agent, create_env)

# Original PPO-RNN.jl style logging
rewards = Float32[]
losses = Float32[]
evals = Float32[]

# Additional detailed logging
policy_losses = Float32[]
value_losses = Float32[]
episode_counts = Int[]
total_steps = 0

# Track timing
total_start_time = time()

@showprogress "Training" for update in 1:num_updates
    update_start_time = time()
    
    # Collect data with episode-centric rewards
    data_collection_start = time()
    data, episode_avg_reward, episode_rewards_batch, episode_lengths_batch, step_avg_reward = collect_episode_rewards_vectorized(create_env, agent, n_envs, n_steps)
    data_collection_time = time() - data_collection_start
    
    # Update networks
    network_update_start = time()
    policy_loss, value_loss = update_networks!(agent, data)
    network_update_time = time() - network_update_start
    
    # Log results (original PPO-RNN.jl style)
    push!(rewards, episode_avg_reward)
    push!(losses, (policy_loss + value_loss) / 2)  # Combined loss like original
    push!(policy_losses, policy_loss)
    push!(value_losses, value_loss)
    push!(episode_counts, length(episode_rewards_batch))
    
    global total_steps += n_envs * n_steps
    
    update_time = time() - update_start_time
    samples_per_second = (n_envs * n_steps) / update_time
    
    # Print progress (original style with additional info)
    if update % 1 == 0
        avg_length_str = length(episode_lengths_batch) > 0 ? "$(round(mean(episode_lengths_batch), digits=1))" : "N/A"
        println("Update $update/$num_updates | ",
                "Episode Reward: $(round(episode_avg_reward, digits=3)) | ",
                "Step Reward: $(round(step_avg_reward, digits=3)) | ",
                "Episodes: $(length(episode_rewards_batch)) | ",
                "Avg Length: $avg_length_str | ",
                "Loss: $(round((policy_loss + value_loss) / 2, digits=4)) | ",
                "Samples/s: $(round(samples_per_second, digits=0))")
    end

    # Evaluation (like original PPO-RNN.jl)
    if update % eval_interval == 0
        eval_start = time()
        eval_score = evaluate(create_env(), agent; num_episodes=50, max_steps=100)
        eval_time = time() - eval_start
        push!(evals, eval_score)
        
        println("-"^60)
        println("EVALUATION @ Update $update | ",
                "Eval Score: $(round(eval_score, digits=2)) | ",
                "Eval Time: $(round(eval_time, digits=2))s")
        println("-"^60)
    end
    
    # Keep entropy coefficient fixed (like original PPO-RNN.jl)
    # agent.ent_coef = max(agent.ent_coef * 0.995f0, 1e-4)  # Commented out to prevent myopic convergence
end

total_training_time = time() - total_start_time

# === RESULTS SUMMARY (Original PPO-RNN.jl style) ===
println("\n" * "="^60)
println("EPISODE-CENTRIC TRAINING COMPLETED!")
println("="^60)
println("Number of training updates: $num_updates")
println("Total training time: $(round(total_training_time, digits=2))s")
println("Average samples/second: $(round(total_steps / total_training_time, digits=0))")

if !isempty(rewards)
    println("Maximum episode reward: $(maximum(rewards))")
    println("Final episode reward: $(rewards[end])")
    println("Average episode reward: $(round(mean(rewards), digits=3))")
end

if !isempty(evals)
    println("Final evaluation score: $(round(evals[end], digits=2))")
end

# Save results (original PPO-RNN.jl style)
println("\nSaving results...")
results_dir = "results"
agents_dir = "agents"
mkpath(results_dir)
mkpath(agents_dir)

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
results_filename = "$(pomdp_name)_episode_centric_results_$(timestamp).jld2"
results_filepath = joinpath(results_dir, results_filename)

agent_filename = "$(pomdp_name)_episode_centric_agent_$(timestamp).jld2"
agent_filepath = joinpath(agents_dir, agent_filename)

# Save in original PPO-RNN.jl format
@save results_filepath rewards losses evals agent gamma num_updates n_envs n_steps
@save agent_filepath agent

println("Results saved to: $results_filepath")
println("Agent saved to: $agent_filepath")

# === FINAL EVALUATION WITH MULTIPLE SEEDS (like rs78.jl) ===
println("\n" * "="^60)
println("FINAL EVALUATION WITH MULTIPLE RANDOM SEEDS")
println("="^60)

final_eval_scores = Float32[]
seeds = [1, 2, 3, 4, 5]  # 5 different random seeds

for (i, seed) in enumerate(seeds)
    println("Running evaluation with seed $seed...")
    
    # Set seed for this evaluation
    Random.seed!(seed)
    
    # Run evaluation with 100,000 episodes (like rs78.jl)
    eval_start = time()
    eval_score = evaluate(create_env(), agent; num_episodes=100000, max_steps=100)
    eval_time = time() - eval_start
    
    push!(final_eval_scores, eval_score)
    
    println("Seed $seed: Eval Score = $(round(eval_score, digits=2)) | Time = $(round(eval_time, digits=2))s")
end

# Calculate statistics
mean_final_eval = mean(final_eval_scores)
std_final_eval = std(final_eval_scores)
min_final_eval = minimum(final_eval_scores)
max_final_eval = maximum(final_eval_scores)

println("\n" * "-"^60)
println("FINAL EVALUATION RESULTS:")
println("-"^60)
println("Mean: $(round(mean_final_eval, digits=2))")
println("Std:  $(round(std_final_eval, digits=2))")
println("Min:  $(round(min_final_eval, digits=2))")
println("Max:  $(round(max_final_eval, digits=2))")
println("-"^60)

# Save final evaluation results
final_eval_filename = "$(pomdp_name)_final_eval_$(timestamp).jld2"
final_eval_filepath = joinpath(results_dir, final_eval_filename)

@save final_eval_filepath final_eval_scores seeds mean_final_eval std_final_eval min_final_eval max_final_eval

println("Final evaluation results saved to: $final_eval_filepath")
println("\nEpisode-centric training completed successfully!") 