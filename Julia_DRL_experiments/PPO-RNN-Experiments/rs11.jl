include("../Envs/Env.jl")
include("../Algorithms/PPO-RNN.jl")


println("Number of threads: ", Threads.nthreads())






using Random

# ENV var first, then CLI arg, else 1
seed = haskey(ENV, "SEED") ? parse(Int, ENV["SEED"]) : (length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1)
Random.seed!(seed)
println("Random seed: ", seed)


using RockSample

pomdp = RockSamplePOMDP(11, 11)
pomdp_name = "RS1111"
bool_full_observability = false
env = Env(pomdp, bool_full_observability)
action_space = GetActionSpace(env)
function create_env()
    return Env(pomdp, bool_full_observability)
end

# define convert_o function
function POMDPs.convert_o(T::Type{<:AbstractArray}, o::Int64, m::RockSamplePOMDP)
    vec = zeros(Float32, 3)
    vec[o] = 1.0f0
    return vec
end

# define process action function
function process_action(action::Int, action_space::UnitRange{Int})
    len = length(action_space)
    idx = action - first(action_space) + 1
    (idx < 1 || idx > len) && error("Action $action not in action space")
    onehot = zeros(Float32, len)
    onehot[idx] = 1.0f0
    return onehot
end




state_dim = GetObsDim(env)
# action_dim = length(action_space)
layer_size = 256
rnn_hidden_size = 256
gamma = discount(pomdp)
training_episodes = 1000
batch_size = 4096





# if want to use gpu, need to uncomment the below line, and use device=Flux.gpu
# using CUDA

# agent = PPORNNAgent(action_space, action_dim, state_dim;
#     hidden_dim=layer_size, 
#     rnn_hidden_size=rnn_hidden_size, 
#     batch_size=batch_size, 
#     device=Flux.cpu) 
agent = PPORNNAgent(action_space, state_dim;
    hidden_dim=layer_size, 
    rnn_hidden_size=rnn_hidden_size, 
    batch_size=batch_size, 
    device=Flux.cpu) 






println("Agent created with the following parameters:")
println("Action space: ", action_space)
println("State dimension: ", state_dim)
println("Layer size: ", layer_size)
println("RNN hidden size: ", rnn_hidden_size)
println("Gamma (discount factor): ", gamma)
println("Training episodes: ", training_episodes)
println("Batch size: ", batch_size)
println("Full observability: ", bool_full_observability)


println("Starting training...")



    # 训练
rewards, losses, evals = train!(create_env, agent, training_episodes)

println("Training completed. Evaluating the agent...")
println("Number of training episodes: ", training_episodes)
#print maximum reward: ", maximum(rewards), "\n")
println("Maximum reward: ", maximum(rewards))




final_eval = evaluate(env, agent; num_episodes=10000, max_steps=100) 

println("Final evaluation completed.")
println("Final evaluation results: ", final_eval)



# save the results 
using JLD2


println("Saving results...")

#create directories if they do not exist
mkpath("results")
mkpath("agents")
mkpath("envs")


println("Types of saved data:")
println(typeof(rewards))
println(typeof(losses))
println(typeof(evals))
println(typeof(final_eval))


using JLD2
using Dates # Useful for creating unique filenames with timestamps

println("Saving results...")

# 1. Define the base path and filename
# It's good practice to make filenames descriptive and unique.
# We'll use the POMDP name and a timestamp.
results_dir = "results"
agents_dir = "agents"
pomdp_name = "RS1111" # You already defined this, but let's keep it local for clarity

# Create directories if they do not exist
mkpath(results_dir)
mkpath(agents_dir)
# You already had mkpath("envs"), which is fine.

# Create a unique timestamp for the filename
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

# Construct the full file path using joinpath for cross-platform compatibility
results_filename = "$(pomdp_name)_results_$(timestamp).jld2"
results_filepath = joinpath(results_dir, results_filename)

agent_filename = "$(pomdp_name)_agent_$(timestamp).jld2"
agent_filepath = joinpath(agents_dir, agent_filename)


# 2. Save the training results using jldsave
# jldsave takes a filepath and then keyword arguments for each variable to save.
# The keyword becomes the name of the variable inside the file.
println("Saving training results to: ", results_filepath)
jldsave(results_filepath; 
    rewards=rewards, 
    losses=losses, 
    evaluations=evals, 
    final_evaluation=final_eval,
    gamma=gamma,
    training_episodes=training_episodes,
    batch_size=batch_size
)