using Flux, Statistics, ProgressMeter, Random
using Flux: params, gradient, mse, onehotbatch

# ==== Core Data Structures ====
mutable struct Episode
    observations::Vector{Vector{Float32}}
    actions::Vector{Int}
    rewards::Vector{Float32}
    dones::Vector{Bool}
    length::Int
end

mutable struct DRQNAgent
    q_network::Chain
    target_network::Chain
    optimizer::Any
    replay_buffer::Vector{Episode}
    action_space::UnitRange{Int}
    obs_dim::Int
    γ::Float32
    ϵ::Float32
    min_ϵ::Float32
    ϵ_decay::Float32
    batch_size::Int
    buffer_size::Int
    # target_update_freq::Int
    # update_count::Int
    τ::Float32
    device::Function
    hidden_dim::Int
    rnn_hidden_size::Int
    seq_length::Int
    burn_in::Int
    
    # RNN state management for inference
    current_hidden_state::Any
    last_action::Int  # Track last action for next input
end

# ==== Constructor ====
function DRQNAgent(action_space::UnitRange{Int}, obs_dim::Int;
                 hidden_dim=128,
                 rnn_hidden_size=64,
                 γ=0.99f0,
                 lr=1e-4,
                 ϵ_start=1.0f0,
                 ϵ_end=0.05f0,
                 ϵ_decay=0.995f0,
                 batch_size=32,
                 buffer_size=10000,
                #  target_update_freq=500,
                 seq_length=32,
                 τ = 0.001,
                 burn_in=10,
                 device=Flux.cpu)

    # Input dimension: action (one-hot) + observation
    input_dim = length(action_space) + obs_dim
    
    # Network architecture with action-observation input
    q_net = Chain(
        Dense(input_dim => input_dim, tanh),
        LSTM(input_dim => rnn_hidden_size),
        # LSTM(input_dim => rnn_hidden_size),
        Dense(rnn_hidden_size => hidden_dim, tanh),
        Dense(hidden_dim => length(action_space))
    ) |> device
    
    optimizer = Flux.setup(Adam(lr), q_net)
    
    agent = DRQNAgent(
        q_net,
        deepcopy(q_net) |> device,
        optimizer,
        Vector{Episode}(),
        action_space,
        obs_dim,
        γ,
        ϵ_start,
        ϵ_end,
        ϵ_decay,
        batch_size,
        buffer_size,
        # target_update_freq,
        # 0,
        τ,
        device,
        hidden_dim,
        rnn_hidden_size,
        seq_length,
        burn_in,
        nothing,
        rand(action_space)  # Initialize with random action
    )
    
    reset_hidden_state!(agent)
    return agent
end

# ==== RNN State Management ====
function reset_hidden_state!(agent::DRQNAgent)
    Flux.reset!(agent.q_network)
    agent.current_hidden_state = nothing
    agent.last_action = rand(agent.action_space)  # Reset to random action
end

function save_hidden_state!(agent::DRQNAgent)
    # Extract and save the current hidden state from LSTM
    lstm_layer = agent.q_network[2]
    if lstm_layer.state !== nothing
        agent.current_hidden_state = (copy(lstm_layer.state[1]), copy(lstm_layer.state[2]))
    end
end

function restore_hidden_state!(agent::DRQNAgent)
    if agent.current_hidden_state !== nothing
        lstm_layer = agent.q_network[2]
        lstm_layer.state = agent.current_hidden_state
    end
end

# ==== Experience Replay ====
function store_episode!(agent::DRQNAgent, observations, actions, rewards, dones)
    episode = Episode(
        [cpu(Float32.(obs)) for obs in observations],
        copy(actions),
        Float32.(rewards),
        copy(dones),
        length(actions)
    )
    
    push!(agent.replay_buffer, episode)
    
    # Maintain buffer size
    if length(agent.replay_buffer) > agent.buffer_size
        popfirst!(agent.replay_buffer)
    end
end

function sample_sequences(agent::DRQNAgent)
    if length(agent.replay_buffer) < agent.batch_size
        return nothing
    end
    
    # Sample episodes - need extra space for next_observations
    available_episodes = filter(ep -> ep.length >= agent.seq_length + agent.burn_in + 1, agent.replay_buffer)
    if isempty(available_episodes)
        return nothing
    end
    
    sampled_episodes = [available_episodes[rand(1:length(available_episodes))] 
                       for _ in 1:min(agent.batch_size, length(available_episodes))]
    
    sequences = []
    for episode in sampled_episodes
        # Ensure we don't go out of bounds
        max_seq_start = episode.length - agent.seq_length - agent.burn_in
        if max_seq_start < 1
            continue
        end
        
        start_idx = rand(1:max_seq_start)
        burn_in_end = start_idx + agent.burn_in - 1
        seq_end = burn_in_end + agent.seq_length
        
        # Validate indices before accessing
        if seq_end > episode.length || burn_in_end >= episode.length
            continue
        end
        
        # Construct burn-in actions (need one action per burn-in observation)
        burn_in_actions = Int[]
        for i in start_idx:burn_in_end
            if i == 1
                # For first observation, use a random action (no previous action exists)
                push!(burn_in_actions, rand(agent.action_space))
            else
                # Use the action that was taken at the previous step
                push!(burn_in_actions, episode.actions[i-1])
            end
        end
        
        seq_data = (
            # Burn-in phase
            burn_in_obs = episode.observations[start_idx:burn_in_end],
            burn_in_actions = burn_in_actions,
            
            # Training sequence  
            observations = episode.observations[burn_in_end+1:seq_end],
            actions = episode.actions[burn_in_end:seq_end-1],  # Previous actions for current obs
            taken_actions = episode.actions[burn_in_end+1:seq_end],  # Actions taken at these obs
            rewards = episode.rewards[burn_in_end+1:seq_end],
            next_observations = episode.observations[burn_in_end+2:min(seq_end+1, length(episode.observations))],
            dones = episode.dones[burn_in_end+1:seq_end]
        )
        
        # Verify all sequences have the same length
        expected_len = agent.seq_length
        if length(seq_data.observations) == expected_len &&
           length(seq_data.taken_actions) == expected_len &&
           length(seq_data.rewards) == expected_len &&
           length(seq_data.dones) == expected_len
            push!(sequences, seq_data)
        end
    end
    
    return sequences
end

function create_action_obs_input(obs, action, agent::DRQNAgent)
    """Create concatenated action-observation input"""
    # Convert action to one-hot
    # action_onehot = Float32.(zeros(length(agent.action_space)))
    # action_onehot[action - first(agent.action_space) + 1] = 1.0f0
    action_onehot = Float32.(Flux.onehot(action, 1:length(agent.action_space)))

    # Concatenate action and observation
    return vcat(action_onehot, Float32.(obs))
end

function create_batched_inputs(agent::DRQNAgent, sequences)
    if isempty(sequences)
        return nothing
    end
    
    batch_size = length(sequences)
    input_dim = length(agent.action_space) + agent.obs_dim
    
    # Create input tensors
    inputs = zeros(Float32, input_dim, agent.seq_length, batch_size)
    next_inputs = zeros(Float32, input_dim, agent.seq_length, batch_size)
    taken_actions = zeros(Int, agent.seq_length, batch_size)
    rewards = zeros(Float32, agent.seq_length, batch_size)
    dones = zeros(Float32, agent.seq_length, batch_size)
    masks = zeros(Float32, agent.seq_length, batch_size)
    
    for (i, seq) in enumerate(sequences)
        seq_len = length(seq.observations)
        
        for t in 1:seq_len
            # Current input: previous_action + current_observation
            inputs[:, t, i] = create_action_obs_input(seq.observations[t], seq.actions[t], agent)
            
            # Next input: current_action + next_observation
            # Handle case where next_observations might be shorter
            if t <= length(seq.next_observations)
                next_action = seq.taken_actions[t]
                next_inputs[:, t, i] = create_action_obs_input(seq.next_observations[t], next_action, agent)
            else
                # Use terminal state (zeros) if no next observation
                terminal_input = create_action_obs_input(zeros(Float32, agent.obs_dim), seq.taken_actions[t], agent)
                next_inputs[:, t, i] = terminal_input
            end
            
            # Other data
            taken_actions[t, i] = seq.taken_actions[t]
            rewards[t, i] = seq.rewards[t]
            dones[t, i] = seq.dones[t]
            masks[t, i] = 1.0f0
        end
    end
    
    return (
        inputs = agent.device(inputs),
        next_inputs = agent.device(next_inputs),
        taken_actions = taken_actions,
        rewards = agent.device(rewards),
        dones = agent.device(dones),
        masks = agent.device(masks),
        burn_in_data = sequences
    )
end

function clip_gradients(grads, clip_value)
    ps = Flux.params(grads)
    if isempty(ps)
        println("Warning: No parameters to clip gradients.")
        return grads  # Nothing to clip
    end

    norm = sqrt(sum(x -> sum(abs2, x), ps))
    scale = min(1.0f0, clip_value / (norm + 1f-6))
    Flux.fmap(g -> g === nothing ? nothing : g .* scale, grads)
end


# ==== Network Updates ====
function update_networks!(agent::DRQNAgent)
    sequences = sample_sequences(agent)
    if sequences === nothing || isempty(sequences)
        return 0.0f0
    end
    
    batch = create_batched_inputs(agent, sequences)
    if batch === nothing
        return 0.0f0
    end
    
    loss, grads = Flux.withgradient(agent.q_network) do model
        compute_drqn_loss(agent, model, batch)
    end
    
    # Clip gradients to prevent exploding gradients
    if grads[1] !== nothing
        clip_gradients(grads, 10.0f0)
    end
    
    # Update parameters
    Flux.update!(agent.optimizer, agent.q_network, grads[1])
    
    # Update target network
    # agent.update_count += 1
    # if agent.update_count % agent.target_update_freq == 0
    #     copy_params!(agent.target_network, agent.q_network)
    # end
    soft_update!(agent.target_network, agent.q_network, agent.τ)  # τ = 0.001 is a typical starting point

    return loss
end

function compute_drqn_loss(agent::DRQNAgent, model, batch)
    batch_size = size(batch.inputs, 3)
    total_loss = 0.0f0
    
    for i in 1:batch_size
        # Reset model state
        Flux.reset!(model)
        
        # Burn-in phase - don't compute gradients, just update hidden state
        burn_in_seq = batch.burn_in_data[i]
        for t in 1:length(burn_in_seq.burn_in_obs)
            burn_in_input = create_action_obs_input(
                burn_in_seq.burn_in_obs[t], 
                burn_in_seq.burn_in_actions[t], 
                agent
            )
            burn_in_input_gpu = agent.device(reshape(burn_in_input, :, 1))
            model(burn_in_input_gpu)  # Update hidden state only
        end
        
        # Main sequence forward pass
        input_seq = batch.inputs[:, :, i:i]  # Keep batch dimension
        q_values = model(input_seq)  # Shape: [num_actions, seq_len, 1]
        
        # Compute targets using target network
        Flux.reset!(agent.target_network)
        
        # Burn-in for target network
        for t in 1:length(burn_in_seq.burn_in_obs)
            burn_in_input = create_action_obs_input(
                burn_in_seq.burn_in_obs[t], 
                burn_in_seq.burn_in_actions[t], 
                agent
            )
            burn_in_input_gpu = agent.device(reshape(burn_in_input, :, 1))
            agent.target_network(burn_in_input_gpu)
        end
        
        next_q_values = agent.target_network(batch.next_inputs[:, :, i:i])
        max_next_q = maximum(next_q_values, dims=1)  # Max over actions: [1, seq_len, 1]
        
        # Compute TD targets
        targets = batch.rewards[:, i] .+ agent.γ .* (1f0 .- batch.dones[:, i]) .* vec(max_next_q)
        
        # Get Q-values for taken actions
        taken_actions_i = batch.taken_actions[:, i]
        selected_q = [q_values[taken_actions_i[t], t, 1] for t in 1:length(taken_actions_i)]
        
        # Apply mask and compute MSE loss
        mask_i = batch.masks[:, i]
        valid_steps = sum(mask_i)
        if valid_steps > 0
            masked_errors = mask_i .* (selected_q .- targets).^2
            seq_loss = sum(masked_errors) / valid_steps
            total_loss += seq_loss
        end
    end
    
    return total_loss / batch_size
end

# ==== Policy Functions ====
function choose_action(agent::DRQNAgent, obs; eval_mode=false)
    # Create action-observation input using last action
    input_vec = create_action_obs_input(obs, agent.last_action, agent)
    input_tensor = agent.device(reshape(input_vec, :, 1))
    
    if !eval_mode && rand() < agent.ϵ
        # Even for random actions, we need to forward through network
        # to maintain RNN state consistency
        q_values = agent.q_network(input_tensor)
        action = rand(agent.action_space)
    else
        q_values = agent.q_network(input_tensor)
        action = argmax(q_values[:, 1])
    end
    
    # Update last action for next step
    agent.last_action = action
    
    return action
end

function decay_exploration!(agent::DRQNAgent)
    agent.ϵ = max(agent.min_ϵ, agent.ϵ * agent.ϵ_decay)
end

# ==== Utility Functions ====
function copy_params!(dest, src)
    for (d, s) in zip(params(dest), params(src))
        d .= s
    end
end

function soft_update!(target, online, τ)
    for (t, o) in zip(params(target), params(online))
        @. t = τ * o + (1 - τ) * t
    end
end


# ==== Training Interface ====
function train!(env, agent::DRQNAgent, num_episodes::Int; 
               eval_interval=100, max_steps=200, verbose=true)
    episode_rewards = Float32[]
    episode_losses = Float32[]
    eval_scores = Float32[]
    
    @showprogress for ep in 1:num_episodes
        # Reset environment and agent
        obs = reset!(env)
        reset_hidden_state!(agent)
        
        # Episode storage
        observations = [obs]
        actions = Int[]
        rewards = Float32[]
        dones = Bool[]
        
        total_reward = 0.0f0
        
        for step in 1:max_steps
            # Choose action (this updates agent.last_action)
            action = choose_action(agent, obs)
            
            # Environment step
            next_obs, reward, done = step!(env, action)
            
            # Store transition
            push!(actions, action)
            push!(rewards, reward)
            push!(dones, done)
            push!(observations, next_obs)
            
            total_reward += reward
            obs = next_obs
            
            if done
                break
            end
        end
        
        # Store complete episode if it's long enough
        # We need observations to include the final observation for next_observations
        if length(actions) >= agent.seq_length + agent.burn_in
            store_episode!(agent, observations, actions, rewards, dones)
        end
        
        # Update networks
        loss = update_networks!(agent)
        
        # Record statistics
        push!(episode_rewards, total_reward)
        push!(episode_losses, loss)
        decay_exploration!(agent)
        
        # Periodic evaluation and logging
        if ep % eval_interval == 0 && verbose
            eval_score = length(eval_scores) > 0 ? eval_scores[end] : 0.0f0
            
            # Evaluate current policy
            if length(agent.replay_buffer) > 0
                eval_score = evaluate(env, agent)
                push!(eval_scores, eval_score)
            end
            
            recent_rewards = episode_rewards[max(1, end-eval_interval+1):end]
            recent_losses = episode_losses[max(1, end-eval_interval+1):end]
            
            println("Episode $ep | ",
                  "Avg Reward: $(round(mean(recent_rewards), digits=2)) | ",
                  "Avg Loss: $(round(mean(recent_losses), digits=4)) | ",
                  "Eval Score: $(round(eval_score, digits=2)) | ",
                  "ϵ: $(round(agent.ϵ, digits=3)) | ",
                  "Buffer: $(length(agent.replay_buffer)) episodes")
        end
    end
    
    return (episode_rewards, episode_losses, eval_scores)
end

# ==== Evaluation Function ====
function evaluate(env, agent::DRQNAgent; num_episodes=5, max_steps=200)
    total_rewards = Float32[]
    
    for _ in 1:num_episodes
        obs = reset!(env)
        reset_hidden_state!(agent)
        total_reward = 0.0f0
        
        for _ in 1:max_steps
            action = choose_action(agent, obs, eval_mode=true)
            obs, reward, done = step!(env, action)
            total_reward += reward
            
            if done
                break
            end
        end
        
        push!(total_rewards, total_reward)
    end
    
    return mean(total_rewards)
end

# ==== Additional Utility Functions ====
function get_network_stats(agent::DRQNAgent)
    """Get statistics about the network weights for debugging"""
    total_params = 0
    for p in params(agent.q_network)
        total_params += length(p)
    end
    
    return (
        total_parameters = total_params,
        buffer_size = length(agent.replay_buffer),
        epsilon = agent.ϵ,
        # update_count = agent.update_count
    )
end

# Convenience functions
function create_lstm_agent(action_space, state_dim; kwargs...)
    return DRQNAgent(action_space, state_dim; kwargs...)
end
