using Flux, Statistics, ProgressMeter, Random, Distributions
using Flux: params, gradient, logitcrossentropy, onehotbatch
using Zygote
using Base.Threads

# === 1. 核心数据结构 ===
mutable struct PPORNNAgent
    policy_net::Chain
    value_net::Chain
    optimizer_policy::Any
    optimizer_value::Any
    action_space::Any
    γ::Float32
    λ::Float32
    ϵ::Float32
    ent_coef::Float32
    n_epochs::Int
    batch_size::Int
    minibatch_size::Int
    device::Function
    clip_grads::Bool
    clip_value::Float32
    # RNN specific parameters
    rnn_hidden_size::Int
    sequence_length::Int
    max_episode_length::Int
    # Action-observation specific
    state_dim::Int
    n_actions::Int
    # Threading specific
    n_envs::Int
    use_threading::Bool
end

# === 2. 初始化函数 ===
function PPORNNAgent(
    action_space::Any, 
    state_dim::Int;
    hidden_dim=64,
    rnn_hidden_size=128,
    sequence_length=32,
    max_episode_length=200,
    γ=0.99f0,
    λ=0.95f0,
    ϵ=0.2f0,
    ent_coef=0.01f0,
    n_epochs=4,
    batch_size=2048,
    minibatch_size=64,
    lr_policy=3e-4,
    lr_value=1e-4,
    device=Flux.cpu,
    clip_grads=true,
    clip_value=1.0f0,
    n_envs=nthreads(),  # 默认使用所有线程
    use_threading=true
)

    n_actions = length(action_space)
    n_obs = state_dim

    # Input dimension: observation + one-hot action
    input_dim = n_obs + n_actions 

    # Policy network with LSTM
    policy_net = Chain(
        # Dense(input_dim => hidden_dim, tanh),
        # LSTM(hidden_dim => rnn_hidden_size),
        LSTM(input_dim => rnn_hidden_size),
        Dense(rnn_hidden_size => hidden_dim, tanh),
        Dense(hidden_dim => n_actions)
    ) |> device
    
    # Value network with LSTM
    value_net = Chain(
        Dense(input_dim => hidden_dim, tanh),
        LSTM(hidden_dim => rnn_hidden_size),
        Dense(rnn_hidden_size => hidden_dim, tanh),
        Dense(hidden_dim => 1, identity)
    ) |> device
    
    # 使用Flux的新优化器接口
    optimizer_policy = Flux.setup(Adam(lr_policy), policy_net)
    optimizer_value = Flux.setup(Adam(lr_value), value_net)
    
    PPORNNAgent(
        policy_net, value_net,
        optimizer_policy, optimizer_value,
        action_space,
        γ, λ, ϵ, ent_coef,
        n_epochs, batch_size, minibatch_size,
        device, clip_grads, clip_value,
        rnn_hidden_size, sequence_length, max_episode_length,
        state_dim, n_actions,
        n_envs, use_threading
    )
end

# === 3. 重置RNN隐藏状态 ===
function reset_hidden_states!(agent::PPORNNAgent)
    Flux.reset!(agent.policy_net)
    Flux.reset!(agent.value_net)
end

function process_action(action::Int, action_space::UnitRange{Int})
    len = length(action_space)
    idx = action - first(action_space) + 1
    (idx < 1 || idx > len) && error("Action $action not in action space")
    onehot = zeros(Float32, len)
    onehot[idx] = 1.0f0
    return onehot
end

# === 5. 构建动作-观察输入 ===
function build_action_obs_input(state, prev_action::Int, agent::PPORNNAgent)
    # 将状态转换为Float32向量
    state_vec = Float32.(vec(state))
    
    # 创建前一个动作的one-hot编码
    action_onehot = process_action(prev_action, agent.action_space)
    
    # 拼接状态和动作
    # input_vec = vcat(state_vec, action_onehot)
    input_vec = vcat(action_onehot, state_vec)

    return input_vec
end

# === 6. 单环境轨迹收集（用于线程化） ===
function collect_single_env_trajectory(env, agent::PPORNNAgent, steps_per_env::Int; max_ep_len=nothing)
    if max_ep_len === nothing
        max_ep_len = agent.max_episode_length
    end
    
    episodes = []
    episode_rewards = Float32[]
    
    state = reset!(env)
    # 为每个线程创建独立的网络副本来避免竞争条件
    policy_net_copy = deepcopy(agent.policy_net)
    value_net_copy = deepcopy(agent.value_net)
    Flux.reset!(policy_net_copy)
    Flux.reset!(value_net_copy)
    
    current_episode = Dict(
        :states => [],
        :actions => Int[],
        :prev_actions => Int[],
        :rewards => Float32[],
        :dones => Bool[],
        :old_log_probs => Float32[],
        :values => Float32[]
    )
    
    steps_collected = 0
    current_ep_reward = 0.0f0
    prev_action = 1
    
    while steps_collected < steps_per_env
        # 构建动作-观察输入
        action_obs_input = build_action_obs_input(state, prev_action, agent)
        action_obs_input_gpu = agent.device(reshape(action_obs_input, :, 1))
        
        # 选择动作
        logits = policy_net_copy(action_obs_input_gpu)
        probs = softmax(logits)
        dist = Categorical(cpu(vec(probs)))
        action = rand(dist)
        log_prob = logpdf(dist, action)
        value = value_net_copy(action_obs_input_gpu)[1]
        
        # 环境交互
        next_state, reward, done = step!(env, action)
        current_ep_reward += reward
        
        # 存储数据
        push!(current_episode[:states], state)
        push!(current_episode[:actions], action)
        push!(current_episode[:prev_actions], prev_action)
        push!(current_episode[:rewards], Float32(reward))
        push!(current_episode[:dones], done)
        push!(current_episode[:old_log_probs], Float32(log_prob))
        push!(current_episode[:values], Float32(cpu(value)))
        
        steps_collected += 1
        prev_action = action
        
        # Episode结束或达到最大长度
        if done || length(current_episode[:states]) >= max_ep_len
            push!(episodes, current_episode)
            push!(episode_rewards, current_ep_reward)
            
            # 重置episode
            current_episode = Dict(
                :states => [],
                :actions => Int[],
                :prev_actions => Int[],
                :rewards => Float32[],
                :dones => Bool[],
                :old_log_probs => Float32[],
                :values => Float32[]
            )
            current_ep_reward = 0.0f0
            state = reset!(env)
            Flux.reset!(policy_net_copy)
            Flux.reset!(value_net_copy)
            prev_action = 1
        else
            state = next_state
        end
    end
    
    # 如果还有未完成的episode，也加入
    if !isempty(current_episode[:states])
        push!(episodes, current_episode)
        push!(episode_rewards, current_ep_reward)
    end
    
    return episodes, episode_rewards
end

# === 7. 并行轨迹收集 ===
function collect_trajectories_parallel(env_constructor, agent::PPORNNAgent, n_steps::Int; max_ep_len=nothing)
    steps_per_env = div(n_steps, agent.n_envs)
    
    # 存储结果的容器
    all_episodes = Vector{Vector{Any}}(undef, agent.n_envs)
    all_rewards = Vector{Vector{Float32}}(undef, agent.n_envs)
    
    # 并行收集轨迹
    @threads for i in 1:agent.n_envs
        env = env_constructor()  # 每个线程创建自己的环境
        episodes, rewards = collect_single_env_trajectory(env, agent, steps_per_env, max_ep_len=max_ep_len)
        all_episodes[i] = episodes
        all_rewards[i] = rewards
    end
    
    # 合并所有结果
    combined_episodes = vcat(all_episodes...)
    combined_rewards = vcat(all_rewards...)
    
    # 转换为序列数据
    trajectory_data = process_episodes_to_sequences_parallel(agent, combined_episodes)
    
    return trajectory_data, mean(combined_rewards)
end

# === 8. 单线程轨迹收集（向后兼容） ===
function collect_trajectories(env, agent::PPORNNAgent, n_steps::Int; max_ep_len=nothing)
    if agent.use_threading
        # 如果使用线程，需要环境构造函数
        error("For threaded collection, use collect_trajectories_parallel with env_constructor")
    end
    
    if max_ep_len === nothing
        max_ep_len = agent.max_episode_length
    end
    
    episodes = []
    episode_rewards = Float32[]
    
    state = reset!(env)
    reset_hidden_states!(agent)
    
    current_episode = Dict(
        :states => [],
        :actions => Int[],
        :prev_actions => Int[],
        :rewards => Float32[],
        :dones => Bool[],
        :old_log_probs => Float32[],
        :values => Float32[]
    )
    
    total_steps = 0
    current_ep_reward = 0.0f0
    prev_action = 1
    
    while total_steps < n_steps
        # 构建动作-观察输入
        action_obs_input = build_action_obs_input(state, prev_action, agent)
        action_obs_input_gpu = agent.device(reshape(action_obs_input, :, 1))
        
        # 选择动作
        logits = agent.policy_net(action_obs_input_gpu)
        probs = softmax(logits)
        dist = Categorical(cpu(vec(probs)))
        action = rand(dist)
        log_prob = logpdf(dist, action)
        value = agent.value_net(action_obs_input_gpu)[1]
        
        # 环境交互
        next_state, reward, done = step!(env, action)
        current_ep_reward += reward
        
        # 存储数据
        push!(current_episode[:states], state)
        push!(current_episode[:actions], action)
        push!(current_episode[:prev_actions], prev_action)
        push!(current_episode[:rewards], Float32(reward))
        push!(current_episode[:dones], done)
        push!(current_episode[:old_log_probs], Float32(log_prob))
        push!(current_episode[:values], Float32(cpu(value)))
        
        total_steps += 1
        prev_action = action
        
        # Episode结束或达到最大长度
        if done || length(current_episode[:states]) >= max_ep_len
            push!(episodes, current_episode)
            push!(episode_rewards, current_ep_reward)
            
            # 重置episode
            current_episode = Dict(
                :states => [],
                :actions => Int[],
                :prev_actions => Int[],
                :rewards => Float32[],
                :dones => Bool[],
                :old_log_probs => Float32[],
                :values => Float32[]
            )
            current_ep_reward = 0.0f0
            state = reset!(env)
            reset_hidden_states!(agent)
            prev_action = 1
        else
            state = next_state
        end
    end
    
    # 如果还有未完成的episode，也加入
    if !isempty(current_episode[:states])
        push!(episodes, current_episode)
        push!(episode_rewards, current_ep_reward)
    end
    
    # 转换为序列数据
    trajectory_data = process_episodes_to_sequences_parallel(agent, episodes)
    
    return trajectory_data, mean(episode_rewards)
end

# === 9. 并行处理episodes到序列 ===
function process_episodes_to_sequences_parallel(agent::PPORNNAgent, episodes)
    n_episodes = length(episodes)
    
    if !agent.use_threading || n_episodes < agent.n_envs
        return process_episodes_to_sequences_single(agent, episodes)
    end
    
    # 将episodes分配给不同线程
    episodes_per_thread = div(n_episodes, nthreads())
    thread_results = Vector{Vector{Any}}(undef, nthreads())
    
    @threads for t in 1:nthreads()
        start_idx = (t-1) * episodes_per_thread + 1
        end_idx = t == nthreads() ? n_episodes : t * episodes_per_thread
        
        if start_idx <= n_episodes
            thread_episodes = episodes[start_idx:min(end_idx, n_episodes)]
            thread_results[t] = process_episodes_to_sequences_single(agent, thread_episodes)
        else
            thread_results[t] = []
        end
    end
    
    # 合并结果
    return vcat(filter(!isempty, thread_results)...)
end

# === 10. 单线程处理episodes到序列 ===
function process_episodes_to_sequences_single(agent::PPORNNAgent, episodes)
    all_sequences = []
    
    for episode in episodes
        ep_length = length(episode[:states])
        
        # 计算该episode的GAE
        advantages, returns = compute_gae_episode(agent, episode)
        
        # 将episode分割成固定长度的序列
        for start_idx in 1:agent.sequence_length:ep_length
            end_idx = min(start_idx + agent.sequence_length - 1, ep_length)
            seq_length = end_idx - start_idx + 1
            
            # 提取序列数据
            states_seq = episode[:states][start_idx:end_idx]
            actions_seq = episode[:actions][start_idx:end_idx]
            prev_actions_seq = episode[:prev_actions][start_idx:end_idx]
            old_log_probs_seq = episode[:old_log_probs][start_idx:end_idx]
            advantages_seq = advantages[start_idx:end_idx]
            returns_seq = returns[start_idx:end_idx]
            
            # 填充到固定长度
            while length(states_seq) < agent.sequence_length
                push!(states_seq, states_seq[end])
                push!(actions_seq, actions_seq[end])
                push!(prev_actions_seq, prev_actions_seq[end])
                push!(old_log_probs_seq, 0.0f0)
                push!(advantages_seq, 0.0f0)
                push!(returns_seq, 0.0f0)
            end
            
            # 构建动作-观察输入序列
            action_obs_inputs = []
            for i in 1:agent.sequence_length
                input_vec = build_action_obs_input(states_seq[i], prev_actions_seq[i], agent)
                push!(action_obs_inputs, input_vec)
            end
            action_obs_matrix = hcat(action_obs_inputs...)
            
            # 创建mask来标记哪些是真实数据
            mask = vcat(ones(Bool, seq_length), zeros(Bool, agent.sequence_length - seq_length))
            
            sequence = Dict(
                :action_obs_inputs => action_obs_matrix,
                :actions => actions_seq,
                :old_log_probs => old_log_probs_seq,
                :advantages => advantages_seq,
                :returns => returns_seq,
                :mask => mask
            )
            
            push!(all_sequences, sequence)
        end
    end
    
    return all_sequences
end

# === 11. 计算单个episode的GAE ===
function compute_gae_episode(agent::PPORNNAgent, episode)
    rewards = episode[:rewards]
    values = episode[:values]
    dones = episode[:dones]
    
    advantages = zeros(Float32, length(rewards))
    last_advantage = 0.0f0
    
    # 反向计算
    for t in reverse(1:length(rewards))
        if t == length(rewards) || dones[t]
            delta = rewards[t] - values[t]
            last_advantage = delta
        else
            delta = rewards[t] + agent.γ * values[t+1] - values[t]
            last_advantage = delta + agent.γ * agent.λ * last_advantage
        end
        advantages[t] = last_advantage
    end
    
    # 标准化优势
    if length(advantages) > 1
        advantages = (advantages .- mean(advantages)) ./ (std(advantages) + 1f-8)
    end
    returns = advantages .+ values
    
    return advantages, returns
end

# === 12. 并行策略更新 ===
function update_policy!(agent::PPORNNAgent, sequences)
    indices = shuffle(1:length(sequences))
    policy_losses = Float32[]
    
    for _ in 1:agent.n_epochs
        n_batches = ceil(Int, length(indices)/agent.minibatch_size)
        batch_losses = Vector{Float32}(undef, n_batches)
        batch_counter = Threads.Atomic{Int}(1)
        
        @threads for _ in 1:min(nthreads(), n_batches)
            i = Threads.atomic_add!(batch_counter, 1)
            if i <= n_batches
                start_idx = (i-1)*agent.minibatch_size + 1
                end_idx = min(i*agent.minibatch_size, length(indices))
                batch_indices = indices[start_idx:end_idx]
                batch_losses[i] = compute_policy_batch_loss(agent, sequences, batch_indices)
            end
        end
        
        append!(policy_losses, batch_losses)
    end
    
    return mean(policy_losses)
end

# === 13. 计算单个策略批次损失 ===
function compute_policy_batch_loss(agent::PPORNNAgent, sequences, batch_indices)
    loss, grads = Flux.withgradient(agent.policy_net) do model
        batch_loss = 0.0f0
        valid_samples = 0
        
        for seq_idx in batch_indices
            # 重置隐藏状态
            Flux.reset!(model)
            
            seq = sequences[seq_idx]
            action_obs_inputs = agent.device(seq[:action_obs_inputs])
            actions = seq[:actions]
            old_log_probs = agent.device(seq[:old_log_probs])
            advantages = agent.device(seq[:advantages])
            mask = seq[:mask]
            
            # 前向传播整个序列
            logits_seq = model(action_obs_inputs)
            
            # 计算每个时间步的损失
            for t in 1:agent.sequence_length
                if mask[t]
                    logits = logits_seq[:, t:t]
                    
                    # Use logsoftmax for numerical stability
                    log_probs = logsoftmax(logits)
                    
                    # 计算新的log概率 (more numerically stable)
                    onehot_action = onehotbatch([actions[t]], 1:length(agent.action_space)) |> agent.device
                    new_log_prob = sum(log_probs .* onehot_action)
                    
                    # PPO损失 (with clipping to prevent extreme values)
                    ratio = exp(clamp(new_log_prob - old_log_probs[t], -10.0f0, 10.0f0))
                    surr1 = ratio * advantages[t]
                    surr2 = clamp(ratio, 1f0 - agent.ϵ, 1f0 + agent.ϵ) * advantages[t]
                    policy_loss = -min(surr1, surr2)
                    
                    # 熵损失 (more numerically stable)
                    probs = softmax(logits)
                    entropy = -sum(probs .* log.(probs .+ 1f-8))
                    
                    batch_loss += policy_loss - agent.ent_coef * entropy
                    valid_samples += 1
                end
            end
        end
        
        batch_loss / max(valid_samples, 1)
    end
    
    # 梯度裁剪和更新
    if agent.clip_grads
        grads = clip_gradients(grads, agent.clip_value)
    end
    
    Flux.update!(agent.optimizer_policy, agent.policy_net, grads[1])
    return loss
end

# === 14. 并行价值函数更新 ===
function update_value!(agent::PPORNNAgent, sequences)
    indices = shuffle(1:length(sequences))
    value_losses = Float32[]
    
    for _ in 1:agent.n_epochs
        epoch_losses = Float32[]
        lk = ReentrantLock()
        
        @threads for i in 1:agent.minibatch_size:length(indices)
            batch_indices = indices[i:min(i+agent.minibatch_size-1, end)]
            batch_loss = compute_value_batch_loss(agent, sequences, batch_indices)
            lock(lk) do
                push!(epoch_losses, batch_loss)
            end
        end
        
        append!(value_losses, epoch_losses)
    end
    
    return mean(value_losses)
end

# === 15. 计算单个价值批次损失 ===
function compute_value_batch_loss(agent::PPORNNAgent, sequences, batch_indices)
    loss, grads = Flux.withgradient(agent.value_net) do model
        batch_loss = 0.0f0
        valid_samples = 0
        
        for seq_idx in batch_indices
            # 重置隐藏状态
            Flux.reset!(model)
            
            seq = sequences[seq_idx]
            action_obs_inputs = agent.device(seq[:action_obs_inputs])
            returns = agent.device(seq[:returns])
            mask = seq[:mask]
            
            # 前向传播整个序列
            values_pred = model(action_obs_inputs)
            
            # 计算每个时间步的损失
            for t in 1:agent.sequence_length
                if mask[t]
                    mse_loss = (values_pred[1, t] - returns[t])^2
                    batch_loss += mse_loss
                    valid_samples += 1
                end
            end
        end
        
        batch_loss / max(valid_samples, 1)
    end
    
    # 梯度裁剪
    if agent.clip_grads
        grads = clip_gradients(grads, agent.clip_value)
    end
    
    Flux.update!(agent.optimizer_value, agent.value_net, grads[1])
    return loss
end

# === 16. 梯度裁剪函数 ===
function clip_gradients(grads, clip_value)
    # 获取所有参数的梯度
    all_grads = []
    Flux.fmap(grads) do x
        if x isa AbstractArray
            push!(all_grads, x)
        end
        return x
    end
    
    if isempty(all_grads)
        return grads
    end
    
    # 计算梯度范数
    grad_norm = sqrt(sum(sum(abs2, g) for g in all_grads))
    
    # 计算缩放因子
    scale = min(1.0f0, clip_value / (grad_norm + 1f-6))
    
    # 应用缩放
    return Flux.fmap(g -> g isa AbstractArray ? g .* scale : g, grads)
end

# === 17. 训练循环 ===
function train!(env_or_constructor, agent::PPORNNAgent, num_updates::Int; eval_interval=100, verbose=true)
    all_rewards = Float32[]
    policy_losses = Float32[]
    value_losses = Float32[]
    eval_scores = Float32[]
    
    # 检查是否使用并行收集
    use_parallel = agent.use_threading && isa(env_or_constructor, Function)
    
    @showprogress for update in 1:num_updates
        # collect trajectories
        if use_parallel
            sequences, avg_reward = collect_trajectories_parallel(env_or_constructor, agent, agent.batch_size)
        else
            sequences, avg_reward = collect_trajectories(env_or_constructor, agent, agent.batch_size)
        end
        
        # update network
        policy_loss = update_policy!(agent, sequences)
        value_loss = update_value!(agent, sequences)
        
        # log results
        push!(all_rewards, avg_reward)
        push!(policy_losses, policy_loss)
        push!(value_losses, value_loss)
        
        # evaluation
        if update % eval_interval == 0
            eval_env = use_parallel ? env_or_constructor() : env_or_constructor
            eval_score = evaluate(eval_env, agent)
            push!(eval_scores, eval_score)
            
            if verbose
                println("Update $update | ",
                      "Threads: $(nthreads()) | ",
                      "Reward: $(round(avg_reward, digits=2)) | ",
                      "Policy Loss: $(round(policy_loss, digits=4)) | ",
                      "Value Loss: $(round(value_loss, digits=4)) | ",
                      "Eval: $(round(eval_score, digits=2))")
            end
        end
        
        # decay entropy coefficient
        agent.ent_coef *= 0.995f0
    end
    
    return (all_rewards, policy_losses, value_losses, eval_scores)
end

# === Evaluation (disounted accumlated rewards) ===
function evaluate(env, agent::PPORNNAgent; num_episodes=50, max_steps=nothing)
    if max_steps === nothing
        max_steps = agent.max_episode_length
    end
    gamma = discount(env.model)
    
    total_reward = 0.0f0
    
    for _ in 1:num_episodes
        state = reset!(env)
        reset_hidden_states!(agent)
        ep_reward = 0.0f0
        prev_action = 1
        
        for i in 0:max_steps
            # build input (action + observation)
            action_obs_input = build_action_obs_input(state, prev_action, agent)
            action_obs_input_gpu = agent.device(reshape(action_obs_input, :, 1))
            
            # select action
            logits = agent.policy_net(action_obs_input_gpu)
            action = argmax(cpu(vec(logits)))
            
            state, reward, done = step!(env, action)
            ep_reward += (gamma^i)*reward
            prev_action = action
            
            done && break
        end
        
        total_reward += ep_reward
    end
    
    return total_reward / num_episodes
end