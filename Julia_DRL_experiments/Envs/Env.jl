using POMDPModels
using POMDPs
using RockSample


# Define a simple CartPole environment
mutable struct Env
    state::Any
    model
    bool_full_observability::Bool
    bool_discrete_actions::Bool
    # bool_discrete_observations::Bool
end

function Env(pomdp, bool_full_observability::Bool)
    bool_discrete_actions = false
    # bool_discrete_observations = false 
    action_set = []
    if typeof(actions(pomdp)) == UnitRange{Int64}
        bool_discrete_actions = true
    end

    # if typeof(observations(pomdp)) == UnitRange{Int64}
    #     bool_discrete_observations = true
    # end

    env = Env(rand(initialstate(pomdp)), pomdp, bool_full_observability, bool_discrete_actions)
    # env = Env(rand(initialstate(pomdp)), pomdp, bool_full_observability, bool_discrete_actions, bool_discrete_observations)
    return env
end

function reset!(env::Env)
    env.state = rand(initialstate(env.model))
    s_vec = convert_s(Vector{Float32}, env.state, env.model)

    if !env.bool_full_observability
        a = rand(actions(env.model))
        sp, o, r = @gen(:sp, :o, :r)(env.model, env.state, a)
        o_vec = convert_o(Vector{Float32}, o, env.model)
        for i in 1:length(o_vec)
            o_vec[i] = 0.0f0
        end
        return o_vec
    end

    return s_vec
end

function step!(env::Env, action_index)
    a = actions(env.model)[action_index]
    sp, o, r = @gen(:sp, :o, :r)(env.model, env.state, a)
    sp_vec = convert_s(Vector{Float32}, sp, env.model)
    o_vec = convert_o(Vector{Float32}, o, env.model)
    done = isterminal(env.model, sp)
    env.state = sp

    if env.bool_full_observability
        return sp_vec, r, done
    else
        return o_vec, r, done
    end
end

function GetStateDim(env::Env)
    b0 = initialstate(env.model)
    s = rand(b0)
    s_vec = convert_s(Vector{Float32}, s, env.model)

    return length(s_vec)
end

function GetObsDim(env::Env)
    # For RockSamplePOMDP with partial observability, observations are converted to 3-dimensional vectors
    if env.model isa RockSamplePOMDP && !env.bool_full_observability
        return 3
    else
        # Fallback for other cases
        b0 = initialstate(env.model)
        action_space = actions(env.model)
        sp, o, r = @gen(:sp, :o, :r)(env.model, rand(b0), rand(action_space))
        o_vec = convert_o(Vector{Float32}, o, env.model)
        return length(o_vec)
    end
end

function GetStateSpace(env::Env)
    return states(env.model)
end

function GetObsSpace(env::Env)
    return observations(env.model)
end
    
function GetActionSpace(env::Env)
    return actions(env.model)
end

function GetDiscount(env::Env)
    return discount(env.model)
end

