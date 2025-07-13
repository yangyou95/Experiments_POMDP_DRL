# using Pkg
# Pkg.add("Distributions")
# Pkg.add("POMDPModels")
# Pkg.add("POMDPs")
# Pkg.add("RockSample")
# Pkg.add("Flux")
# Pkg.add("Wandb")
# Pkg.add("ProgressMeter")
# Pkg.add("Zygote")
# using Pkg
# Pkg.add("LuxCUDA")
using ArgParse

# parse command-line arguments for grid size and observability
function parse_args()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--rows"
        help = "Number of rows in RockSample grid"
        arg_type = Int

        "--cols"
        help = "Number of columns in RockSample grid"
        arg_type = Int
    end
    return ArgParse.parse_args(s)
end

const ARGS_PARSED = parse_args()
#Print the parsed arguments
println("Parsed arguments: $(ARGS_PARSED)")
const ROWS = ARGS_PARSED["rows"]
const COLS = ARGS_PARSED["cols"]
const BOOL_FULL_OBSERVABILITY = false#ARGS_PARSED["full-observability"]
# using CUDA
include("../Envs/Env.jl")
include("../Algorithms/PPO-RNN.jl")
using RockSample

pomdp = RockSamplePOMDP(ROWS, COLS)
pomdp_name = "RS$(ROWS)$(COLS)"
bool_full_observability = BOOL_FULL_OBSERVABILITY
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
action_dim = length(action_space)
layer_size = 64
rnn_hidden_size = 64
gamma = discount(pomdp)
training_episodes = 10000
batch_size = 2048


# if want to use gpu, need to uncomment the below line, and use device=Flux.gpu
# using CUDA

agent = PPORNNAgent(action_space, action_dim, state_dim;
    hidden_dim=layer_size, 
    rnn_hidden_size=rnn_hidden_size, 
    batch_size=batch_size, 
    device=Flux.cpu) 


    # 训练
rewards, losses, evals = train!(create_env, agent, training_episodes, run_name=pomdp_name)




evaluate(env, agent; num_episodes=10000, max_steps=100) 

