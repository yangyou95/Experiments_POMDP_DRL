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

        "--layer-size"
        help = "Size of the hidden dense layer"
        arg_type = Int
        default = 64

        "--rnn-hidden-size"
        help = "Size of the RNN hidden state"
        arg_type = Int
        default = 64

        "--batch-size"
        help = "Batch size for training"
        arg_type = Int
        default = 4096

        "--hidden-dim"
        help = "Size of the hidden dense layer"
        arg_type = Int
        default = 64

        "--sequence-length"
        help = "Length of RNN training sequences"
        arg_type = Int
        default = 32

        "--max-episode-length"
        help = "Maximum episode length"
        arg_type = Int
        default = 200

        "--gamma"
        help = "Discount factor"
        arg_type = Float64
        default = 0.99

        "--lambda"
        help = "GAE λ parameter"
        arg_type = Float64
        default = 0.95

        "--epsilon"
        help = "PPO clip ε"
        arg_type = Float64
        default = 0.2

        "--ent-coef"
        help = "Entropy coefficient"
        arg_type = Float64
        default = 0.01

        "--n-epochs"
        help = "PPO epochs per update"
        arg_type = Int
        default = 4

        "--minibatch-size"
        help = "Minibatch size"
        arg_type = Int
        default = 64

        "--lr-policy"
        help = "Policy learning rate"
        arg_type = Float64
        default = 3e-4

        "--lr-value"
        help = "Value learning rate"
        arg_type = Float64
        default = 1e-4

        "--num-updates"
        help = "Number of training updates"
        arg_type = Int
        default = 10000

    end
    return ArgParse.parse_args(s)
end

const ARGS_PARSED = parse_args()
#Print the parsed arguments
println("Parsed arguments: $(ARGS_PARSED)")
const ROWS = ARGS_PARSED["rows"]
const COLS = ARGS_PARSED["cols"]
const LAYER_SIZE = ARGS_PARSED["layer-size"]
const RNN_HIDDEN_SIZE = ARGS_PARSED["rnn-hidden-size"]
const BOOL_FULL_OBSERVABILITY = false#ARGS_PARSED["full-observability"]


const HIDDEN_DIM          = ARGS_PARSED["hidden-dim"]
const SEQUENCE_LENGTH     = ARGS_PARSED["sequence-length"]
const MAX_EPISODE_LENGTH  = ARGS_PARSED["max-episode-length"]
const GAMMA               = Float32(ARGS_PARSED["gamma"])
const LAMBDA              = Float32(ARGS_PARSED["lambda"])
const EPSILON             = Float32(ARGS_PARSED["epsilon"])
const ENT_COEF            = Float32(ARGS_PARSED["ent-coef"])
const N_EPOCHS            = ARGS_PARSED["n-epochs"]
const BATCH_SIZE          = ARGS_PARSED["batch-size"]
const MINIBATCH_SIZE      = ARGS_PARSED["minibatch-size"]
const LR_POLICY           = ARGS_PARSED["lr-policy"]
const LR_VALUE            = ARGS_PARSED["lr-value"]
const NUM_UPDATES         = ARGS_PARSED["num-updates"]


# using CUDA
include("../Envs/Env.jl")
include("../Algorithms/PPO-RNN.jl")
using RockSample

pomdp = RockSamplePOMDP(ROWS, COLS)



training_episodes = 10000
batch_size = ARGS_PARSED["batch-size"] 
#4096






pomdp_name = "RS$(ROWS)$(COLS)_Layer$(LAYER_SIZE)_RNN$(RNN_HIDDEN_SIZE)_BS$(batch_size)"
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
layer_size = LAYER_SIZE
rnn_hidden_size = RNN_HIDDEN_SIZE
gamma = discount(pomdp)



# if want to use gpu, need to uncomment the below line, and use device=Flux.gpu
# using CUDA

agent = PPORNNAgent(action_space, state_dim;
    hidden_dim=HIDDEN_DIM,
    rnn_hidden_size=RNN_HIDDEN_SIZE,
    sequence_length=SEQUENCE_LENGTH,
    max_episode_length=MAX_EPISODE_LENGTH,
    γ=GAMMA,
    λ=LAMBDA,
    ϵ=EPSILON,
    ent_coef=ENT_COEF,
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    minibatch_size=MINIBATCH_SIZE,
    lr_policy=LR_POLICY,
    lr_value=LR_VALUE,
    device=Flux.cpu)


    # 训练
rewards, losses, evals = train!(create_env, agent, NUM_UPDATES, run_name=pomdp_name)




evaluate(env, agent; num_episodes=10000, max_steps=100) 

