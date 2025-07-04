<!-- # README

## Installation Instructions

To run this project, you need to install the following dependencies:

```bash
pip install gym torch sb3_contrib
```

### Installing `gym_pomdp`
Since `gym_pomdp` is not available via pip, you need to install it manually:

```bash
git clone https://github.com/d3sm0/gym_pomdp.git
cd gym_pomdp
python setup.py install
```


### Running the Project
Once the dependencies are installed, you can run your script as follows:

```bash
python sb3_RecurrentPPO.py
```
 -->


# README

## INSTALL JULIA

## INSTALL IJULIA

## INSTALL POMDPs.jl

## Training Tips
- First try PPO-RNN
- PPO-RNN seems better with large batch size
- DRQN seems better with very small batch size (4-32)