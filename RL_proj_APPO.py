import flow.networks as networks
from flow.networks import RingNetwork
from flow.core.params import NetParams, InitialConfig # input parameter classes to the network class
from flow.core.params import VehicleParams # input parameter classes to the network class
from flow.core.params import SumoParams, EnvParams
from flow.networks.ring import ADDITIONAL_NET_PARAMS # network-specific parameters
from flow.controllers import IDMController, ContinuousRouter # vehicles dynamics models
from flow.controllers import RLController
#from flow.envs import WaveAttenuationEnv
from flow.envs import WaveAttenuationPOEnv
import flow.envs as flowenvs

import json
import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

network_name = RingNetwork # ring road network class
name = "training_example" # name of the network
net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS)
initial_config = InitialConfig(spacing="uniform", perturbation=1) # initial configuration to vehicles

vehicles = VehicleParams()
vehicles.add("human",
             acceleration_controller=(IDMController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=21)

vehicles.add(veh_id="rl",
             acceleration_controller=(RLController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=1)

sim_params = SumoParams(sim_step=0.1, render=False)
# Define horizon as a variable to ensure consistent use across notebook
HORIZON=600   # horizon represents the time that the whole process operates.
env_params = EnvParams(
    
    horizon=HORIZON, # the duration of one episode (during which the RL-agent acquire data).
    additional_params={
        # maximum acceleration of autonomous vehicles
        "max_accel": 5,
        # maximum deceleration of autonomous vehicles
        "max_decel": -7,
        # bounds on the ranges of ring road lengths the autonomous vehicle 
        # is trained on
        "ring_length": [220, 270],  # the length of the ring is 230m
    },
)

env_name = WaveAttenuationPOEnv       # when use PPO Algo

flow_params = dict(
    exp_tag=name, # name of the experiment
    env_name=env_name, # name of the flow environment the experiment is running on
    network=network_name, # name of the network class the experiment uses
    simulator='traci', # simulator that is used by the experiment
    sim=sim_params, # simulation-related parameters
    env=env_params, # environment related parameters
    net=net_params, # network-related parameters
    veh=vehicles, # vehicles to be placed in the network at the start of a rollout
    initial=initial_config)
    

N_CPUS = 2  # number of parallel workers
N_ROLLOUTS = 1  
# In Each iteration, samples are aggregated from multiple rollouts into a batch and the resulting gradient is used to update the policy. This process of performing rollouts to collect batches of samples followed by updating the policy is repeated until the average cumulative reward has stabilized, at which point we say that training has converged.

ray.init(num_cpus=N_CPUS)


alg_run = "APPO"

agent_cls = get_agent_class(alg_run)
config = agent_cls._default_config.copy()
config["num_workers"] = N_CPUS - 1  # number of parallel workers
config["train_batch_size"] = HORIZON * N_ROLLOUTS  # batch size
config["gamma"] = 0.999  # discount rate
config["model"].update({"fcnet_hiddens": [16, 16]})  # size of hidden layers in network
config["use_gae"] = True  # using generalized advantage estimation
config["lambda"] = 0.97  
#config["sgd_minibatch_size"] = min(16 * 1024, config["train_batch_size"])
config["kl_target"] = 0.02  # target KL divergence
config["num_sgd_iter"] = 10  # number of SGD iterations
config["horizon"] = HORIZON  # rollout horizon

# save the flow params for replay
flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True,
                       indent=4)  # generating a string version of flow_params
config['env_config']['flow_params'] = flow_json  # adding the flow_params to config dict
config['env_config']['run'] = alg_run

# Call the utility function make_create_env to be able to 
# register the Flow env for this experiment
create_env, gym_name = make_create_env(params=flow_params, version=0)

# Register as rllib env with Gym
register_env(gym_name, create_env)

trials = run_experiments({
    flow_params["exp_tag"]: {
        "run": alg_run,
        "env": gym_name,
        "config": {
            **config
        },
        "checkpoint_freq": 1,  # number of iterations between checkpoints
        "checkpoint_at_end": True,  # generate a checkpoint at the end
        "max_failures": 999,
        "stop": {  # stopping conditions
            "training_iteration": 1000,  # number of iterations to stop after
        },
    },
})
