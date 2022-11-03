"""Visualizer for rllib experiments.

Attributes
----------
EXAMPLE_USAGE : str
    Example call to the function, which is
    ::

        python ./visualizer_rllib.py /tmp/ray/result_dir 1

parser : ArgumentParser
    Command-line argument parser
"""

import argparse
import gym
import numpy as np
import os
import sys
import time
import csv
import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env

from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_config
from flow.utils.rllib import get_rllib_pkl


EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /ray_results/experiment_dir/result_dir 1

Here the arguments are:
1 - the path to the simulation results
2 - the number of the checkpoint
"""


def visualizer_rllib(args):
    """Visualizer for RLlib experiments.

    This function takes args (see function create_parser below for
    more detailed information on what information can be fed to this
    visualizer), and renders the experiment associated with it.
    """
    result_dir = args.result_dir if args.result_dir[-1] != '/' \
        else args.result_dir[:-1]

    config = get_rllib_config(result_dir)

    # check if we have a multiagent environment but in a
    # backwards compatible way
    if config.get('multiagent', {}).get('policies', None):
        multiagent = True
        pkl = get_rllib_pkl(result_dir)
        config['multiagent'] = pkl['multiagent']
    else:
        multiagent = False

    # Run on only one cpu for rendering purposes
    config['num_workers'] = 0

    flow_params = get_flow_params(config)

    # hack for old pkl files
    # TODO(ev) remove eventually
    sim_params = flow_params['sim']
    setattr(sim_params, 'num_clients', 1)

    # for hacks for old pkl files TODO: remove eventually
    if not hasattr(sim_params, 'use_ballistic'):
        sim_params.use_ballistic = False

    # Determine agent and checkpoint
    config_run = config['env_config']['run'] if 'run' in config['env_config'] \
        else None
    if args.run and config_run:
        if args.run != config_run:
            print('visualizer_rllib.py: error: run argument '
                  + '\'{}\' passed in '.format(args.run)
                  + 'differs from the one stored in params.json '
                  + '\'{}\''.format(config_run))
            sys.exit(1)
    if args.run:
        agent_cls = get_agent_class(args.run)
    elif config_run:
        agent_cls = get_agent_class(config_run)
    else:
        print('visualizer_rllib.py: error: could not find flow parameter '
              '\'run\' in params.json, '
              'add argument --run to provide the algorithm or model used '
              'to train the results\n e.g. '
              'python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO')
        sys.exit(1)

    sim_params.restart_instance = True
    dir_path = os.path.dirname(os.path.realpath(__file__))
    emission_path = '{0}/test_time_rollout/'.format(dir_path)
    sim_params.emission_path = emission_path if args.gen_emission else None

    # pick your rendering mode
    if args.render_mode == 'sumo_web3d':
        sim_params.num_clients = 2
        sim_params.render = False
    elif args.render_mode == 'drgb':
        sim_params.render = 'drgb'
        sim_params.pxpm = 4
    elif args.render_mode == 'sumo_gui':
        sim_params.render = False  # will be set to True below
    elif args.render_mode == 'no_render':
        sim_params.render = False
    if args.save_render:
        if args.render_mode != 'sumo_gui':
            sim_params.render = 'drgb'
            sim_params.pxpm = 4
        sim_params.save_render = True

    # Create and register a gym+rllib env
    create_env, env_name = make_create_env(params=flow_params, version=0)
    register_env(env_name, create_env)

    # check if the environment is a single or multiagent environment, and
    # get the right address accordingly
    # single_agent_envs = [env for env in dir(flow.envs)
    #                      if not env.startswith('__')]

    # if flow_params['env_name'] in single_agent_envs:
    #     env_loc = 'flow.envs'
    # else:
    #     env_loc = 'flow.envs.multiagent'

    # Start the environment with the gui turned on and a path for the
    # emission file
    env_params = flow_params['env']
    env_params.restart_instance = False
    if args.evaluate:
        env_params.evaluate = True

    # lower the horizon if testing
    if args.horizon:
        config['horizon'] = args.horizon
        env_params.horizon = args.horizon

    # create the agent that will be used to compute the actions
    agent = agent_cls(env=env_name, config=config)
    checkpoint = result_dir + '/checkpoint_' + args.checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + args.checkpoint_num
    agent.restore(checkpoint)

    if hasattr(agent, "local_evaluator") and \
            os.environ.get("TEST_FLAG") != 'True':
        env = agent.local_evaluator.env
    else:
        env = gym.make(env_name)
#we dont want the sumo-gui, so citation this two sentences.
    #if args.render_mode == 'sumo_gui':
    #    env.sim_params.render = True  # set to True after initializing agent and env

    if multiagent:
        rets = {}
        # map the agent id to its policy
        policy_map_fn = config['multiagent']['policy_mapping_fn']
        for key in config['multiagent']['policies'].keys():
            rets[key] = []
    else:
        rets = []

    if config['model']['use_lstm']:
        use_lstm = True
        if multiagent:
            state_init = {}
            # map the agent id to its policy
            policy_map_fn = config['multiagent']['policy_mapping_fn']
            size = config['model']['lstm_cell_size']
            for key in config['multiagent']['policies'].keys():
                state_init[key] = [np.zeros(size, np.float32),
                                   np.zeros(size, np.float32)]
        else:
            state_init = [
                np.zeros(config['model']['lstm_cell_size'], np.float32),
                np.zeros(config['model']['lstm_cell_size'], np.float32)
            ]
    else:
        use_lstm = False

    # if restart_instance, don't restart here because env.reset will restart later
    if not sim_params.restart_instance:
        env.restart_simulation(sim_params=sim_params, render=sim_params.render)

    # Simulate and collect metrics
    final_outflows = []
    final_inflows = []
    mean_speed = []
    std_speed = []
    mean_time_headway = []
    mean_vel_per_rollouts = []
    min_vel_per_rollouts = []
    mean_head_per_rollouts = []
    min_head_per_rollouts = []
    pos_array = []
    allcar_pos_array = []
    labels = ['id', 'position']
    min_head_all_merging = []
    mean_head_all_merging = []
    mean_speed_all_merging = []
    
    for i in range(args.num_rollouts):  # num_rollouts = 1
        mean_vel_per_horizon = []
        min_vel_per_horizon = []
        mean_head_per_horizon = []
        min_head_per_horizon = []
        min_head_per_rollout_merging = []
        mean_speed_per_rollout_merging = []
        percentage_low_THW = []
        
        collsion_num = 0
        state = env.reset()
        if multiagent:
            ret = {key: [0] for key in rets.keys()}
        else:
            ret = 0
        for _ in range(env_params.horizon): #每秒所有车的value
            
            min_head_per_horizon_merging=[]
            speed_per_horizon_merging=[]
            vehicles = env.unwrapped.k.vehicle

            # choose the RL_controller and their leader vehicles
            rl_ids = vehicles.get_rl_ids()
            for r in rl_ids: # collision rate is how many times per second
                h = vehicles.get_headway(r)
                if h < 0 or h ==0:
                    collsion_num += 1
            
            # for all cars:
            speeds = vehicles.get_speed(vehicles.get_ids())
            time_headway = vehicles.get_headway(vehicles.get_ids())  #computing distance_headway in each horizon
            # all cars' id in 1s:
            id_for_all = vehicles.get_ids()
            
            for i, id in enumerate(id_for_all):
                pos_all = vehicles.get_position(id)
                print("id: ", id, " position: ", pos_all)
                pos_dict_all = {'id': str(id), 'position': str(pos_all)}
                allcar_pos_array.append(pos_dict_all)

                if pos_all >= 400: #and len(id)<=9
                    time_headway_mergingZone = vehicles.get_headway(id)
                    min_head_per_horizon_merging.append(time_headway_mergingZone)
                    speed_merging_zone = vehicles.get_speed(id)
                    speed_per_horizon_merging.append(speed_merging_zone)
                    if i+1 < len(id_for_all):
                        if len(id_for_all[i+1])<=9:
                            time_headway_mergingZone_post1 = vehicles.get_headway(id_for_all[i+1])
                            min_head_per_horizon_merging.append(time_headway_mergingZone_post1)
                            speed_merging_zone_post1 = vehicles.get_speed(id_for_all[i+1])
                            speed_per_horizon_merging.append(speed_merging_zone_post1)
                    if i+2 < len(id_for_all):
                        if len(id_for_all[i+2])<=9:
                            time_headway_mergingZone_post2 = vehicles.get_headway(id_for_all[i+2])
                            min_head_per_horizon_merging.append(time_headway_mergingZone_post2)
                            speed_merging_zone_post2 = vehicles.get_speed(id_for_all[i+2])
                            speed_per_horizon_merging.append(speed_merging_zone_post2)
                    if i+3 < len(id_for_all):
                        if len(id_for_all[i+3])<=9:
                            time_headway_mergingZone_post3 = vehicles.get_headway(id_for_all[i+3])
                            min_head_per_horizon_merging.append(time_headway_mergingZone_post3)
                            speed_merging_zone_post3 = vehicles.get_speed(id_for_all[i+3])
                            speed_per_horizon_merging.append(speed_merging_zone_post3)
            
            
            # only include non-empty speeds
            if speeds:
                mean_vel_per_horizon.append(np.mean(speeds))
                min_vel_per_horizon.append(np.min(speeds))
            
            if time_headway:
                
                lowTHW = np.sum(np.array(time_headway) < 1) # threshold = 1s
                lowTHW_percent = lowTHW / len(np.array(time_headway))
                percentage_low_THW.append(lowTHW_percent)
                if np.mean(time_headway) > 0:
                    mean_head_per_horizon.append(np.mean(time_headway))
                if np.min(time_headway) > 0 :
                    min_head_per_horizon.append(np.min(time_headway))  # minimum distance headway of all cars in one horizon
                    
            #getting the merge vehicle:
            merge_ids = vehicles.get_ids_by_edge("inflow_merge")     # all the merging vehicles
            # for merge vehicles:
            merge_speeds = []
            merge_timeheadways = []
            
            for merge_id in merge_ids:
                pos = vehicles.get_position(merge_id)
                #print("id: ", merge_id, " position: ", pos)
                pos_dict = {'id': str(merge_id), 'position': str(pos)}
                pos_array.append(pos_dict)
                merge_speed = vehicles.get_speed(merge_id)
                merge_timeheadway = vehicles.get_headway(merge_id)   # each merging vehicle's time headway of each horizon
               
                min_head_per_horizon_merging.append(merge_timeheadway)  # store each merging vehicle's time headway of one horizon in list
                speed_per_horizon_merging.append(merge_speed)
                
                if merge_speed:
                    merge_speeds.append(merge_speed)
                if merge_timeheadway:
                    merge_timeheadways.append(merge_timeheadway)
                    #print(merge_timeheadway)
            if min_head_per_horizon_merging:
                # min_head_per_rollout_merging.append(min(min_head_per_horizon_merging))
                min_head_per_rollout_merging.append(np.min(min_head_per_horizon_merging)) # min distance headway of merging vehicles in one horizon
            if speed_per_horizon_merging:
                mean_speed_per_rollout_merging.append(np.mean(speed_per_horizon_merging))
                
            if multiagent:
                action = {}
                for agent_id in state.keys():
                    if use_lstm:
                        action[agent_id], state_init[agent_id], logits = \
                            agent.compute_action(
                            state[agent_id], state=state_init[agent_id],
                            policy_id=policy_map_fn(agent_id))
                    else:
                        action[agent_id] = agent.compute_action(
                            state[agent_id], policy_id=policy_map_fn(agent_id))
            else:
                action = agent.compute_action(state)
            state, reward, done, _ = env.step(action)
            if multiagent:
                for actor, rew in reward.items():
                    ret[policy_map_fn(actor)][0] += rew
            else:
                ret += reward
            if multiagent and done['__all__']:
                break
            if not multiagent and done:
                break
        
        min_head_all_merging.append(min(min_head_per_rollout_merging))  # min to horizon, min to cars
        mean_head_all_merging.append(np.mean(min_head_per_rollout_merging)) # mean to horizon, min to cars
        mean_speed_all_merging.append(np.mean(mean_speed_per_rollout_merging))
                
        if multiagent:
            for key in rets.keys():
                rets[key].append(ret[key])
        else:
            rets.append(ret)
        outflow = vehicles.get_outflow_rate(500)
        final_outflows.append(outflow)
        inflow = vehicles.get_inflow_rate(500)
        final_inflows.append(inflow)
        if np.all(np.array(final_inflows) > 1e-5):
            throughput_efficiency = [x / y for x, y in
                                     zip(final_outflows, final_inflows)]
        else:
            throughput_efficiency = [0] * len(final_inflows)

        # collision rate
        collsion_rate = collsion_num/env_params.horizon  # how many collision happens per second

        mean_vel_per_rollouts.append(np.mean(mean_vel_per_horizon))
        min_vel_per_rollouts.append(np.min(min_vel_per_horizon))
        mean_head_per_rollouts.append(np.mean(mean_head_per_horizon))
        #min_head_per_rollouts.append(np.min(min_head_per_horizon))
        min_head_per_rollouts.append(np.mean(min_head_per_horizon)) # min to cars, mean to all horizons (all cars on the highway)
        #print("min and mean headway of each rollout:\n")
        #print(mean_head_per_rollouts)
        #print(min_head_per_rollouts)
        #print(args.num_rollouts)
        #mean_speed.append(np.mean(vel))
        #std_speed.append(np.std(vel))
        #mean_time_headway.append(np.mean(head))   # timeheadway
        
        if multiagent:
            for agent_id, rew in rets.items():
                print('Round {}, Return: {} for agent {}'.format(
                    i, ret, agent_id))
        else:
            print('Round {}, Return: {}'.format(i, ret))
    '''       
    with open('/mnt/e/NorthwesternUniversity/Research/pos_dict_for_all_custom.csv', 'w') as f: 
        writer = csv.DictWriter(f, fieldnames=labels)
        writer.writeheader()
        for elem in allcar_pos_array:
            writer.writerow(elem)
    ''' 
        
    print('==== Summary of results ====')
    print("Return:")
    #print("THW you need:")
    #print('min_min')
    #print(min_head_all_merging)  # 在merging zone 的全局求min
    print('collision rate:', collsion_rate)
    print(collsion_rate)
    
    print('mean_min_headway_mergingzone')
    print(mean_head_all_merging)
    
    print("\nMean_min_headway_all")
    print(min_head_per_rollouts)
    print("\nMean_min_THeadway_all_cars, min (m/s):")
    print(np.mean(min_head_per_rollouts))
    
    print('mean speed you need:')
    print("\nSpeed, mean (m/s)_merging:")
    print(mean_speed_all_merging)
    
    print("mean speed of whole highway")
    print(np.mean(mean_vel_per_rollouts))
    
    
    #print(mean_speed)
    print(mean_time_headway)
    if multiagent:
        for agent_id, rew in rets.items():
            print('For agent', agent_id)
            print(rew)
            print('Average, std return: {}, {} for agent {}'.format(
                np.mean(rew), np.std(rew), agent_id))
    else:
        print(rets)
        print('Average, std: {}, {}'.format(
            np.mean(rets), np.std(rets)))

    #print("\nMean Speed of each rollout")
    #print(mean_vel_per_horizon)
    
    
    print("\nMin Speed of each rollout")
    print(min_vel_per_rollouts)
    print("\nSpeed, min (m/s):")
    print(np.mean(min_vel_per_rollouts))
    
    print("\nMean THeadway of each rollout")
    print(mean_head_per_rollouts)
    print("\nTHeadway, mean (m/s):")
    print(np.mean(mean_head_per_rollouts))
    
    #print("\nMin THeadway of each rollout")
    print("\nMean_min")
    print(min_head_per_rollouts)
    print("\nTHeadway, min (m/s):")
    print(np.mean(min_head_per_rollouts))
    
    print("\nAverage minimum THeadway value")
    print(np.mean(min_head_per_horizon))
    
    print("\npercent of low THW vehicle")
    print(np.mean(percentage_low_THW))
    
    #print('Average, std: {}, {}'.format(np.mean(mean_speed), np.std(
    #    mean_speed)))
    #print("\nSpeed, std (m/s):")
    #print(std_speed)
    #print('Average, std: {}, {}'.format(np.mean(std_speed), np.std(
    #    std_speed)))

    # Compute arrival rate of vehicles in the last 500 sec of the run
    #print("\nOutflows (veh/hr):")
    #print(final_outflows)
    #print('Average, std: {}, {}'.format(np.mean(final_outflows),
    #                                    np.std(final_outflows)))
    # Compute departure rate of vehicles in the last 500 sec of the run
    #print("Inflows (veh/hr):")
    #print(final_inflows)
    #print('Average, std: {}, {}'.format(np.mean(final_inflows),
    #                                    np.std(final_inflows)))
    # Compute throughput efficiency in the last 500 sec of the
    #print("Throughput efficiency (veh/hr):")
    #print(throughput_efficiency)
    #print('Average, std: {}, {}'.format(np.mean(throughput_efficiency),
    #                                    np.std(throughput_efficiency)))

    # terminate the environment
    env.unwrapped.terminate()

    # if prompted, convert the emission file into a csv file
    if args.gen_emission:
        time.sleep(0.1)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        emission_filename = '{0}-emission.xml'.format(env.network.name)

        emission_path = \
            '{0}/test_time_rollout/{1}'.format(dir_path, emission_filename)

        # convert the emission file into a csv file
        emission_to_csv(emission_path)

        # print the location of the emission csv file
        emission_path_csv = emission_path[:-4] + ".csv"
        print("\nGenerated emission file at " + emission_path_csv)

        # delete the .xml version of the emission file
        os.remove(emission_path)

    return mean_head_all_merging, np.mean(min_head_per_rollouts), np.mean(mean_vel_per_rollouts), collsion_rate


def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Evaluates a reinforcement learning agent '
                    'given a checkpoint.',
        epilog=EXAMPLE_USAGE)

    # required input parameters
    parser.add_argument(
        'result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')

    # optional input parameters
    parser.add_argument(
        '--run',
        type=str,
        help='The algorithm or model to train. This may refer to '
             'the name of a built-on algorithm (e.g. RLLib\'s DQN '
             'or PPO), or a user-defined trainable function or '
             'class registered in the tune registry. '
             'Required for results trained with flow-0.2.0 and before.')
    parser.add_argument(
        '--num_rollouts',
        type=int,
        default=1,
        help='The number of rollouts to visualize.')
    parser.add_argument(
        '--gen_emission',
        action='store_true',
        help='Specifies whether to generate an emission file from the '
             'simulation')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Specifies whether to use the \'evaluate\' reward '
             'for the environment.')
    parser.add_argument(
        '--render_mode',
        type=str,
        default='sumo_gui',
        help='Pick the render mode. Options include sumo_web3d, '
             'rgbd and sumo_gui')
    parser.add_argument(
        '--save_render',
        action='store_true',
        help='Saves a rendered video to a file. NOTE: Overrides render_mode '
             'with pyglet rendering.')
    parser.add_argument(
        '--horizon',
        type=int,
        help='Specifies the horizon.')
    return parser


if __name__ == '__main__':
    exp_times = 10	
    head_merging = []	
    head_highway = []		
    speed_highway = []	
    collision = []
    	
    ray.init(num_cpus=20)	
    for i in range(exp_times):	
        parser = create_parser()	
        args = parser.parse_args()	
        mean_min_head_merging, mean_min_head_highway, mean_speed_highway, collsion_rate = visualizer_rllib(args)	
        head_merging.append(mean_min_head_merging)	
        head_highway.append(mean_min_head_highway)		
        speed_highway.append(mean_speed_highway)
        collision.append(collsion_rate)	
        #ray.shutdown()	
    print('total in total')	
    print('average collision rate is: ', np.mean(collision))

    print('min merging THW of'+ str(exp_times) + 'exp:')	
    print(np.mean(head_merging))	
    print('min highway THW of '+ str(exp_times) + 'exp:')	
    print(np.mean(head_highway))		
    print('mean highway speed of '+ str(exp_times) + 'exp:')	
    print(np.mean(speed_highway))

