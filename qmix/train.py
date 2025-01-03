import torch
import numpy as np
import os, sys

from data.replay_buffer import ReplayBuffer
from data.data_collector import DataCollector
from action.action import MultiAgentAction
from learner.q_learner import QLearner

from types import SimpleNamespace as NS

def train(config_dict):
    hid_size = config_dict["hid_size"]
    t_max = config_dict["t_max"]
    batch_size = config_dict["batch_size"]
    buffer_size = config_dict["buffer_size"]
    map_name = config_dict["map_name"]
    agent_name = config_dict["agent_name"]
    epsilon_start = config_dict["epsilon_start"]
    epsilon_finish = config_dict["epsilon_finish"]
    epsilon_time_length = config_dict["epsilon_time_length"]
    double_q = config_dict["double_q"]
    gamma = config_dict["gamma"]
    grad_norm_clip = config_dict["grad_norm_clip"]
    target_update_interval = config_dict["target_update_interval"]
    test_interval = config_dict["test_interval"]
    test_batch_size = config_dict["test_batch_size"]
    lr = config_dict["lr"]
    optim_alpha = config_dict["optim_alpha"]
    optim_eps = config_dict["optim_eps"]
    mix = config_dict["mix"]

    args = NS(**config_dict)

    data_collect = DataCollector(map_name, batch_size)
    env_info = data_collect.env.get_env_info()
    
    n_agents = env_info["n_agents"]
    n_actions = env_info["n_actions"]
    output_size = n_actions
    state_shape = env_info["state_shape"]
    max_seq_length = env_info["episode_limit"] + 1


    scheme = {
        "state" : {"group": False, "shape": env_info["state_shape"]},
        "obs" : {"group": True, "shape": env_info["obs_shape"]},
        "actions" : {"group": True, "shape": 1},
        "avail_actions" : {"group": True, "shape": env_info["n_actions"]},
        "reward" : {"group": False, "shape": 1}, 
        "terminated" : {"group": False, "shape": 1},
        "actions_onehot" : {"group": True, "shape": env_info["n_actions"]},
        "filled": {"group": False, "shape": 1},
    }

    print("scheme ... ")
    print(scheme)

    ma_action = MultiAgentAction(scheme, n_agents, output_size, args, )
    learner = QLearner(ma_action, scheme, max_seq_length, n_agents, args, )
    buffer = ReplayBuffer(scheme, n_agents, buffer_size, max_seq_length, )
    data_collect.set_action(ma_action)
    
    last_test = - test_interval - 1
    episode_num, last_episode_num = 0, 0
    train_returns, train_wons = 0.0, 0.0
    last_t_env = 0
    while data_collect.t_env < t_max:

        train_return, train_won, env_info_ = data_collect.collect(buffer, test_mode=False)
        train_returns += train_return 
        train_wons += train_won

        if buffer.can_sample(batch_size):
            episode_sample, max_filled = buffer.sample(batch_size)
            for k, v in episode_sample.items():
                v = v[:, :max_filled]
                episode_sample[k] = v
            learner.train(episode_sample, data_collect.t_env, episode_num , max_filled)

        episode_num += 1
        if (data_collect.t_env - last_test) / test_interval >= 1.0:
            print("train_returns %f"%(train_returns / (episode_num - last_episode_num)))
            print("train_wons %f"%(train_wons / (episode_num - last_episode_num)))
            print("timesteps: %d"%data_collect.t_env)
            print("episode_num: %d, timesteps: %d"%(episode_num, data_collect.t_env - last_t_env) )
            train_returns, train_wons = 0.0, 0.0
            last_episode_num = episode_num
            last_test = data_collect.t_env
            episode_returns, test_battle_won_rate = 0.0 , 0.0
            print("test")
            for i1 in range(test_batch_size):
                episode_return, test_battle_won, env_info_ = data_collect.collect(buffer, test_mode=True)
                episode_returns += episode_return 
                test_battle_won_rate += test_battle_won
                #print("inx %d, return %f, won: %d"%(i1, episode_return, test_battle_won))
                print(env_info_)
            print("timesteps: %d, episode_returns: %f, test_battle_won_rate: %f"%(data_collect.t_env, episode_returns/test_batch_size, test_battle_won_rate/test_batch_size))
        last_t_env = data_collect.t_env

