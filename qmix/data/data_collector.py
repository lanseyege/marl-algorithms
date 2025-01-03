import torch
import numpy as np
import sys, os

from smac.env import StarCraft2Env
import gym


class DataCollector:
    def __init__(self, map_name, batch_size):
        self.length = 0
        self.t_env = 0
        self.env = StarCraft2Env(map_name = map_name)
        self.batch_size = batch_size
        pass

    def set_action(self, ma_action):
        self.ma_action = ma_action

    def reset(self, ):
        self.t = 0
        self.env.reset()
        
        pass

    def collect(self, buffer, test_mode=False):
        self.reset()
        terminated = False
        self.ma_action.init_hidden(1)
        episode_return = 0.0
        test_battle_won, train_battle_won = False, False
        t1 = buffer.length % buffer.buffer_size
        while not terminated:
            buffer.update_indicator()
            pre_trans_data = {
                "state" :  [self.env.get_state()],
                "avail_actions" : [self.env.get_avail_actions()],
                "obs" :  [self.env.get_obs()]
            }
            #print("self.env.get_avail_actions()")
            #print(self.env.get_avail_actions())
            buffer.update_one(pre_trans_data, self.t, mark_filled = True)
            actions = self.ma_action.select_action(buffer.data, self.t, self.t_env, slice(t1, t1+1), test_mode)
            #print("actions")
            #print(actions)
            reward, terminated, info = self.env.step(actions[0])
            pos_trans_data = {
                "actions" : actions,
                "reward" :  [reward],
                "terminated" : [terminated]
            }
            if test_mode and "battle_won" in info and info["battle_won"]:
                test_battle_won = True
            elif not test_mode and "battle_won" in info and info["battle_won"]:
                train_battle_won = True
            #print("reward: %f"%reward)
            episode_return += reward
            buffer.update_one(pos_trans_data, self.t, mark_filled=False)
            self.t += 1
            #if self.t >= 2:
            #    sys.exit()
        last_data = {
            "state" : [self.env.get_state()],
            "avail_actions" : [self.env.get_avail_actions()],
            "obs" : [self.env.get_obs()]
        }
        buffer.update_one(last_data, self.t, mark_filled = False)
        actions = self.ma_action.select_action(buffer.data, self.t, self.t_env, slice(t1, t1+1), test_mode)
        buffer.update_one({"actions": actions}, self.t, mark_filled=False, actions_oned=False)
        buffer.update_episode(test_mode=test_mode)

        if not test_mode:
            self.t_env += self.t
        if test_mode and test_battle_won:
            test_battle_won = True
        elif test_mode and "battle_won" in info:
            test_battle_won = info["battle_won"]
        elif not test_mode and "battle_won" in info:
            train_battle_won = info["battle_won"]

        if test_mode:
            return episode_return , int(test_battle_won), info
        else:
            return episode_return , int(train_battle_won), info


