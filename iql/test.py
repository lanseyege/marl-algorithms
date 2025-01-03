import torch
import torch.nn as nn
import numpy as np
from smac.env import StarCraft2Env

def main2():
    env = StarCraft2Env(map_name = "3m")
    env_info = env.get_env_info()
    print(env_info.keys())
    print(env_info.values())
    n_actions, n_agents = env_info["n_actions"], env_info["n_agents"]
    n_obs, n_states = env_info["obs_shape"], env_info["state_shape"] 
    n_episodes = 10
    print(n_actions)
    print(n_agents)
    print(n_obs)
    print(n_states)
    env.close()

def main():
    env = StarCraft2Env(map_name = "3m")
    env_info = env.get_env_info()
    n_actions, n_agents = env_info["n_actions"], env_info["n_agents"]
    n_obs, n_states = env_info["obs_shape"], env_info["state_shape"] 
    n_episodes = 10
    print(n_actions)
    
    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0
        while not terminated:
            obs, state = env.get_obs(), env.get_state()
            print("obs ... ")
            print(obs)
            print("state ... ")
            print(state)

            env.render()
            
            actions = []
            
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                print('agent id : %d'%agent_id)
                print(avail_actions)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                print(avail_actions_ind)
                action = np.random.choice(avail_actions_ind)
                actions.append(action)
            
            reward, terminated, _ = env.step(actions)
            print('actions')
            print(actions)
            print(reward)
            episode_reward += reward
    
        print("Total reward in episode {} = {}".format(e, episode_reward))
    
    env.close()


main2()


