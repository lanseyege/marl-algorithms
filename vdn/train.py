import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from smac.env import StarCraft2Env
import gym
import argparse

from model import DQN_NN
from util import ReplyBuffer

def test(env, agent_list, n_agents, n_obs, n_states, n_actions):
    won_num = 0
    for temp_e in range(20):
        env.reset()
        terminated = False
        obs, state = env.get_obs(), env.get_state()
        obs = torch.FloatTensor(obs)
        state = torch.FloatTensor(state)

        while not terminated:
            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions = torch.Tensor(avail_actions)
                cats = torch.cat((state, obs[agent_id]), 0)
                cats = cats.view(-1, n_obs+n_states)
                avail_actions = avail_actions.view(-1, n_actions)
                
                action = agent_list[agent_id].select_action(cats, avail_actions, 0.0)
                actions.append(action)
            reward, terminated, what = env.step(actions)
            if 'battle_won' in what and what['battle_won']:
                won_num += 1
    print('test win rate: %f'% (1.0 * won_num / 20))
        

def main(args):
    env = StarCraft2Env(map_name = "3m")
    env_info = env.get_env_info()
    n_actions, n_agents = env_info["n_actions"], env_info["n_agents"]
    n_obs, n_states = env_info["obs_shape"], env_info["state_shape"] 
    n_episodes = args.n_episodes
    print('actions dim %d'%n_actions)
    print('state dim %d'%n_states)
    print('obs dim %d'%n_obs)
    print('agent number %d'%n_agents)
    gamma = torch.Tensor([args.gamma])
    agent_list, reply_list, opt_list, agent_bev_list = [], [], [], []
    for i in range(n_agents):
        agent_list.append(DQN_NN(n_obs+n_states, n_actions))
        agent_bev_list.append(DQN_NN(n_obs+n_states, n_actions))
        reply_list.append(ReplyBuffer(args.batch_size))

    for i in range(n_agents):
        opt_list.append(optim.Adam(agent_list[i].parameters(), lr = args.lr))
    index_e = 0       
    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0
        obs, state = env.get_obs(), env.get_state()
        obs = torch.FloatTensor(obs)
        state = torch.FloatTensor(state)
        while not terminated:
            index_e += 1
            actions = []
            temp, temp_avail = [], []
            
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions = torch.Tensor(avail_actions)
                temp_avail.append(avail_actions)
                cats = torch.cat((state, obs[agent_id]), 0)
                temp.append(cats)
                cats = cats.view(-1, n_obs+n_states)
                avail_actions = avail_actions.view(-1, n_actions)
                
                action = agent_list[agent_id].select_action(cats, avail_actions, args.epsilon)
                actions.append(action)
            reward, terminated, what = env.step(actions)
            obs, state = env.get_obs(), env.get_state()
            obs = torch.FloatTensor(obs)
            state = torch.FloatTensor(state)
            for agent_id in range(n_agents):
                next_avail_actions = env.get_avail_agent_actions(agent_id)
                next_avail_actions = torch.Tensor(next_avail_actions)
                reply_list[agent_id].add([temp[agent_id], torch.cat((state, obs[agent_id]),0), reward, actions[agent_id], terminated, temp_avail[agent_id], next_avail_actions])
                
            episode_reward += reward
            Q_, y_bar_ = [], []
            for agent_id in range(n_agents):
                data = reply_list[agent_id].sample()
                sobs_0, sobs_1, reward, action, term, avail_action, next_avail_action = zip(*data)
                reward = torch.FloatTensor(reward)
                sobs_0 = torch.stack(list(sobs_0), dim = 0)
                sobs_1 = torch.stack(list(sobs_0), dim = 0)
                avail_action = torch.stack(list(avail_action), dim = 0)
                next_avail_action = torch.stack(list(next_avail_action), dim = 0)

                #Q = torch.max(agent_list[agent_id](sobs_1, next_avail_action), dim=1)[0].detach()
                Q = torch.max(agent_bev_list[agent_id](sobs_1, next_avail_action), dim=1)[0].detach()
                Q_.append(Q)
                #y = reward.add(gamma * Q)
                y_bar = torch.max(agent_list[agent_id](sobs_0, avail_action), dim=1)[0]
                y_bar_.append(y_bar)
                #loss_dqn = (y - y_bar).pow(2).mean()
                
                #opt_list[agent_id].zero_grad()
                #loss_dqn.backward()
                #opt_list[agent_id].step()
            Q_sum = torch.sum(torch.stack(Q_), dim = 0)
            y = reward.add(gamma * Q_sum)
            y_bar_sum = torch.sum(torch.stack(y_bar_), dim = 0)
            loss_dqn = (y - y_bar_sum).pow(2).mean()
            for agent_id in range(n_agents):
                opt_list[agent_id].zero_grad()
            loss_dqn.backward()
            for agent_id in range(n_agents):
                opt_list[agent_id].step()

            if e % 100 == 0:
                for agent_id in range(n_agents):
                    agent_bev_list[agent_id].load_state_dict(agent_list[agent_id].state_dict())
            if index_e % 1000 == 0:
                test(env, agent_list, n_agents, n_obs, n_states, n_actions)
        print("Total reward in episode {} = {}".format(e, episode_reward))
    
    env.close()

   

if __name__ == '__main__':

    epsilon = 0.2
    batch_size = 32
    lr = 0.01
    gamma = 0.9
    n_episodes = 100000
    parser = argparse.ArgumentParser(description='Independent Q-Learning ... ')
    parser.add_argument('--epsilon', default=epsilon, type=float, help='epsilon-greedy action selection')
    parser.add_argument('--batch_size', default=batch_size, type=int, help='batch size')
    parser.add_argument('--lr', default=lr, type=float, help='learning rate')
    parser.add_argument('--gamma', default=gamma, type=float, help='discount factor')
    parser.add_argument('--n_episodes', default=n_episodes, type=int, help='episodes')
    args = parser.parse_args()
    main(args)
