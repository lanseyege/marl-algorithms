import torch
from agent.model import DQN_RNN
import torch.nn.functional as F
from torch.distributions import Categorical
import sys

class MultiAgentAction:
    #def __init__(self, ):
    #    pass
    
    def __init__(self, scheme, n_agents, output_size, args, ):
        self.n_agents = n_agents
        input_shape = self._get_input_shape(scheme)
        self.agent = self._build_agent(args.agent_name, input_shape, args.hid_size, output_size)
        self.epsilon_start = args.epsilon_start
        self.epsilon_finish = args.epsilon_finish
        self.epsilon_time_length = args.epsilon_time_length
        self.delta = (self.epsilon_start - self.epsilon_finish) / self.epsilon_time_length
        self.epsilon = self._eval(0)

    def init_hidden(self, batch_size):
        self.batch_size = batch_size
        self.hidden_state = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["shape"]
        input_shape += self.n_agents
        input_shape += scheme["actions_onehot"]["shape"]
        return input_shape

    def _build_agent(self, agent_name, input_size, hid_size, output_size):
        agent = None
        if agent_name == "RNN":
            agent = DQN_RNN(input_size, output_size, hid_size )
        return agent

    def _build_input(self, episode, bs, t):
        input_data = []
        input_data.append(episode['obs'][bs, t]) ## add obs to input 
        #print("episode['obs'][:, t]")
        #print(episode['obs'][:, t].shape)
        if t == 0: # add actions to input
            input_data.append(torch.zeros_like(episode['actions_onehot'][bs, t]))
            #print("torch.zeros_like(episode['actions_onehot'][:, t])")
            #print(torch.zeros_like(episode['actions_onehot'][:, t]).shape)
            #print(torch.zeros_like(episode['actions_onehot'][:, t]))
        else:
            input_data.append(episode['actions_onehot'][bs, t-1])
            #print("episode['actions_onehot'][bs, t-1]")
            #print(episode['actions_onehot'][:, t-1].shape)
            #print(episode['actions_onehot'][:, t-1])
        ## add agent id to input 
        #print("agents ")
        input_data.append(torch.eye(self.n_agents).unsqueeze(0).expand(self.batch_size, -1, -1))
        #print(torch.eye(self.n_agents).unsqueeze(0).expand(self.batch_size, -1, -1).shape)
        #print(torch.eye(self.n_agents).unsqueeze(0).expand(self.batch_size, -1, -1))
        input_data = torch.cat([x.reshape(self.batch_size*self.n_agents,-1 ) for x in input_data], dim=1)
        #print('input_data')
        #print(input_data.shape)
        #print(input_data)
        #if t >= 4:
        #    sys.exit()
        return input_data

    def _eval(self, T):
        return max(self.epsilon_finish, self.epsilon_start - self.delta * T)

    def _select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        self.epsilon = self._eval(t_env)
        #print("self.epsilon")
        #print(self.epsilon)
        if test_mode:
            self.epsilon = 0.0
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")
        random_numbers = torch.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        #print("avail_actions")
        #print(avail_actions.shape)
        #print(avail_actions)
        random_actions = Categorical(avail_actions.float()).sample().long()
        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions

    def select_action(self, episode, t, t_env, bs=slice(None, None), test_mode=False):
        avail_actions = episode["avail_actions"][bs, t]
        #print("seelaction ")
        #print(avail_actions)
        agent_inputs = self.forward(episode,  t, bs, test_mode=False)

        chosen_action = self._select_action(agent_inputs, avail_actions, t_env, test_mode=test_mode)
        return chosen_action

    def forward(self, episode, t, bs=slice(None, None), test_mode=False):
        input_data = self._build_input(episode, bs, t)
        #print("input data")
        #print(input_data.shape)
        #print(input_data)
        agent_outs, self.hidden_state = self.agent.forward(input_data, self.hidden_state)

        #print("agent_outs")
        #print(agent_outs.shape)
        #print(agent_outs)

        #print("self.hidden_state")
        #print(self.hidden_state.shape)
        #print(self.hidden_state)
        #if t >= 1:
        #    sys.exit()
        #self.hidden_state = h
        if "coma" == "pi_logits":
            avail_actions = episode["avail_actions"][bs, t]
            reshaped_avail_actions = avail_actions.reshape( self.batch_size * self.n_agents, -1)
            print("reshaped_avail_actions")
            print(reshaped_avail_actions.shape)
            print(reshaped_avail_actions)
        
            agent_outs[reshaped_avail_actions == 0] = -1e10
            #agent_outs = F.sigmoid(agent_outs)
            agent_outs = F.softmax(agent_outs, dim=-1)
            print("agent_outs 1 ")
            print(agent_outs.shape)
            print(agent_outs)
            #print("out")
            #print(out)
            if not test_mode:
                epsilon_action_num = agent_outs.size(-1)
                epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()
                agent_outs = ((1-self.epsilon)*agent_outs + 
                        torch.ones_like(agent_outs)*self.epsilon / epsilon_action_num)
                agent_outs[reshaped_avail_actions == 0] = 0.0
                #print("agent_outputs 2")
                #print(agent_outs.shape)
                #print(agent_outs)
                #print("self.epsilon: %f "self.epsilon)
            #return out.argmax(dim = 1).numpy()[0]
        agent_outputs = agent_outs.view(self.batch_size, self.n_agents, -1)
        #print("agent_outputs")
        #print(agent_outputs)
        #if t >= 1:
        #    sys.exit()

        return agent_outputs

    def parameters(self, ):
        return self.agent.parameters()
    
    def named_parameters(self, ):
        return self.agent.named_parameters()

    def load_state(self, others):
        self.agent.load_state_dict(others.agent.state_dict())
