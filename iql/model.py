import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical

class DQN_NN(nn.Module):
    def __init__(self, obs_size, act_size, hid_size = 128, layer_nums = 2, active='tanh'):
        super(DQN_NN, self).__init__()
        self.active = torch.sigmoid
        if  active == 'tanh':
            self.active = torch.tanh
        elif active == 'relu':
            self.active = torch.relu
        self.linear_1 = nn.Linear(obs_size, hid_size)
        self.linear_2 = nn.Linear(hid_size, hid_size)
        self.linear_3 = nn.Linear(hid_size, act_size)
        self.linear_3.weight.data.mul_(0.1)
        self.linear_3.bias.data.mul_(0.0)

        pass

    def forward(self, x, avail_actions=None):
        x = self.linear_1(x)
        x = self.active(x)
        x = self.linear_2(x)
        x = self.active(x)
        x = self.linear_3(x)
        #print('x shape')
        #print(x.shape)
        if avail_actions is not  None:
            x[ avail_actions == 0]  = -1e10
        #x = self.active(x)
        #print('x avail')
        #print(x)
        x = F.softmax(x, dim=1)
        #print('x softmax')
        #print(x)
        return x

    def select_action(self, x, avail_actions, epsilon):
        x = self.forward(x, avail_actions)
        
        if avail_actions is not None:

            if np.random.rand() < epsilon:
                #print('random action ...')
                temp_1 = torch.nonzero(avail_actions, as_tuple=True)[1]
                #print(temp_1)
                temp_1 = temp_1.numpy()
                #temp1 = torch.nonzero(avail_actions).view(-1).numpy()
                #print(temp_1)
                temp2 = np.random.choice(temp_1)
                #print(temp2)
                #return torch.from_numpy(temp2)
                return temp2
            else:
                #print('max action ... ')
                #print(x.argmax(dim=-1).numpy()[0])
                return x.argmax(dim=-1).numpy()[0]
            
        else:
            pass

    def select_action2(self, x, avail_actions, epsilon):
        action = []
        x = self.forward(x, avail_actions)
        '''
        if np.random.rand() > epsilon:
            if avail_actions is not None:
                torch.nonzero(avail_actions)
                return x[avail_actions == 0].random.choice()
            else:
                return x[].random.
        else:
            return x.argmax(dim=-1)
        '''
        if avail_actions is not None:
            for temp_ in x:
                if np.random.rand() < epsilon:
                    temp_1 = torch.nonzero().view(-1).numpy()
                    temp_2 = np.random.choice(temp_1)
                    actions.append(temp_2)
                    pass
                else:
                    actions.append(temp_.argmax(dim = -1).numpy()[0])
                    pass
            '''
            if np.random.rand() < epsilon:
                temp1 = torch.nonzero(avail_actions).view(-1).numpy()
                temp2 = np.random.choice(temp1)
                #return torch.from_numpy(temp2)
                return temp2
            else:
                return x.argmax(dim=-1).numpy()[0]
            '''
        else:
            pass

class DQN_RNN(nn.Module):
    def __init__(self, obs_size, act_size, hid_size = 64, layer_nums = 2, active='tanh'):
        super(DQN_RNN, self).__init__()
        self.hid_size = hid_size
        self.fc1 = nn.Linear(obs_size, hid_size)
        self.rnn = nn.GRUCell(hid_size, hid_size)
        self.fc2 = nn.Linear(hid_size, act_size)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.hid_size).zero_()

    def forward(self, x):
        x = F.relu(self,fc1(x))
        h_in = hidden_state.reshape(-1, self.hid_size)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

