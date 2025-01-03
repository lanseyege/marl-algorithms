import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical


class DQN_RNN(nn.Module):
    def __init__(self, input_size, act_size, hid_size = 64, ):
        super(DQN_RNN, self).__init__()
        self.hid_size = hid_size
        self.fc1 = nn.Linear(input_size, hid_size)
        self.rnn = nn.GRUCell(hid_size, hid_size)
        self.fc2 = nn.Linear(hid_size, act_size)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.hid_size).zero_()

    def forward(self, x, hidden_state):
        x = F.relu(self.fc1(x))
        h_in = hidden_state.reshape(-1, self.hid_size)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

