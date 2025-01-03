import torch
import torch.nn as nn
import torch.nn.functional as F

class QMIX(nn.Module):
    def __init__(self, state_shape, n_agents, args):
        super(QMIX, self).__init__()
        self.args = args
        self.n_agents = n_agents
        self.state_shape = state_shape
        self.mix_embed_dim = args.mix_embed_dim
        self.hypernet_embed = args.hypernet_embed 
        
        self.hyper_w_1 = nn.Sequential(
                            nn.Linear(self.state_shape, self.hypernet_embed),
                            nn.ReLU(),
                            nn.Linear(self.hypernet_embed, self.mix_embed_dim*self.n_agents))
        self.hyper_w_2 = nn.Sequential(
                            nn.Linear(self.state_shape, self.hypernet_embed),
                            nn.ReLU(),
                            nn.Linear(self.hypernet_embed, self.mix_embed_dim))
        self.hyper_b_1 = nn.Linear(self.state_shape, self.mix_embed_dim)
        self.hyper_b_2 = nn.Sequential(
                            nn.Linear(self.state_shape, self.mix_embed_dim),
                            nn.ReLU(),
                            nn.Linear(self.mix_embed_dim, 1))

    def forward(self, q_values, states):
        lens = q_values.size(0)
        states = states.reshape(-1, self.state_shape)
        q_values = q_values.view(-1, 1, self.n_agents)
        w1 = torch.abs(self.hyper_w_1(states)).view(-1, self.n_agents, self.mix_embed_dim)
        b1 = self.hyper_b_1(states).view(-1, 1, self.mix_embed_dim)
        hidden = F.elu(torch.bmm(q_values, w1) + b1)
        w2 = torch.abs(self.hyper_w_2(states)).view(-1, self.mix_embed_dim, 1)
        b2 = self.hyper_b_2(states).view(-1, 1, 1)
        q_tot = (torch.bmm(hidden, w2) + b2).view(lens, -1, 1)
        
        return q_tot

