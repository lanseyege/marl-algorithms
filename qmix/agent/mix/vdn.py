import torch
import torch.nn as nn

class VDN(nn.Module):
    def __init__(self, ):
        super(VDN, self).__init__()

    def forward(self, q_values, batch):
        return torch.sum(q_values, dim = 2, keepdim = True)

