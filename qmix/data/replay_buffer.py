import os, sys
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, scheme, n_agents, buffer_size, max_seq_length, ):
        self.length = 0
        self.buffer_size = buffer_size
        self.data = {}
        self.episode_length = np.zeros(buffer_size, dtype=np.int32)
        self.episode_occupied = [True] * buffer_size
        for k, v in scheme.items():
            if "group" in v and v["group"]:
                shape = (n_agents, v["shape"])
            else:
                shape = (v["shape"], )
            if k == "actions" or  k == "filled":
                dtype = torch.int32
                self.data[k] = torch.zeros((buffer_size, max_seq_length, *shape), dtype=torch.long)
            else:
                dtype = torch.float32
                self.data[k] = torch.zeros((buffer_size, max_seq_length, *shape))
            if k == "actions_onehot":
                self.out_dim = v["shape"]
            #if k == "actions":
            #    self.data["act_one_hot"] = self.data[k].new(*shape[:-1], out_dim).zero_()
         
        for k, v in self.data.items():
            print(k + " shape: " )
            print(self.data[k].shape)

    def _one_hot(self, out_dim, temp_data):
        #print("temp_data.shape")
        #print(temp_data.shape)
        #print(*temp_data.shape[:-1])
        #print(out_dim)
        #print(*temp_data.shape[:-1], out_dim)
        act_one_hot = temp_data.new(*temp_data.shape[:-1], out_dim).zero_()
        act_one_hot.scatter_(-1, temp_data.long(), 1)
        return act_one_hot.float()

    def update_one(self, trans_data, t_step, mark_filled=True, actions_oned=True):
        _length = self.length % self.buffer_size


        for k, v in trans_data.items():
            #print("K: " + k )
            v = torch.tensor(v ,dtype=torch.float32)
            #print(v.shape)
            self.data[k][_length][t_step:t_step+1] = v.view_as(self.data[k][_length][t_step:t_step+1])
            if k == "actions" and actions_oned:
                self.data["actions_onehot"][_length][t_step:t_step+1] = self._one_hot(self.out_dim, self.data[k][_length][t_step:t_step+1])
            if mark_filled :
                self.data["filled"][_length][t_step: t_step+1] = 1
                mark_filled = False
    
    def update_indicator(self, ):
        _length = self.length % self.buffer_size

        if self.episode_occupied[_length] :
            #print("_length: %d"%_length)
            for k, v in self.data.items():
                #print("k1: " + k)
                #print(self.data[k][_length].shape)
                self.data[k][_length] = 0
                #print("k2: " + k)
                #print(self.data[k][_length].shape)
            self.episode_occupied[_length] = False
            self.episode_length[_length] = 1
        else:
            self.episode_length[_length] += 1

    def update_episode(self, test_mode=False):

        self.episode_occupied[self.length % self.buffer_size] = True
        if not test_mode:
            self.length += 1
        #print("self.length: %d, test_mode: %d"%(self.length, test_mode))
    
    def can_sample(self, batch_size):
        return self.length >= batch_size
    
    def _sample(self, indxs):
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = v[indxs]
        return new_data

    def sample(self, batch_size):
        #print("self.length: %d, self.buffer_size:%d, batch_size: %d"%(self.length, self.buffer_size, batch_size))
        if self.length == batch_size:
            return self._sample(list(range(batch_size))), max(self.episode_length[:batch_size])+1
        else:
            indxs = np.random.choice(min(self.length, self.buffer_size), batch_size, replace=False)
            #print("indxs")
            #print(indxs)
            return self._sample(indxs), max(self.episode_length[indxs])+1

    def get_current(self, t_ep):
        new_data = {}
        for k, v in self.data.items():
            #new_data[k] = v[self.length % self.buffer_size][t_ep].unsqueeze(0)
            new_data[k] = v[self.length % self.buffer_size].unsqueeze(0)
        return new_data


