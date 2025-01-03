import torch
import torch.nn as nn
import numpy as np
import os, sys
import copy
from torch.optim import RMSprop

from agent.mix.vdn import VDN
from agent.mix.qmix import QMIX

class QLearner:
    def __init__(self, ma_action, scheme, max_seq_length, n_agents, args,):
        self.ma_action = ma_action
        self.batch_size = args.batch_size
        self.max_seq_length = max_seq_length
        self.double_q = args.double_q
        self.gamma = args.gamma
        self.grad_norm_clip = args.grad_norm_clip
        self.target_update_interval = args.target_update_interval 
        self.lr = args.lr
        self.optim_alpha = args.optim_alpha
        self.optim_eps = args.optim_eps

        self.last_target_update = 0

        self.params = list(ma_action.parameters())
        self.named_params = list(ma_action.named_parameters())
        print("self.params")
        print(self.params)
        print("self. named params")
        print(self.named_params)
        state_shape = scheme["state"]["shape"]
        self.mix = None
        if args.mix != "":
            if args.mix == "vdn":
                self.mix = VDN()
            elif args.mix == "qmix":
                self.mix = QMIX(state_shape, n_agents, args)
            else:
                raise ValueError("mix error ")
            self.params += list(self.mix.parameters())
            self.target_mix = copy.deepcopy(self.mix)

        self.optimiser = RMSprop(params=self.params, lr = self.lr, alpha = self.optim_alpha, eps=self.optim_eps )
        self.target_ma_action = copy.deepcopy(ma_action)

    def train(self, batch, t_env, t_episode, max_filled):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        ma_action_out = []
        self.ma_action.init_hidden(self.batch_size)
        for k in range(max_filled):
            agent_outs = self.ma_action.forward(batch, t = k, bs=slice(None, None))
            ma_action_out.append(agent_outs)
        ma_action_out = torch.stack(ma_action_out, dim = 1)
        chosen_action_qvals = torch.gather(ma_action_out[:, :-1], dim=3, index=actions).squeeze(3)

        target_ma_action_out = []
        self.target_ma_action.init_hidden(self.batch_size)
        for k in range(max_filled):
            target_agent_outs = self.target_ma_action.forward(batch, t=k, bs=slice(None, None))
            target_ma_action_out.append(target_agent_outs)
        target_ma_action_out = torch.stack(target_ma_action_out[1:], dim = 1)
        target_ma_action_out[avail_actions[:, 1:] == 0] = -9999999
        if self.double_q:
            ma_action_out_detach = ma_action_out.clone().detach()
            ma_action_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = ma_action_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = torch.gather(target_ma_action_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_ma_action_out.max(dim = 3)[0]

        if self.mix is not None:
            chosen_action_qvals = self.mix(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mix(target_max_qvals, batch["state"][:, 1:])

        target = rewards + self.gamma *(1 - terminated) * target_max_qvals

        td_error = (chosen_action_qvals - target.detach())

        mask = mask.expand_as(td_error)
        mask_td_error = td_error * mask
        loss = (mask_td_error ** 2).sum() / mask.sum()
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.params, self.grad_norm_clip)
        self.optimiser.step()

        if t_episode % 100 == 0:
            print("loss: %f, "%(loss.float()))
            print("grad_norm: %f"%(grad_norm.float()))

        if (t_episode - self.last_target_update) / self.target_update_interval >= 1.0:
            print("update target ... %d"%t_episode)
            self.target_ma_action.load_state(self.ma_action)
            if self.mix is not None:
                self.target_mix.load_state_dict(self.mix.state_dict())
            self.last_target_update = t_episode


   
