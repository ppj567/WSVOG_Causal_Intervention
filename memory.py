# -*- coding: utf-8 -*-
import torch
from torch.autograd import Function
from torch import nn
import torch.nn.functional as F

class MemoryBank(nn.Module):
    def __init__(self, memory_num, memory_dim, momentum=0.9):
        super(MemoryBank, self).__init__()
        self.momentum = momentum
        self.register_buffer('memory', torch.zeros(memory_num, memory_dim))
        self.flag = 0
        self.memory =  self.memory.cuda()
    def forward(self, x, y):
        out = torch.mm(x, self.memory.t())/self.T
        return out

    def update_weight(self, features, index):
        index = torch.tensor(index).cuda()
        weight_pos = self.memory.index_select(0, index)
        weight_pos.mul_(self.momentum)
        weight_pos.add_(torch.mul(F.normalize(features.data,dim=-1), 1 - self.momentum))

        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        self.memory.index_copy_(0, index, updated_weight)
        self.memory = F.normalize(self.memory,dim=-1)


    def set_weight(self, features, index):
        self.memory.index_copy_(0, index, features)


    def update_temp_weight(self, features, index):
        index = torch.tensor(index).cuda()
        weight_pos = self.memory.index_select(0, index)
        weight_pos.mul_(self.momentum)
        weight_pos.add_(torch.mul(F.normalize(features.data.view(20,-1),dim=-1).view(-1), 1 - self.momentum))

        updated_weight = F.normalize(weight_pos.view(20,-1),dim=-1).view(1,-1)
        self.memory.index_copy_(0, index, updated_weight)
