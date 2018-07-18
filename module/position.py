#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.autograd import Variable
import math
import numpy as np

'''
    Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, L.; and Polosukhin, I. 2017.
    Attention is all you need. arXiv preprint arXiv:1706.03762
    timing(t, 2i) = sin(t/(10000^2i/d))
    timing(t, 2i + 1) = cos(t/(10000^2i/d)) 
'''

class TimeSignal(nn.Module):
    '''
       http://export.arxiv.org/pdf/1706.03762
       sinusoid position embedding
    '''
    def __init__(self, dim, max_length=5000):
        super(TimeSignal, self).__init__()

        self.dim = dim

        # 1 / (10000^(2i/d))
        inv_timescales = torch.exp(torch.arange(0, self.dim, 2) * (-math.log(10000) / self.dim))
        # t / (10000^(2i/d))
        times = torch.arange(0, max_length).unsqueeze(1) * inv_timescales.unsqueeze(0)

        self.embedding = nn.Parameter(torch.cat([times.sin(), times.cos()], - 1), requires_grad=False)

        self.scaled_factor = nn.Parameter(torch.ones(1))

    def parameters(self):
        yield self.scaled_factor

    def forward(self, input, batch_first=False):
        if batch_first:
            batch_first, max_len = input.size()
            return self.scaled_factor * self.embedding[0:max_len].unsqueeze(0).expand(batch_first, -1, -1)
        else:
            max_len, batch_first = input.size()
            return self.scaled_factor * self.embedding[0:max_len].unsqueeze(1).expand(-1, batch_first, -1)
