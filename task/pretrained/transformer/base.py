#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import copy
import torch
from torch import nn
import torch.nn.functional as F

from typing import Union
from sklearn.utils.murmurhash import murmurhash3_32
from task.util import utils

PAD = '<pad>'
BOS = '<s>'
EOS = '</s>'

MIN_SCORE = -1e5

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def hash_next(key: int, subword: Union[str, int]):
    next = murmurhash3_32(subword) if isinstance(subword, str) else subword
    if key is None:
        key = next
    else:
        key = (key * 8978948897894561157) ^ ((1 + next) * 17894857484156487943);
    return key


def hash(word: str):
    current = None
    tlen = 0
    for t, subword in utils.replace_entity(word):
        current = hash_next(current, subword)
        tlen += 1
    return current, tlen


# add and norm
class AddNormLayer(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout=0.1):
        super(AddNormLayer, self).__init__()
        self.output_linear = nn.Linear(size, size)
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.output_linear.weight, std=0.02)
        if self.output_linear.bias is not None:
            nn.init.normal_(self.output_linear.bias, std=0.02)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        output = self.output_linear(sublayer(x))
        return self.norm(x + self.dropout(output))


# reference https://github.com/google-research/bert/blob/master/modeling.py
def gelu(input: torch.Tensor):
    cdf = 0.5 * (1.0 + torch.erf(input / torch.sqrt(2.0)))
    return input * cdf


activations = {'gelu': gelu, 'relu': F.relu, 'sigmoid': F.sigmoid, 'tanh': F.tanh}


def get_act_func(name: str):
    fn = activations.get(name)
    return fn if fn else activations['relu']


# ------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, dim, act_name='relu', dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear = nn.Linear(dim, dim)
        self.activation = get_act_func(act_name)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear.weight, std=0.02)
        if self.linear.bias is not None:
            nn.init.normal_(self.linear.bias, std=0.02)

    def forward(self, x):
        return self.activation(self.linear(x))