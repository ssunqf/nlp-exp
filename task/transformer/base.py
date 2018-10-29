#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import copy
import torch
from torch import nn
import torch.nn.functional as F

from typing import Union
from sklearn.utils import murmurhash
from task.util import utils


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def hash_next(key: int, subword: Union[str, int]):
    next = murmurhash.murmurhash3_32(subword) if isinstance(subword, str) else subword
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


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(dim))
        self.b_2 = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# add and norm
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


# ------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, input_dim, ff_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(input_dim, ff_dim)
        self.w_2 = nn.Linear(ff_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))