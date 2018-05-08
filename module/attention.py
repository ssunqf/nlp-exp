#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, List
import math

# http://web.stanford.edu/class/cs224n/lectures/lecture11.pdf

# attention is all you need https://arxiv.org/pdf/1706.03762.pdf

class Attention(nn.Module):

    def calcDist(self,
                 query: torch.Tensor,
                 key: torch.Tensor,
                 value: torch.Tensor,
                 key_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                key_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param query: FloatTensor(batch, trg_len, dim) or FloatTensor(batch, dim)
        :param key: FloatTensor(batch, src_len, dim)
        :param key_mask: ByteTensor(batch, len)
        :return:
            context: FloatTensor(batch, trg_len, dim) or FloatTensor(batch, dim)
            atten_dist: FloatTensor(batch, trg_len, src_len) or FloatTensor(batch, src_len)
        """
        one_step = False
        if query.dim() == 2:
            query = query.unsqueeze(1)
            one_step = True

        context, atten_dist = self.calcDist(query, key, value, key_mask)

        if one_step:
            context = context.squeeze(1)
            atten_dist = atten_dist.squeeze(1)

        return context, atten_dist


class ScaledDotProduct(Attention):
    def __init__(self, hidden_dim: int):
        super(ScaledDotProduct, self).__init__()
        self.hidden_dim = hidden_dim
        self.scale_ratio = 1 / math.sqrt(self.hidden_dim)

    def calcDist(self,
                 query: torch.Tensor,
                 key: torch.Tensor,
                 value: torch.Tensor,
                 key_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        key_batch, key_len, key_dim = key.size()
        query_batch, query_len, query_dim = query.size()

        assert key_batch == query_batch
        assert key_dim == query_dim

        # [batch, key_len, dim] -> [batch, dim, key_len]
        key_t = key.transpose(1, 2)

        # [batch, key_len] -> [batch, query_len, key_len]
        key_mask = key_mask.unsqueeze(1).expand(key_batch, query_len, key_len).contiguous()
        # [batch, query_len, dim] * [batch, dim, key_len] -> [batch, query_len, key_len]
        atten_dist = F.softmax(torch.bmm(query, key_t).masked_fill(key_mask == 0, -1e20) * self.scale_ratio, 2)

        # [batch, query_len, key_len] * [batch, key_len, dim] -> [batch, query_len, dim]
        context = torch.bmm(atten_dist, value)

        return context, atten_dist


class Additive(Attention):
    def __init__(self, query_dim: int, key_dim: int):
        super(Additive, self).__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim

        middle_dim = (query_dim + key_dim) // 2
        self.query_trans = nn.Linear(query_dim, middle_dim)
        self.key_trans = nn.Linear(key_dim, middle_dim)
        self.ffn = nn.Sequential(nn.Tanh(),
                                 nn.Linear(middle_dim, 1))

    def calcDist(self,
                 query: torch.Tensor,
                 key: torch.Tensor,
                 value: torch.Tensor,
                 key_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        key_batch, key_len, key_dim = key.size()
        query_batch, query_len, query_dim = query.size()

        assert key_batch == query_batch

        # [batch, query_len, key_len, dim]
        trans = self.query_trans(query).unsqueeze(2).expand(query_batch, query_len, key_len, -1).contiguous()
        trans += self.key_trans(key).unsqueeze(1).expand(key_batch, query_len, key_len, -1).contiguous()

        # [batch, query_len, key_len]
        dist = F.softmax(self.ffn(trans).unsqueeze(-1).masked_fill(key_mask == 0, -1e20), 2)

        # [batch, query_len, key_len] * [batch, key_len, dim] -> [batch, query_len, dim]
        context = torch.bmm(dist, value)

        return context, dist


class Multiplicative(Attention):
    def __init__(self, query_dim: int, key_dim: int):
        super(Multiplicative, self).__init__()
        self.bilinear = nn.Bilinear(query_dim, key_dim, 1, False)
        self.scale_ratio = 1 / math.sqrt(self.hidden_dim)

    def calcDist(self,
                 query: torch.Tensor,
                 key: torch.Tensor,
                 value: torch.Tensor,
                 key_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        key_batch, key_len, key_dim = key.size()
        query_batch, query_len, query_dim = query.size()

        assert key_batch == query_batch

        query = query.unsqueeze(2).expand(query_batch, query_len, key_len, query_dim).contiguous()
        key = key.unsqueeze(1).expand(key_batch, query_len, key_len, key_dim).contiguous()

        # [batch, query_len, key_len]
        dist = F.softmax(self.bilinear(query, key).unsqueeze(-1).masked_fill(key_mask == 0, -1e20) * self.scale_ratio, 2)

        # [batch, query_len, key_len] * [batch, key_len, dim] -> [batch, query_len, dim]
        context = torch.bmm(dist, value)

        return context, dist


class MultiHead(Attention):
    def __init__(self,
                 query_dim: int,
                 key_dim: int,
                 num_heads: int,
                 mode: str='dot'):
        super(MultiHead, self).__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        assert (query_dim % num_heads == 0) and (key_dim % num_heads == 0)
        self.query_head_dim = query_dim//num_heads
        self.key_head_dim = key_dim//num_heads

        assert mode in ['dot', 'add', 'mul']

        if mode == 'dot':
            assert query_dim == key_dim
            self.atten = ScaledDotProduct(self.query_head_dim)
        elif mode == 'add':
            self.atten = Additive(self.query_head_dim, self.key_head_dim)
        elif mode == 'mul':
            self.atten = Multiplicative(self.query_head_dim, self.key_head_dim)

    # [batch, len, dim] -> [batch, len, num_head, head_dim] ->  [batch * num_head, len, head_dim]
    def _flatten_head(self, data: torch.Tensor):

        batch, len, dim = data.size()
        assert dim % self.num_heads == 0
        head_dim = dim // self.num_heads
        return data.contiguous().view(batch, len, self.num_heads, head_dim) \
            .transpose(1, 2)\
            .contiguous() \
            .view(-1, len, head_dim) \


    def calcDist(self,
                 query: torch.Tensor,
                 key: torch.Tensor,
                 value: torch.Tensor,
                 key_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        key_batch, key_len, key_dim = key.size()
        query_batch, query_len, query_dim = query.size()
        value_batch, value_len, value_dim = value.size()

        assert key_batch == query_batch
        assert key_dim == query_dim

        key = self._flatten_head(key)

        # [batch * num_head, query_len, head_dim]
        query = self._flatten_head(query)

        value = self._flatten_head(value)

        key_mask = key_mask.unsqueeze(1) \
            .expand(key_batch, self.num_heads, key_len) \
            .contiguous() \
            .view(key_batch * self.num_heads, key_len)

        context, dist = self.atten(query, key, value, key_mask)

        # [batch, query_len, dim]
        value_head_dim = value_dim // self.num_heads
        context = context.contiguous()\
            .view(key_batch, self.num_heads, query_len, value_head_dim)\
            .transpose(1, 2)\
            .contiguous()\
            .view(key_batch, query_len, value_dim)

        dist = dist.contiguous().view(key_batch, self.num_heads, query_len, key_len).mean(1)

        return context, dist


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim//num_heads

        self.qkv_transformor = nn.Linear(hidden_dim, 3*hidden_dim)

        self.multi_head = MultiHead(self.hidden_dim, self.num_heads)

    @staticmethod
    def _mask(batch_size: int, seq_len: int, lengths: List[int]) -> torch.Tensor:
        mask = torch.zeros(batch_size, seq_len, dtype=torch.uint8)
        for i, l in enumerate(lengths):
            mask[i, 0:l] = 1

        return mask

    def forward(self, data: torch.Tensor, lens: List[int]):
        '''
        :param data: [seq_len, batch_size, hidden_dim]
        :param lens: List[int]
        :return:
            context: FloatTensor(batch, trg_len, dim) or FloatTensor(batch, dim)
            scores: FloatTensor(batch, trg_len, src_len) or FloatTensor(batch, src_len)
        '''
        # [seq_len, batch_size, hidden_dim] -> ([batch_size, seq_len,hidden_dim], [batch_size, seq_len, hidden_dim], [batch_size, seq_len, hidden_dim])

        seq_len, batch_size, _ = data.size()
        query, key, value = self.qkv_transformor(data).transpose(0, 1).chunk(3, -1)
        src_mask = self._mask(batch_size, seq_len, lens)
        context, scores = self.multi_head(query, key, value, src_mask)
        return context.transpose(0, 1), scores.transpose(0, 1)
