#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, List
import math


# attention is all you need https://arxiv.org/pdf/1706.03762.pdf
class ScaledDotProduct(nn.Module):
    def __init__(self, hidden_dim: int):
        super(ScaledDotProduct, self).__init__()
        self.hidden_dim = hidden_dim

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
            score: FloatTensor(batch, trg_len, src_len) or FloatTensor(batch, src_len)
        """
        one_step = False
        if query.dim() == 2:
            query = query.unsqueeze(1)
            one_step = True

        key_batch, key_len, key_dim = key.size()
        query_batch, query_len, query_dim = query.size()

        assert key_batch == query_batch
        assert key_dim == query_dim

        # [batch, key_len, dim] -> [batch, dim, key_len]
        key_t = key.transpose(1, 2)

        # [batch, query_len, dim] * [batch, dim, query_len] -> [batch, query_len, query_len]
        key_mask = key_mask.unsqueeze(1).expand(query_batch, query_len, key_len)

        score = F.softmax(torch.bmm(query, key_t).masked_fill(key_mask == 0, -1e20)/math.sqrt(key_dim), 2)

        # [batch, query_len, key_len] * [batch, key_len, dim] -> [batch, key_len, dim]
        context = torch.bmm(score, value)

        if one_step:
            context = context.squeeze(1)
            score = score.squeeze(1)

        return context, score


class MultiHead(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super(MultiHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim//num_heads

        self.atten = ScaledDotProduct(self.head_dim)


    # [batch, len, dim] -> [batch, len, num_head, head_dim] ->  [batch * num_head, len, head_dim]
    def _flatten_head(self, data: torch.Tensor):

        batch, len, dim = data.size()
        return data.contiguous().view(batch, len, self.num_heads, self.head_dim) \
            .transpose(1, 2)\
            .contiguous() \
            .view(-1, len, self.head_dim) \


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param query: FloatTensor(batch, query_len, dim) or FloatTensor(batch, dim)
        :param key: FloatTensor(batch, src_len, dim)
        :param src_mask: ByteTensor(batch, len)
        :return:
            context: FloatTensor(batch, query_len, dim) or FloatTensor(batch, dim)
            score: FloatTensor(batch, query_len, src_len) or FloatTensor(batch, src_len)
        """

        one_step = False
        if query.dim() == 2:
            query = query.unsqueeze(1)
            one_step = True

        key_batch, key_len, key_dim = key.size()
        query_batch, query_len, query_dim = query.size()

        assert key_batch == query_batch
        assert key_dim == query_dim

        key = self._flatten_head(key)

        # [batch * num_head, query_len, head_dim]
        query = self._flatten_head(query)

        value = self._flatten_head(value)

        src_mask = src_mask.contiguous()\
            .view(key_batch, 1, key_len).expand(key_batch, self.num_heads, key_len) \
            .contiguous()\
            .view(key_batch * self.num_heads, key_len)

        context, score = self.atten(key, query, value, src_mask)

        # [batch, query_len, dim]
        context = context.contiguous()\
            .view(key_batch, self.num_heads, query_len, self.head_dim)\
            .transpose(1, 2)\
            .contiguous()\
            .view(key_batch, query_len, self.num_heads * self.head_dim)

        score = score.contiguous().view(key_batch, self.num_heads, query_len, key_len).mean(1)

        if one_step:
            context = context.squeeze(1)
            score = score.squeeze(1)

        return context, score


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
