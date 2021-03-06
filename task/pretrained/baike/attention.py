#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import math
from typing import Tuple
import torch
from torch import nn
from .base import MIN_SCORE
from ..transformer.base import PositionwiseFeedForward


class GlobalAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(GlobalAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        "Compute 'Scaled Dot Product Attention'"
        head_dim = query.size(-1)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        if mask is not None:
            mask = mask.unsqueeze(-2)
            attention_scores = attention_scores.masked_fill(mask == 0, MIN_SCORE)
        attention_probs = self.dropout(attention_scores.softmax(-1))

        return torch.matmul(attention_probs, value), attention_probs


class LocalAttention(nn.Module):
    def __init__(self, window_size, dropout=0.3):
        super(LocalAttention, self).__init__()
        assert window_size % 2 == 1
        self.window_size = window_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        '''
        :param query: FloatTensor(*, seq_len, dim)
        :param key: FloatTensor(*, seq_len, dim)
        :param value: FloatTensor(*, seq_len, dim)
        :param mask: ByteTensor(*, seq_len)
        :return:
        '''
        value_size = value.size()
        seq_len, dim = value_size[-2:]
        query = query.reshape(-1, seq_len, dim)
        key = key.reshape(-1, seq_len, dim)
        value = value.reshape(-1, seq_len, dim)
        mask = mask.reshape(-1, seq_len)

        half = min(self.window_size//2, seq_len//2)

        scores = []
        for offset in range(-half, half+1, 1):
            if offset < 0:
                w = (query[:, -offset:] * key[:, :offset]).sum(-1).masked_fill(mask[:, :offset], MIN_SCORE)
                w = torch.cat([w.new_full((w.size(0), -offset), MIN_SCORE), w], dim=-1)
            elif offset == 0:
                w = (query * key).sum(-1).masked_fill(mask, MIN_SCORE)
            else:
                w = (query[:, :-offset] * key[:, offset:]).sum(-1).masked_fill(mask[:, offset:], MIN_SCORE)
                w = torch.cat([w, w.new_full((w.size(0), offset), MIN_SCORE)], dim=-1)
            scores.append(w)

        probs = self.dropout(torch.stack(scores, dim=-1).softmax(-1))

        new_value = value.new_zeros(value.size())

        for offset in range(-half, half+1, 1):
            if offset < 0:
                new_value[:, -offset:] += probs[:, -offset:, offset+half:offset+half+1] * value[:, :offset]
            elif offset == 0:
                new_value += probs[:, :, half:half+1] * value
            else:
                new_value[:, :-offset] += probs[:, :-offset, offset+half:offset+half+1] * value[:, offset:]

        return new_value, probs


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_head, hidden_dim, atten_window_size=None, dropout=0.0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert hidden_dim % num_head == 0
        # We assume d_v always equals d_k
        self.head_dim = hidden_dim // num_head
        self.num_head = num_head
        self.query_linearity = nn.Linear(hidden_dim, hidden_dim)
        self.key_linearity = nn.Linear(hidden_dim, hidden_dim)
        self.value_linearity = nn.Linear(hidden_dim, hidden_dim)
        self.attention = GlobalAttention(dropout) if atten_window_size is None or atten_window_size < 0 \
            else LocalAttention(atten_window_size, dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.query_linearity.weight)
        nn.init.xavier_normal_(self.key_linearity.weight)
        nn.init.xavier_normal_(self.value_linearity.weight)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask=None,
                batch_first=False) -> Tuple[torch.Tensor, torch.Tensor]:
        if not batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1).expand(mask.size(0), self.num_head, mask.size(1))

        # 1) Do all the linear projections in batch from hidden_dim => num_head x head_dim
        query = self._split_head(self.query_linearity(query))
        key = self._split_head(self.key_linearity(key))
        value = self._split_head(self.value_linearity(value))

        # 2) Apply attention on all the projected vectors in batch.
        value, attention_weights = self.attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        value = self._concat_head(value)

        if not batch_first:
            value = value.transpose(0, 1)
            attention_weights.mean()

        return value, attention_weights

    def _split_head(self, input):
        batch_size, seq_len, dim = input.size()
        return input.reshape(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)

    def _concat_head(self, input):
        batch_size, num_head, seq_len, head_dim = input.size()
        assert num_head == self.num_head
        assert head_dim == self.head_dim
        return input.transpose(1, 2).reshape(batch_size, seq_len, self.num_head * self.head_dim)


class TransformerLayer(nn.Module):
    """
    attention -> add & norm -> PositionwiseFeedForward
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, num_heads, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.output_linear = nn.Linear(size, size)
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

        self.atten = MultiHeadedAttention(num_heads, size, dropout=dropout)

        self.ffn = PositionwiseFeedForward(size, dropout=dropout)

    def forward(self, input, mask=None, batch_first=False):
        atten_output = self.atten(input, input, input, mask, batch_first)
        atten_output = self.output_linear(atten_output)
        return self.ffn(self.norm(input + self.dropout(atten_output)))