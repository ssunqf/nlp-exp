#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import math

import torch
from torch import nn
import torch.nn.functional as F
from .base import clones, LayerNorm, SublayerConnection, PositionwiseFeedForward
from .lattice import LatticeEncoderLayer


class GlobalAttention(nn.Module):
    def __init__(self, dropout=0.2):
        super(GlobalAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        head_dim = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        if mask is not None:
            mask = mask.unsqueeze(-2)
            scores = scores.masked_fill(mask == 0, -1e10)
        p_attn = F.softmax(scores, dim = -1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class LocalAttention(nn.Module):
    def __init__(self, window_size=5, dropout=0.2):
        super(LocalAttention, self).__init__()
        assert window_size % 2 == 1
        self.windown_size = window_size
        self.unfold = nn.Unfold(kernel_size=(window_size, 1), padding=(window_size//2, 0))
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
        query = query.contiguous().view(-1, seq_len, dim)
        key = key.contiguous().view(-1, seq_len, dim)
        value = value.contiguous().view(-1, seq_len, dim)
        mask = mask.contiguous().view(-1, seq_len)
        values = []
        attns = []
        for t in range(seq_len):
            begin = max(0, t - self.windown_size//2)
            end = min(seq_len, t + self.windown_size//2 + 1)

            window_key = key[:, begin:end]
            window_value = value[:, begin:end]
            window_mask = mask[:, begin:end]

            score = torch.matmul(query[:, t:t+1], window_key.transpose(-1, -2)) / math.sqrt(dim)
            window_mask = window_mask.unsqueeze(-2)
            score = score.masked_fill(window_mask == 0, -1e10)

            p_attn = self.dropout(score.softmax(-1))

            values.append(torch.matmul(p_attn, window_value))
            attns.append(p_attn)

        values = torch.cat(values, dim=-2)

        return values, attns


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_head, hidden_dim,
                 atten_window_size=-1,
                 dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert hidden_dim % num_head == 0
        # We assume d_v always equals d_k
        self.head_dim = hidden_dim // num_head
        self.num_head = num_head
        self.linears = clones(nn.Linear(hidden_dim, hidden_dim), 4)
        self.attn = GlobalAttention(dropout) if atten_window_size == -1 \
            else LocalAttention(atten_window_size, dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1).expand(value.size(0), self.num_head, value.size(1))
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from hidden_dim => num_head x head_dim
        query, key, value = \
            [l(x).view(nbatches, -1, self.num_head, self.head_dim).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attn(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.num_head * self.head_dim)
        return self.linears[-1](x)


# -------------------------------------------------------

# attention -> (add + norm) -> ffn -> (add + norm)
class Transformer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(Transformer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# https://github.com/vdumoulin/conv_arithmetic
# https://arxiv.org/pdf/1603.07285.pdf
# https://arxiv.org/pdf/1803.01271.pdf
def _pad_size(kernel, dilation):
    width = (kernel - 1) * dilation + 1
    return width // 2


class ConvUnit(nn.Module):
    def __init__(self, hidden_dim, kernel_size, dilation=1, dropout=0.2):
        super(ConvUnit, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim,
                      kernel_size=kernel_size, padding=_pad_size(kernel_size, dilation), dilation=dilation),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, input, mask) -> torch.Tensor:
        return self.layers(input)


class ConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_layer, kernel_size, dropout=0.2):
        super(ConvLayer, self).__init__()
        layers = []
        input_dim = input_dim
        dilation = 1
        for l in range(num_layer):
            layers.append(ConvUnit(input_dim, output_dim, kernel_size, dilation, dropout=dropout))
            input_dim = output_dim
            dilation *= 2

        self.layers = nn.Sequential(*layers)

    def forward(self, input, mask):
        return self.layers(input)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask, batch_first=False):
        "Pass the input (and mask) through each layer in turn."
        if batch_first is False:
            x = x.transpose(0, 1).contiguous()
            mask = mask.transpose(0, 1).contiguous()

        for layer in self.layers:
            x = layer(x, mask)

        if batch_first is False:
            x = x.transpose(0, 1).contiguous()
            mask = mask.transpose(0, 1).contiguous()
        return self.norm(x)


class LatticeEncoder(nn.Module):
    def __init__(self, layer, N, lattice: LatticeEncoderLayer):
        super(LatticeEncoder, self).__init__()
        self.char_layers = clones(layer, N // 2)
        self.lattice_layer = lattice
        self.join_layers = clones(layer, N - (N // 2))
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask, words, batch_first=False):

        if batch_first is False:
            x = x.transpose(0, 1).contiguous()
            mask = mask.transpose(0, 1).contiguous()

        for layer in self.char_layers:
            x = layer(x, mask)

        x = self.lattice_layer(x, mask, words)

        for layer in self.join_layers:
            x = layer(x, mask)

        if batch_first is False:
            x = x.transpose(0, 1).contiguous()
            mask = mask.transpose(0, 1).contiguous()

        return self.norm(x)
