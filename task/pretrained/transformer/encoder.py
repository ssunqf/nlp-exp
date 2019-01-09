#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import math

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from .base import clones, AddNormLayer, PositionwiseFeedForward
from .lattice import LatticeEncoderLayer


# attention -> (add + norm) -> ffn -> (add + norm)
class Transformer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(Transformer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(AddNormLayer(size, dropout), 2)
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
    "Core encode is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

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
        self.norm = nn.LayerNorm(layer.size)

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


class LSTMEncoder(nn.Module):
    def __init__(self, hidden_dim, hidden_layer, attn, dropout=0.2):
        super(LSTMEncoder, self).__init__()
        self.size = hidden_dim
        self.lstm = nn.LSTM(hidden_dim, hidden_dim//2, hidden_layer,
                            batch_first=False, bidirectional=True, dropout=dropout)
        self.attn = attn

    def forward(self, input, mask, words, batch_first=False):

        if batch_first:
            input = input.transpose(0, 1).contiguous()
            mask = mask.transpose(0, 1).contiguous()

        lens = mask.sum(0)
        packed_input = pack_padded_sequence(input, lens)
        packed_hidden, _ = self.lstm(packed_input)
        hidden, _ = pad_packed_sequence(packed_hidden)

        hidden = self.attn(hidden, hidden, hidden, mask, batch_first=False)

        return hidden