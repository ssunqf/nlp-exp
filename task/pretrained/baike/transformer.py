#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import math

import torch
from torch import nn

from .attention import MultiHeadedAttention
from ..transformer.base import clones

from .base import make_masks
from .encoder import Encoder


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

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, size: int, dropout=0.3):
        super(PositionwiseFeedForward, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size))

    def forward(self, input):
        return self.model(input)


#-------------------------------------------- embedding -------------------------------------------
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


# token embedding  * math.sqrt(embed_dim) + pos_emb
class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx=None, dropout=0.3):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos = PositionalEncoding(embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_dim = embed_dim

    def forward(self, x):
        return self.dropout(self.pos(self.lut(x)))


# -------------------------------------------------------

# attention -> (add + norm) -> ffn -> (add + norm)
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout=0.3):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(AddNormLayer(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Transformer(Encoder):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, input: torch.Tensor, lens: torch.Tensor, batch_first=False):
        "Pass the input (and mask) through each layer in turn."

        mask = make_masks(input, lens)
        if batch_first is False:
            input = input.transpose(0, 1).contiguous()
            mask = mask.transpose(0, 1).contiguous()

        for layer in self.layers:
            input = layer(input, mask)

        if batch_first is False:
            input = input.transpose(0, 1).contiguous()

        return input

    @staticmethod
    def create(encoder_size, attention_num_head, encoder_depth, atten_window_size=None, dropout=0.3):
        attention = MultiHeadedAttention(attention_num_head, encoder_size,
                                         atten_window_size=atten_window_size, dropout=0.3)
        ffn = PositionwiseFeedForward(encoder_size, dropout=0.3)
        transformer = EncoderLayer(encoder_size, attention, ffn, dropout=0.3)

        return Encoder(transformer, encoder_depth)
