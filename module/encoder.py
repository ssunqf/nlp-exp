#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from .utils import *
from .attention import MultiHead
import math

# https://github.com/vdumoulin/conv_arithmetic
# https://arxiv.org/pdf/1603.07285.pdf


def _pad_size(kernel, dilation):
    width = (kernel - 1) * dilation + 1
    return width // 2


class ConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=1, dilation=1, dropout=0.2):
        super(ConvLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(input_dim, output_dim,
                      kernel_size=kernel_size, padding=_pad_size(kernel_size, dilation), dilation=dilation),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.layers(input)
        return input + output if self.input_dim == self.output_dim else output


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_layer, kernel_size, dropout=0.2):
        super(ConvBlock, self).__init__()
        layers = []
        input_dim = input_dim
        dilation = 1
        for l in range(num_layer):
            layers.append(ConvLayer(input_dim, output_dim, kernel_size, dilation, dropout=dropout))
            input_dim = output_dim
            dilation *= 2

        self.layers = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor):
        return self.layers(input)


class CNN(nn.Module):
    r"""
        TODO: 当 num_blocks > 1 采用IDCNN的策略，共享参数避免过拟合
    """
    def __init__(self, embed_dim, hidden_dim, hidden_layers: int, atten_heads, num_blocks=1, kernel_size=3, dropout=0.2):
        super(CNN, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.input_transformer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden_dim))
        self.blocks = nn.Sequential(
            *[ConvBlock(hidden_dim, hidden_dim, hidden_layers, kernel_size, dropout) for i in range(num_blocks)])
    
        self.atten_layer = MultiHead(hidden_dim, hidden_dim, atten_heads)

    def forward(self, input: torch.Tensor, masks: torch.Tensor, batch_first=False):
        '''
        :param input: Tensor(seq_len, batch_size, emb_dim) or (batch_size, seq_len, embed_dim)
        :param batch_first: Boolean
        :return:
        '''
        # (seq_len, batch_size, emb_dim) -> (batch_size, hidden_dim, seq_len)
        input = self.input_transformer(input)
        if batch_first is False:
            input = input.transpose(0, 1).contiguous()
        input = input.transpose(1, 2).contiguous()

        input = self.blocks(input)

        output = input.transpose(1, 2).contiguous()
        if batch_first is False:
            output = output.transpose(0, 1).contiguous()

        output, _ = self.atten_layer(output, output, output, masks)
        return output


class RNNBlock(nn.Module):
    def __init__(self,
                 hidden_model,
                 hidden_dim: int,
                 hidden_layers: int,
                 atten_heads: int,
                 dropout: float = 0.2):
        super(RNNBlock, self).__init__()

        assert hidden_model in ['LSTM', 'GRU']
        self.rnn_layer = getattr(nn, hidden_model)(hidden_dim, hidden_dim,
                                                   hidden_layers=hidden_layers,
                                                   bidirectional=True,
                                                   dropout=dropout)

        self.atten_layer = MultiHead(hidden_dim, num_heads=atten_heads)

    def fold(self, output: torch.Tensor):
        dim = output.size(-1) // 2

        return (output[:, :, 0:dim] + output[:, :, dim:]) / 2

    def forward(self, input: torch.Tensor, mask: torch.Tensor, batch_first=False):
        hidden, _ = self.rnn_layer(input)

        hidden = input + self.fold(hidden)

        hidden, _ = self.atten_layer(hidden, mask)
        return hidden


class Recurrent(nn.Module):
    def __init__(self, embed_dim: int, hidden_mode,
                 hidden_dim: int, hidden_layers: int,
                 atten_heads: int,
                 num_blocks=1,
                 dropout: float=0.2):
        super(Recurrent, self).__init__()
        assert hidden_mode in ['LSTM', 'GRU']
        self.input_transformer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden_dim))

        self.blocks = nn.Sequential(
            *[RNNBlock(hidden_mode, hidden_dim, hidden_layers, atten_heads=atten_heads)
              for b in range(num_blocks)]
        )

    def forward(self, input: torch.Tensor, mask: torch.Tensor, last_state=None):
        return self.layers(self.input_transformer(input), mask)


class Encoder(nn.Module):
    def __init__(self, embed_dim, hidden_mode,
                 hidden_dim: int, hidden_layers: int,
                 atten_heads=1,
                 num_blocks=1,
                 dropout: float = 0.2):
        super(Encoder, self).__init__()
        assert hidden_mode in ['LSTM', 'GRU', 'CNN']

        if hidden_mode == 'CNN':
            self.hidden_module = CNN(embed_dim, hidden_dim,
                                     hidden_layers=hidden_layers,
                                     atten_heads=atten_heads,
                                     num_blocks=num_blocks,
                                     dropout=dropout)
        else:
            self.hidden_module = Recurrent(embed_dim, hidden_mode, hidden_dim,
                                           hidden_layers=hidden_layers,
                                           atten_heads=atten_heads,
                                           num_blocks=num_blocks,
                                           dropout=dropout)

    def forward(self, input: torch.Tensor, lengths: torch.Tensor):
        max_len, batch_size, *_ = input.size()
        mask = input.new_ones(max_len, batch_size, dtype=torch.int8)
        for id, len in enumerate(lengths):
            if len < max_len:
                mask[len:, id] = 0
        return self.hidden_module(input, mask)
