#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from .utils import *


class Encoder(nn.Module):
    def __init__(self, embed_dim, hidden_model,
                 hidden_dim: int, num_layers: int,
                 bidirectional: bool = True,
                 folding: bool = True,
                 dropout: float = 0.2):
        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers= num_layers
        self.bidirectional = bidirectional
        self.folding = folding
        self.dropout = dropout

        self.hidden_model = hidden_model.upper()
        assert hidden_model in ['LSTM', 'GRU']

        self.rnn = getattr(nn, self.hidden_model)(embed_dim, hidden_dim,
                                                  num_layers=num_layers,
                                                  bidirectional=bidirectional,
                                                  dropout=dropout)

    def forward(self, input: torch.Tensor, state=None):

        output, state = self.rnn(input, state)

        return fold(output, state) if self.bidirectional and self.folding else (output, state)


class ResidualEncoder(nn.Module):
    def __init__(self, embed_dim: int, hidden_model,
                 hidden_dim: int, num_layers: int,
                 bidirectional: bool = True,
                 folding: bool = True,
                 dropout: float = 0.2):
        super(ResidualEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.folding = folding
        self.num_directions = 2 if bidirectional else 1

        assert hidden_model in ['LSTM', 'GRU']
        self.rnn_layers = nn.ModuleList()
        input_dim = embed_dim
        for l in range(num_layers):
            rnn = getattr(nn, hidden_model)(input_dim, hidden_dim,
                                             num_layers=1,
                                             bidirectional=bidirectional,
                                             dropout=dropout)
            self.layers.add_module('%s-%d' % (hidden_model, l), rnn)
            input_dim = hidden_dim * self.num_directions

    def forward(self, input: torch.Tensor, last_state=None):
        """
        :param input: FloatTensor(batch, dim)
        :param last_state: h or (h, c).  h, c is FloatTensor(num_layers * num_directions, batch, hidden_size)
        :return:
        """

        last_state = split(last_state, self.num_layers)
        new_state = []
        for l in range(1, self.num_layers, 1):
            output, output_state = self.rnn_layers[l](input, last_state[l])

            input = input + output if l > 0 else output

            new_state.append(output_state)

        new_state = cat(new_state)

        return fold(output, new_state) if self.bidirectional and self.folding else (output, new_state)