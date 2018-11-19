#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.rnn import RNNCellBase
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from .base import get_dropout_mask


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, hidden_layer: int,
                 dropout: float=0.2):
        super(LSTMEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim//2, hidden_layer,
                            batch_first=False, bidirectional=True, dropout=dropout)

    def forward(self,
                input: torch.Tensor,
                mask: torch.Tensor,
                batch_first=False) -> torch.Tensor:

        if batch_first:
            input = input.transpose(0, 1).contiguous()
            mask = mask.transpose(0, 1).contiguous()

        lens = mask.sum(0)
        packed_input = pack_padded_sequence(input, lens)
        packed_hidden, _ = self.lstm(packed_input)
        hidden, _ = pad_packed_sequence(packed_hidden)

        return hidden


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, go_forward: bool, recurrent_dropout=0.2):
        super(LSTMCell, self).__init__()
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.go_forward = go_forward
        self.recurrent_dropout_prob = recurrent_dropout

    def forward(self, input: PackedSequence):

        input, batch_sizes = input

        seq_len = batch_sizes.size()[0]
        max_batch_size = batch_sizes[0]

        output = input.new_zeros(input.size(0), self.hidden_size)

        hidden_state = input.new_zeros(max_batch_size, self.hidden_size)
        cell_state = input.new_zeros(max_batch_size, self.hidden_size)

        recurrent_mask = get_dropout_mask(self.recurrent_dropout_prob, hidden_state) if self.training else None

        cumsum_sizes = torch.cumsum(batch_sizes, dim=0)
        for timestep in range(seq_len):
            timestep = timestep if self.go_forward else seq_len - timestep - 1
            len_t = batch_sizes[timestep]
            begin, end = (cumsum_sizes[timestep]-len_t, cumsum_sizes[timestep])

            input_t = input[begin:end]
            hidden_t, cell_t = self.cell(input_t, (hidden_state[0:len_t], cell_state[0:len_t]))

            if self.training:
                hidden_t = hidden_t * recurrent_mask[:len_t]

            output[begin:end] = hidden_t
            hidden_state = hidden_state.clone()
            cell_state = cell_state.clone()
            hidden_state[0:batch_sizes[timestep]] = hidden_t
            cell_state[0:batch_sizes[timestep]] = cell_t

        return PackedSequence(output, batch_sizes), (hidden_state, cell_state)


class LSTMLayer(nn.Module):
    def __init__(self, input_size, output_size, recurrent_dropout):
        super(LSTMLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        assert output_size % 2 == 0

        self.forward_cell = LSTMCell(input_size, output_size//2, True, recurrent_dropout)
        self.backward_cell = LSTMCell(input_size, output_size//2, False, recurrent_dropout)

    def forward(self, input: PackedSequence):

        f_out, (f_h, f_c) = self.forward_cell(input)
        b_out, (b_h, b_c) = self.backward_cell(input)

        out = torch.cat([f_out.data, b_out.data], dim=-1)

        h = torch.cat([f_h, b_h], dim=-1)
        c = torch.cat([f_c, b_c], dim=-1)

        return PackedSequence(out, input.batch_sizes), (h, c)


class StackLSTM(nn.Module):
    def __init__(self,
                 input_size: int, hidden_size: int, num_layers: int,
                 residual=True,
                 recurrent_dropout=0.2, dropout=0.2):
        super(StackLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

        i_size = input_size
        layers = []
        for l in range(num_layers):
            layers.append(LSTMLayer(i_size, hidden_size, recurrent_dropout))
            i_size = hidden_size

        self.layers = nn.ModuleList(layers)

    def forward(self,
                input: torch.Tensor,
                mask: torch.Tensor,
                batch_first=False) -> torch.Tensor:
        if batch_first:
            input = input.transpose(0, 1).contiguous()
            mask = mask.transpose(0, 1).contiguous()

        lens = mask.sum(0)
        packed_input = pack_padded_sequence(input, lens)

        for i in range(self.num_layers):
            packed_hidden, _ = self.layers[i](packed_input)
            if i == 0 or self.residual is False:
                packed_input = packed_hidden
            else:
                packed_input = PackedSequence(
                    packed_input.data + packed_hidden.data, packed_hidden.batch_sizes)

        hidden, _ = pad_packed_sequence(packed_input)

        return hidden


class LatticeLSTMCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LatticeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')

        return self._forward(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,)

    # copy from torch.nn._funcations.rnn::LSTMCell
    def _forward(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        hx, cx = hidden
        gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

        ingate, forgetgate, cellgate = gates.chunk(3, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        return cy


