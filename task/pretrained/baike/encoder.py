#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import torch
from torch import nn

from typing import Tuple, List

from .base import get_dropout_mask, block_init

from torch.jit import ScriptModule, script_method, trace


class LSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, go_forward, dropout=0.3):
        super(LSTMLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.go_forward = go_forward

        self.dropout = dropout

        self.cell = nn.LSTMCell(input_dim, hidden_dim)

    def forward(self, input: torch.Tensor):

        seq_len, batch_size, input_dim = input.size()

        output, hx = [], None

        if self.training:
            recurrent_mask = get_dropout_mask(batch_size, self.hidden_dim, prob=self.dropout, device=input.device)
        else:
            recurrent_mask = None

        for timestep in (range(seq_len) if self.go_forward else range(seq_len-1, -1, -1)):
            hidden_t, cell_t = self.cell(input[timestep], hx)

            if self.training:
                hidden_t = hidden_t * recurrent_mask
            output.append(hidden_t)
            hx = (hidden_t, cell_t)

        if not self.go_forward:
            output = list(reversed(output))
        output = torch.stack(output, dim=0)

        return output, hx


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, bidirectional=True, residual=False, dropout=0.3):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.residual = residual
        self.dropout = dropout

        forwards, backwards = [], []

        _input_dim = input_dim
        for l in range(num_layers):
            forwards.append(LSTMLayer(_input_dim, hidden_dim, True, dropout))
            backwards.append(LSTMLayer(_input_dim, hidden_dim, False, dropout))

            _input_dim = hidden_dim * 2

        self.forwards = nn.ModuleList(forwards)
        self.backwards = nn.ModuleList(backwards)

    def forward(self, input: torch.Tensor):

        prev_out = input
        prev_h, prev_c = None, None
        for lid, (f_layer, b_layer) in enumerate(zip(self.forwards, self.backwards)):

            f_out, (f_h, f_c) = f_layer(prev_out)
            b_out, (b_h, b_c) = b_layer(prev_out)

            new_out = torch.cat([f_out, b_out], dim=-1)
            if not self.residual or lid == 0:
                prev_out = new_out
            else:
                prev_out = prev_out + new_out

            prev_h = torch.cat([f_h, b_h], dim=-1)
            prev_c = torch.cat([f_c, b_c], dim=-1)

        return prev_out, (prev_h, prev_c)


class LSTMPCell(ScriptModule):
    __constants__ = ['input_dim', 'hidden_dim', 'cell_dim',
                     'go_forward', 'recurrent_dropout_prob',
                     'cell_clip_value', 'hidden_clip_value']

    def __init__(self, input_dim, hidden_dim, cell_dim,
                 go_forward: bool, recurrent_dropout=0.3,
                 cell_clip_value=None, hidden_clip_value=None):
        super(LSTMPCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell_dim = cell_dim

        self.input_linearity = nn.Linear(input_dim, 4 * cell_dim, bias=False)
        self.hidden_linearity = nn.Linear(hidden_dim, 4 * cell_dim, bias=True)

        if hidden_dim != cell_dim:
            self.output_projector = nn.Linear(cell_dim, hidden_dim, bias=False)
        else:
            self.output_projector = None

        self.go_forward = go_forward
        self.recurrent_dropout_prob = recurrent_dropout

        self.cell_clip_value = cell_clip_value
        self.hidden_clip_value = hidden_clip_value

        self.reset_parameters()

    def reset_parameters(self):
        block_init(self.input_linearity.weight.data, [self.cell_dim, self.input_dim], func=torch.nn.init.orthogonal_)
        block_init(self.hidden_linearity.weight.data, [self.cell_dim, self.hidden_dim], func=torch.nn.init.orthogonal_)

        self.hidden_linearity.bias.data.fill_(0.0)
        # init forget gate biases to 1.0
        self.hidden_linearity.bias.data[self.cell_dim:2*self.cell_dim].fill_(1.0)

        if self.output_projector:
            torch.nn.init.orthogonal_(self.output_projector.weight.data)

    @script_method
    def _forward_step(self, input: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):

        i, f, g, o = torch.split(self.input_linearity(input) + self.hidden_linearity(hidden), self.cell_dim, dim=-1)

        i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()

        cell = f * cell + i * g

        if self.cell_clip_value is not None:
            cell = torch.clamp(cell, -self.cell_clip_value, self.cell_clip_value)

        hidden = o * cell.tanh()

        if self.output_projector:
            hidden = self.output_projector(hidden)

        if self.hidden_clip_value is not None:
            hidden = torch.clamp(hidden, -self.hidden_clip_value, self.hidden_clip_value)

        return hidden, cell

    def forward(self, input: torch.Tensor, lens: torch.Tensor):

        seq_len, batch_size, input_dim = input.size()

        output = torch.zeros(seq_len, batch_size, self.hidden_dim, device=input.device)

        hidden_state = torch.zeros(batch_size, self.hidden_dim, device=input.device)
        cell_state = torch.zeros(batch_size, self.cell_dim, device=input.device)

        if self.training:
            recurrent_mask = get_dropout_mask(self.recurrent_dropout_prob, hidden_state)
        else:
            recurrent_mask = None

        print(seq_len, lens)
        for timestep in (range(seq_len) if self.go_forward else range(seq_len-1, -1, -1)):

            batch_size_t = 0
            for length in lens.tolist():
                if length <= timestep:
                    break
                batch_size_t += 1

            print(self.go_forward, timestep, batch_size_t)
            input_t = input[timestep, :batch_size_t]
            hidden_t, cell_t = self._forward_step(input_t,
                                                  hidden_state[0:batch_size_t].clone(),
                                                  cell_state[0:batch_size_t].clone())

            if self.training:
                hidden_t = hidden_t * recurrent_mask[:batch_size_t]

            output[timestep, :batch_size_t] = hidden_t
            hidden_state = hidden_state.clone()
            cell_state = cell_state.clone()
            hidden_state[0:batch_size_t] = hidden_t
            cell_state[0:batch_size_t] = cell_t

        return output, (hidden_state, cell_state)


class BiLSTMP(ScriptModule):
    def __init__(self, input_dim, hidden_dim, cell_dim, recurrent_dropout):
        super(BiLSTMP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_dim
        self.cell_dim = cell_dim

        assert hidden_dim % 2 == 0

        self.forward_cell = LSTMPCell(input_dim, hidden_dim // 2, cell_dim // 2, True, recurrent_dropout)
        self.backward_cell = LSTMPCell(input_dim, hidden_dim // 2, cell_dim // 2, False, recurrent_dropout)

    def forward(self, input: torch.Tensor, lens: torch.Tensor):

        f_out, (f_h, f_c) = self.forward_cell(input, lens)
        b_out, (b_h, b_c) = self.backward_cell(input, lens)

        out = torch.cat([f_out, b_out], dim=-1)

        h = torch.cat([f_h, b_h], dim=-1)
        c = torch.cat([f_c, b_c], dim=-1)

        return out, (h, c)


class StackLSTM(nn.Module):
    def __init__(self,
                 input_dim: int, hidden_dim: int, cell_dim: int, num_layers: int,
                 residual=False,
                 dropout=0.3):
        super(StackLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.residual = residual

        i_size = input_dim
        layers = []
        for l in range(num_layers):
            layers.append(BiLSTMP(i_size, hidden_dim, cell_dim, dropout))
            i_size = hidden_dim

        self.layers = nn.ModuleList(layers)

    def forward(self,
                input: torch.Tensor,
                lens: torch.Tensor,
                batch_first=False) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if batch_first:
            input = input.transpose(0, 1).contiguous()

        for i in range(self.num_layers):
            hidden, (h, c) = self.layers[i](input, lens)
            if i == 0 or self.residual is False:
                input = hidden
            else:
                input = input + hidden

        if batch_first:
            input = input.transpose(0, 1).contiguous()

        return input, (h, c)


class ElmoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout=0.3):
        super(ElmoEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.forwards = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=False, dropout=dropout)
        self.backwards = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=False, dropout=dropout)

    def forward(self, input: torch.Tensor, lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forwards(input)[0], self.backwards(input.flip([0]))[0].flip([0])

    def encoder_word(self, input: torch.Tensor, lens: torch.Tensor, word_ids: List[List[Tuple[int, int]]]):
        forward_h, backward_h = self.forward(input, lens)

        char_seq_len, batch_size, dim = forward_h.size()

        word_lens = [len(s) for s in word_ids]
        new_hidden = torch.FloatTensor(max(word_lens), batch_size, dim * 2, device=input.device)
        for bid, words in enumerate(word_ids):
            sen = []
            for begin, end in words:
                sen.append(torch.cat([forward_h[end-1, bid],backward_h[begin, bid]], -1))

            new_hidden[:word_lens[bid], bid] = torch.stack(sen, dim=0)

        return new_hidden, word_lens







