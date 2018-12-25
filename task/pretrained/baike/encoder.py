#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import torch
from torch import nn

from .base import get_dropout_mask, block_init


class Encoder(nn.Module):
    def forward(self, input: torch.Tensor, lens: torch.Tensor, batch_first=False):
        raise NotImplementedError()

class LSTMPCell(nn.Module):
    def __init__(self, input_dim,hidden_dim, cell_dim,
                 go_forward: bool, recurrent_dropout=0.3,
                 cell_clip_value=None, hidden_clip_value=None):
        super(LSTMPCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell_dim = cell_dim

        self.input_linearity = nn.Linear(input_dim, 4 * cell_dim, bias=False)
        self.hidden_linearity = nn.Linear(hidden_dim, 4 * cell_dim, bias=True)

        self.output_projector = nn.Linear(cell_dim, hidden_dim, bias=False)

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

        torch.nn.init.orthogonal_(self.output_projector.weight.data)

    def _forward_step(self, input: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):

        i, f, g, o = torch.split(self.input_linearity(input) + self.hidden_linearity(hidden), self.cell_dim, dim=-1)

        i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()

        cell = f * cell + i * g

        if self.cell_clip_value:
            cell = torch.clamp(cell, -self.cell_clip_value, self.cell_clip_value)

        hidden = self.output_projector(o * cell.tanh())

        if self.hidden_clip_value:
            hidden = torch.clamp(hidden, -self.hidden_clip_value, self.hidden_clip_value)

        return hidden, cell

    def forward(self, input: torch.Tensor, lens: torch.Tensor):

        seq_len, batch_size, input_dim = input.size()

        output = input.new_zeros(seq_len, batch_size, self.hidden_dim)

        hidden_state = input.new_zeros(batch_size, self.hidden_dim)
        cell_state = input.new_zeros(batch_size, self.cell_dim)

        recurrent_mask = get_dropout_mask(self.recurrent_dropout_prob, hidden_state) if self.training else None

        for timestep in range(seq_len):
            timestep = timestep if self.go_forward else seq_len - timestep - 1

            batch_size_t = 0
            for length in lens:
                if length <= timestep:
                    break
                batch_size_t += 1

            input_t = input[timestep, :batch_size_t]
            hidden_t, cell_t = self._forward_step(input_t, hidden_state[0:batch_size_t], cell_state[0:batch_size_t])

            if self.training:
                hidden_t = hidden_t * recurrent_mask[:batch_size_t]

            output[timestep, :batch_size_t] = hidden_t
            hidden_state = hidden_state.clone()
            cell_state = cell_state.clone()
            hidden_state[0:batch_size_t] = hidden_t
            cell_state[0:batch_size_t] = cell_t

        return output, (hidden_state, cell_state)


class BiLSTMP(nn.Module):
    def __init__(self, input_dim, hidden_dim, cell_dim, recurrent_dropout):
        super(BiLSTMP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_dim
        self.cell_dim = cell_dim

        assert hidden_dim % 2 == 0

        self.forward_cell = LSTMPCell(input_dim, hidden_dim // 2, cell_dim, True, recurrent_dropout)
        self.backward_cell = LSTMPCell(input_dim, hidden_dim // 2, cell_dim, False, recurrent_dropout)

    def forward(self, input: torch.Tensor, lens: torch.Tensor):

        f_out, (f_h, f_c) = self.forward_cell(input, lens)
        b_out, (b_h, b_c) = self.backward_cell(input, lens)

        out = torch.cat([f_out, b_out], dim=-1)

        h = torch.cat([f_h, b_h], dim=-1)
        c = torch.cat([f_c, b_c], dim=-1)

        return out, (h, c)


class StackLSTM(Encoder):
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
                batch_first=False) -> torch.Tensor:
        if batch_first:
            input = input.transpose(0, 1).contiguous()

        for i in range(self.num_layers):
            hidden, _ = self.layers[i](input, lens)
            if i == 0 or self.residual is False:
                input = hidden
            else:
                input = input + hidden

        if batch_first:
            input = input.transpose(0, 1).contiguous()

        return input


class ElmoEncoder(Encoder):
    def __init__(self, input_dim: int, hidden_dim: int, cell_dim: int, num_layers: int, dropout=0.3):
        super(ElmoEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        in_dim = input_dim
        forwards, backwards = [], []
        for l in range(num_layers):
            forwards.append(LSTMPCell(in_dim, hidden_dim, cell_dim, go_forward=True, recurrent_dropout=dropout))
            backwards.append(LSTMPCell(in_dim, hidden_dim, cell_dim, go_forward=False, recurrent_dropout=dropout))
            in_dim = hidden_dim

        self.forwards = nn.ModuleList(forwards)
        self.backwards = nn.ModuleList(backwards)

    def forward(self, input: torch.Tensor, lens: torch.Tensor, batch_first=False) -> torch.Tensor:
        if batch_first:
            input = input.transpose(0, 1).contiguous()

        f_outs, b_outs = [input], [input]

        for forward_cell, backward_cell in zip(self.forwards, self.backwards):
            f_out, (f_h, f_c) = forward_cell(f_outs[-1], lens)
            b_out, (b_h, b_c) = backward_cell(b_outs[-1], lens)
            f_outs.append(f_out)
            b_outs.append(b_out)

        outs = [torch.cat([f_out, b_out], dim=-1) for f_out, b_out in zip(f_outs, b_outs)]

        h = torch.cat([f_h, b_h], dim=-1)
        c = torch.cat([f_c, b_c], dim=-1)

        if batch_first:
            outs = [out.transpose(0, 1).contiguous() for out in outs]

        return outs[-1]