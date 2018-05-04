#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from torch import Tensor
from module.attention import *
from module.utils import *


class DecoderState:

    def __init__(self, src_hidden, src_mask, context, hidden_state):

        self.src_hidden = src_hidden
        self.src_mask = src_mask
        self.context = context
        self.hidden_state = hidden_state

    def use_cuda(self):
        return self.src_mask.is_cuda

    def repeat(self, beam_size):

        batch_size, _, _ = self.src_hidden.size()
        assert batch_size == 1
        src_hidden = self.src_hidden.repeat(beam_size, 1, 1)
        src_mask = self.src_mask.repeat(beam_size, 1)

        context = self.context.repeat(beam_size, 1)

        if isinstance(self.hidden_state, tuple):
            h, c = self.hidden_state
            h = h.repeat(1, beam_size, 1)
            c = c.repeat(1, beam_size, 1)
            hidden_state = (h, c)
        else:

            hidden_state = self.hidden_state.repeat(1, beam_size, 1)

        return DecoderState(src_hidden, src_mask, context, hidden_state)

    def update(self, context, hidden_state, ids=None):

        if ids is None:
            self.context = context
            self.hidden_state = hidden_state
        else:
            self.context = context.index_select(0, ids)
            self.hidden_state =\
                (hidden_state[0].index_select(1, ids), hidden_state[1].index_select(1, ids)) \
                    if isinstance(hidden_state, tuple) else hidden_state.index_select(1, ids)


    @staticmethod
    def chunk(log_prob: Tensor, context: Tensor, hidden_state: HiddenState, attn: Tensor, size: int):

        log_prob = log_prob.chunk(size, 0)
        context = context.chunk(size, 0)
        if isinstance(hidden_state, tuple):
            h, c = hidden_state
            hs = h.chunk(size, 1)
            cs = c.chunk(size, 1)
            states = list(zip(hs, cs))
        else:
            states = hidden_state.chunk(size, 1)

        attns = attn.chunk(size, 0)
        return log_prob, context, states, attns


    @staticmethod
    def init_split(src_hidden: Tensor,
                   src_mask: Tensor,
                   context: Tensor,
                   hidden_state: HiddenState, split_size):
        src_hidden = src_hidden.split(split_size)
        src_mask = src_mask.split(split_size)
        context = context.split(split_size)

        if isinstance(hidden_state, tuple):
            h, c = hidden_state
            hs = h.split(split_size, 1)
            cs = c.split(split_size, 1)

            hidden_state = list(zip(hs, cs))
        else:
            hidden_state = hidden_state.split(split_size, 1)

        return [DecoderState(src_h, src_m, output, state)
                for src_h, src_m, output, state in zip(src_hidden, src_mask, context, hidden_state)]


    @classmethod
    def cat(cls, states):

        context = torch.cat([b.context for b in states], 0)
        if isinstance(states[0].hidden_state, tuple):
            h = torch.cat([b.hidden_state[0] for b in states], 1)
            c = torch.cat([b.hidden_state[1] for b in states], 1)
            hidden_state = (h, c)
        else:
            hidden_state = torch.cat([b.hidden_state for b in states], 1)

        src_hidden = torch.cat([b.src_hidden for b in states], 0)
        src_mask = torch.cat([b.src_mask for b in states], 0)

        return DecoderState(src_hidden, src_mask, context, hidden_state)


class Decoder(nn.Module):
    def __init__(self, embed_dim: int,
                 hidden_model: str, hidden_dim: int, num_layers: int, dropout=0.2):
        super(Decoder, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout=0.2

        self.attention = ScaledDotProduct(self.hidden_dim)

        self.hidden_model = hidden_model.upper()

        if self.hidden_model in ['LSTM', 'GRU']:
            self.hidden = getattr(nn, self.hidden_model)(
                embed_dim + hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        else:
            raise RuntimeError('Can\'t support hidden_model %s' % (hidden_model))

        self.context = nn.Sequential(nn.Dropout(dropout),
                                     nn.Linear(hidden_dim*2, hidden_dim),
                                     nn.Tanh())

    def forward(self, input: Tensor, state: DecoderState):
        """
        :param input: FloatTensor[batch, embed_dim]
        :param state: DecoderState
        :return:
            log_prob: FloatTensor[batch, vocab_size]
            context: FloatTensor[batch, hidden_dim]
            hidden_state: (h, c)  FloatTensor[num_layer, batch, hidden_dim]
            attn_score: FloatTensor[batch, src_len]
        """

        # [1, batch, embed_size + hidden_size]
        hidden_input = torch.cat([input, state.context], -1).unsqueeze(0)
        hidden_output, hidden_state = self.hidden(hidden_input, state.hidden_state)
        hidden_output = hidden_output.squeeze(0)

        # [batch, hidden_dim], [batch, trg_len]
        context_s, attn_score = self.attention(hidden_output, state.src_hidden, state.src_mask)

        context = self.context(torch.cat([context_s, hidden_output], -1))

        return context, hidden_state, attn_score



class ResidualDecoder(nn.Module):
    def __init__(self, embed_dim: int,
                 hidden_model: str, hidden_dim: int, num_layers: int, dropout=0.2):
        super(ResidualDecoder, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_model = hidden_model.upper()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = 0.2

        self.attention = ScaledDotProduct(self.hidden_dim)


        assert self.hidden_dim in ['LSTM', 'GRU']
        self.rnn_layers = nn.ModuleList()
        input_dim = embed_dim + hidden_dim
        for l in range(self.num_layers):
            self.rnn_layers.append(
                getattr(nn, self.hidden_model)(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout))

            input_dim = hidden_dim


        self.context = nn.Sequential(nn.Dropout(dropout),
                                     nn.Linear(hidden_dim * 2, hidden_dim),
                                     nn.Tanh())


    def forward(self, input: Tensor, state: DecoderState):
        """
        :param input: FloatTensor[batch, embed_dim]
        :param state: DecoderState
        :return:
            log_prob: FloatTensor[batch, vocab_size]
            context: FloatTensor[batch, hidden_dim]
            hidden_state: (h, c)  FloatTensor[num_layer, batch, hidden_dim]
            attn_score: FloatTensor[batch, src_len]
        """

        # [1, batch, embed_size + hidden_size]
        hidden_input = torch.cat([input, state.context], -1).unsqueeze(0)

        last_state = split(state.hidden_state, self.num_layers)
        new_states = []
        for l in range(self.num_layers):
            hidden_output, hidden_state = self.rnn_layers[l](hidden_input, last_state[l])
            hidden_output = hidden_output.squeeze(0)
            new_states.append(hidden_state)

            hidden_input = hidden_input + hidden_output if l > 0 else hidden_output

        # [batch, hidden_dim], [batch, trg_len]
        context_s, attn_score = self.attention(hidden_input, state.src_hidden, state.src_mask)

        context = self.context(torch.cat([context_s, hidden_input], -1))

        return context, cat(hidden_state), attn_score
