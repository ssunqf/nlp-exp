#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from typing import List, Union, Tuple

import torch
from torch import nn
from torch.nn import functional as F

PAD = '<pad>'
SOS = '<sos>'
EOS = '<eos>'
HiddenState = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]


def split(state: HiddenState, size):
    if isinstance(state, tuple):
        return zip(state[0].split(size, 0), state[1].split(size, 0))
    else:
        return state.split(size, 0)


def cat(states: List[HiddenState]):
    if states is None or len(states) == 0:
        return None
    elif isinstance(states[0], tuple):
        return (torch.cat([s[0] for s in states]), torch.cat([s[1] for s in states]))


def fold(output: torch.Tensor, state: HiddenState):

    dim = output.size(-1)//2

    output = (output[:, :, 0:dim] + output[:, :, dim:]) / 2
    if isinstance(state, tuple) == 2:
        h, c = state
        h = (h[::2] + h[1::2]) / 2
        c = (c[::2] + c[1::2]) / 2

        state = (h, c)
    else:
        state = (state[::2] + state[1::2]) / 2

    return output, state


def embedding(embed: nn.Embedding, indices: torch.Tensor, padding_idx, use_cuda):
    _weight = embed.weight.index_select(0, indices)
    _backend = embed._backend
    if use_cuda:
        weight = _weight.cuda()

    def _forward(input: torch.Tensor):
        return _backend.Embedding.apply(
            input, _weight,
            padding_idx, embed.max_norm, embed.norm_type,
            embed.scale_grad_by_freq, embed.sparse
        )

    return _forward


def linear(linear: nn.Linear, indices: torch.Tensor, use_cuda):
    weight = linear.weight.index_select(0, indices)
    if linear.bias is None:
        bias = None
    else:
        bias = linear.bias.index_select(0, indices)

    if use_cuda:
        weight = weight.cuda()
        bias = None if bias is None else bias.cuda()

    def _forward(input: torch.Tensor):
        return F.linear(input, weight, bias)

    return _forward
