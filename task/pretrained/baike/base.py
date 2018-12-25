#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import numpy as np
import json
import torch
import gzip
import os
import itertools
import copy

from typing import List

UNK_TOKEN = '<unk>'
INIT_TOKEN = '<s>'
EOS_TOKEN = '</s>'
PAD_TOKEN = '<pad>'
MASK_TOKEN = '<mask>'

MIN_SCORE = -1e5

class Label:
    __slots__ = 'begin', 'end', 'tags'

    def __init__(self, begin, end, tags):
        self.begin = begin
        self.end = end
        self.tags = tags


class PhraseLabel:
    __slots__ = 'begin', 'end', 'labels'

    def __init__(self, begin: int, end: int, **kwargs):
        self.begin = begin
        self.end = end
        self.labels = {k: np.array(v, dtype=np.str) for k, v in kwargs.items()}

    def to_json(self):
        data = {'begin': self.begin, 'end': self.end}
        if len(self.labels) > 0:
            data['labels'] = {k:v.tolist() for k, v in self.labels.items()}
        return json.dumps(data, ensure_ascii=False)

    def to_np(self):
        return self.begin, self.end, self.labels['keys'] #, self.labels['attrs'], self.labels['subtitles']

    @staticmethod
    def get_type():
        dtype = [('begin', np.int16),
                 ('end', np.int16),
                 ('keys', [('key', np.str)]),
                 #('attrs', [('attr', np.str)]),
                 #('subtitles', [('sub', np.str)])
        ]
        return dtype

    @staticmethod
    def from_json(s: str):
        attr = json.loads(s)
        if 'labels' in attr:
            return PhraseLabel(attr['begin'], attr['end'], **attr['labels'])
        else:
            return PhraseLabel(attr['begin'], attr['end'])


def get_dropout_mask(prob: float, tensor_to_mask: torch.Tensor):
    mask = torch.rand(tensor_to_mask.size(), device=tensor_to_mask.device) > prob
    return mask.float().div(1 - prob)


def block_init(tensor: torch.Tensor,
               split_sizes: List[int],
               func=torch.nn.init.orthogonal_,
               gain=1.0):
    data = tensor.data
    sizes = list(tensor.size())

    for max_size, split_size in zip(sizes, split_sizes):
        assert max_size % split_size == 0

    for block_start_indices in itertools.product(*[list(range(0, max_size, split_size))
                                    for max_size, split_size in zip(sizes, split_sizes)]):
        block_slice = tuple([slice(start_index, start_index + split_size)
                      for start_index, split_size in zip(block_start_indices, sizes)])

        data[block_slice] = func(tensor[block_slice].contiguous(), gain)


def clones(module, N):
    "Produce N identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def make_masks(sens: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    masks = torch.ones(sens.size(), dtype=torch.uint8, device=sens.device)
    for i, l in enumerate(lens):
        masks[l:, i] = 0
    return masks


def mixed_open(path: str, mode):
    if path.endswith('.gz'):
        return gzip.open(path, mode=mode, compresslevel=6)
    else:
        return open(path, mode=mode)


def listfile(path: str):
    if os.path.isdir(path):
        return [os.path.join(path, name) for name in os.listdir(path)]
    else:
        dir, prefix = os.path.split(path)
        if len(dir) == 0:
            dir = './'
        return [os.path.join(dir, name)
                for name in os.listdir(dir) if name.startswith(prefix)]
