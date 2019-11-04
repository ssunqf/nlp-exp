#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import numpy as np
import json
import torch
import gzip
import os
import itertools
import copy
from collections import Counter

from typing import List

UNK_TOKEN = '<unk>'
INIT_TOKEN = '<s>'
EOS_TOKEN = '</s>'
PAD_TOKEN = '<pad>'
MASK_TOKEN = '<mask>'

MIN_SCORE = -1e10

class Label:
    __slots__ = 'begin', 'end', 'tags'

    def __init__(self, begin, end, tags):
        self.begin = begin
        self.end = end
        self.tags = tags

    def __len__(self):
        return self.end - self.begin


class PhraseLabel:
    __slots__ = 'begin', 'end', 'labels'

    def __init__(self, begin: int, end: int, **kwargs):
        self.begin = begin
        self.end = end
        self.labels = {k: np.array(v, dtype=np.str) if isinstance(v, list) else v for k, v in kwargs.items()}

    def __len__(self):
        return self.end - self.begin

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


def get_dropout_mask(*sizes, prob: float=None, device: torch.device=None):
    mask = torch.rand(*sizes, device=device) > prob
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


def make_masks(lens: torch.Tensor, device=None) -> torch.Tensor:
    masks = torch.ones(lens.max().item(), lens.size(0), dtype=torch.uint8, device=device if device else lens.device)
    for i, l in enumerate(lens):
        masks[l:, i] = 0
    return masks


def smart_open(path: str, mode='r'):
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


def save_counter(path: str, counter: Counter):
    with smart_open(path, 'wt') as file:
        file.write(json.dumps(counter.most_common(), ensure_ascii=False, indent=2))


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
            rstring += chr(inside_code)
        elif (inside_code >= 65281 and inside_code <= 65374 and inside_code != 65292):  # 全角字符（除空格,逗号）根据关系转化
            inside_code -= 65248
            rstring += chr(inside_code)
        else:
            rstring += uchar
    return rstring


def to_bmes(length, tag):
    if length == 1:
        return ['S_%s' % tag]
    elif length > 1:
        return ['B_%s' % tag] + ['M_%s' % tag] * (length - 2) + ['E_%s' % tag]
    else:
        raise RuntimeError('length must be big than 0.')


def bio_to_bmeso(chars, types):
    n_chars, n_types = [], []
    buffer_c, buffer_t = [], []

    for token, type in zip(chars, types):
        token = strQ2B(token)
        if type[0] in ['B', 'O'] and len(buffer_c) > 0:
            tag = buffer_t[0][2:] if buffer_t[0].startswith('B-') else buffer_t[0]
            buffer_t = to_bmes(len(buffer_c), tag)

            n_chars.extend(buffer_c)
            n_types.extend(buffer_t)

            buffer_c, buffer_t = [], []

        if type[0] in ['B', 'O']:
            buffer_c, buffer_t = [token], [type]
        elif type[0] == 'I':
            buffer_c.append(token)
            buffer_t.append(type)

    if len(buffer_c) > 0:
        tag = buffer_t[0][2:] if buffer_t[0].startswith('B-') else buffer_t[0]
        buffer_t = to_bmes(len(buffer_c), tag)

        n_chars.extend(buffer_c)
        n_types.extend(buffer_t)

    return n_chars, n_types



