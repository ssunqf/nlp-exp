#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import numpy as np
import json
import torch
import gzip
import os


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
        return json.dumps({'begin': self.begin, 'end': self.end, 'labels': self.labels},
                          ensure_ascii=False)

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
        return PhraseLabel(attr['begin'], attr['end'], **attr['labels'])


def get_dropout_mask(prob: float, tensor_to_mask: torch.Tensor):
    mask = torch.rand(tensor_to_mask.size(), device=tensor_to_mask.device) > prob
    return mask.float().div(1 - prob)


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