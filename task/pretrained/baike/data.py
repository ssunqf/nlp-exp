#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import os
import json
import gzip
import numpy as np
from collections import Counter, OrderedDict
from typing import List

import torch
from torchtext import data
from torchtext.data import Dataset
from tqdm import tqdm

from .base import Label, PhraseLabel, listfile


class Field(data.Field):
    '''
        build vocab from counter file
    '''
    def __init__(self, *args, **kwargs):
        super(Field, self).__init__(*args, **kwargs)

    def build_vocab(self, counter_file, **kwargs):
        with gzip.open(counter_file, mode='rt', compresslevel=6) as file:
            counter = Counter(dict(json.loads(file.read())))

        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)


class LabelField(Field):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        super(LabelField, self).__init__(sequential=False, *args, **kwargs)

    def process(self,
                batch: List[List[PhraseLabel]],
                device=None) -> List[List[Label]]:
        return [[Label(phrase.begin + 1,
                       phrase.end + 1,
                       torch.tensor(
                           list(filter(lambda x: x != 0,
                                       [self.vocab.stoi[label] for label in phrase.labels[self.name]])),
                           dtype=torch.long,
                           device=device)) for phrase in phrases] for phrases in batch]


class BaikeDataset(Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path: str, fields: List, **kwargs):
        examples = []
        with gzip.open(path, mode='rt', compresslevel=6) as file:
            for line in tqdm(file, desc=('load dataset from %s' % path)):
                words, *labels = line.split('\t\t')
                words = words.split(' ')
                labels = [PhraseLabel.from_json(label) for label in labels]
                if len(labels) > 0 and len(words) < 300:
                    words = np.array(words, dtype=np.str)
                    # labels = np.array([l.to_np() for l in labels], dtype=PhraseLabel.get_type())
                    examples.append(data.Example.fromlist([words, labels], fields))

        super(BaikeDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, fields: List,
               path=None, root='.data',
               train=None, validation=None, test=None, **kwargs):
        return super(BaikeDataset, cls).splits(
            path=path,
            root=root, fields=fields,
            train=train, validation=None, test=test, **kwargs)

    @classmethod
    def iters(cls, fields, batch_size=16, device=torch.device('cpu'),
              root='.data', path=None, train=None,
              batch_size_fn=None, vectors=None, **kwargs):

        train, *left  = cls.splits(fields, root=root, path=path, train=train, **kwargs)

        if len(left) == 0:
            train, valid = train.split(split_ratio=(len(train)-10000)/len(train))
        else:
            valid = left[0]

        return data.BucketIterator.splits(
            [train, valid],
            batch_size=batch_size, batch_size_fn=batch_size_fn, sort_within_batch=True,
            device=device, **kwargs)


def lazy_iter(fields, data_prefix: str, path=None, lazy=True, repeat=True, **kwargs):
    files = [file[len(path)+1:] for file in listfile(os.path.join(path, data_prefix))]

    if not lazy:
        datasets = [BaikeDataset.iters(fields, train=file, path=path, **kwargs) for file in files]
        while True:
            for dataset in datasets:
                yield dataset
            if not repeat:
                break
    else:
        while True:
            for file in files:
                print(os.path.join(path, file))
                yield BaikeDataset.iters(fields, train=file, path=path, **kwargs)
            if not repeat:
                break
