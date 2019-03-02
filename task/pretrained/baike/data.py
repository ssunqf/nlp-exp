#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import os
import json
import gzip
import random
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
    def __init__(self, mask_token=None, *args, **kwargs):
        super(Field, self).__init__(*args, **kwargs)
        self.mask_token = mask_token

    def build_vocab(self, counter_file, **kwargs):
        with gzip.open(counter_file, mode='rt', compresslevel=6) as file:
            counter = Counter(dict(json.loads(file.read())))

        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token, self.mask_token]
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
                                       [self.vocab.stoi[label] for label in phrase.labels[self.name]] if self.name in phrase.labels else [])),
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
                chars, *labels = line.strip().split('\t\t')
                # words = words.split(' ')
                chars = list(chars)
                try:
                    labels = [PhraseLabel.from_json(label) for label in labels if len(label) > 0]
                    if 10 < len(chars) < 300 and (len(labels) == 0 or labels[0].end - labels[0].begin < len(chars)):
                        chars = np.array(chars, dtype=np.str)
                        # labels = np.array([l.to_np() for l in labels], dtype=PhraseLabel.get_type())
                        examples.append(data.Example.fromlist([chars, labels], fields))
                except:
                    pass

        super(BaikeDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, fields: List,
               path=None, root='.data',
               train=None, validation=None, test=None, **kwargs):
        return super(BaikeDataset, cls).splits(
            path=path,
            root=root, fields=fields,
            train=train, validation=None, test=test, **kwargs)


def lazy_iter(fields,
              path: str, data_prefix: str, valid_file: str,
              batch_size=16, batch_size_fn=None,
              device=torch.device('cpu'),
              repeat=True,
              **kwargs):
    files = [file[len(path)+1:] for file in listfile(os.path.join(path, data_prefix))]
    random.shuffle(files)

    valid, *_ = BaikeDataset.splits(fields, path=path, train=valid_file)

    while True:
        for file in files:
            print(os.path.join(path, file))

            train, *_ = BaikeDataset.splits(fields, path=path, train=file)
            yield data.BucketIterator.splits(
                [train, valid],
                batch_sizes=[batch_size, batch_size*2],
                batch_size_fn=batch_size_fn,
                shuffle=True,
                sort_within_batch=True,
                device=device, **kwargs)
        if not repeat:
            break
