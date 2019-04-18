#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import os
import json
import gzip
import re
import itertools
import random
import jieba
import nltk
from flashtext import KeywordProcessor
import numpy as np
from collections import Counter, OrderedDict
from typing import List, Tuple

import torch
from torchtext import data
from torchtext.data import Dataset
from tqdm import tqdm
from pyhanlp import *

from .base import Label, PhraseLabel, listfile
from .extractor import Extractor


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


class PhraseField(data.Field):
    max_length = 15

    def __init__(self, name, *args, **kwargs):
        self.name = name
        super(PhraseField, self).__init__(sequential=False, *args, **kwargs)

    def process(self,
                batch: List[List[PhraseLabel]],
                device=None) -> List[List[Tuple[int, int, int, float]]]:

        return [self.process_sentence(phrases) for phrases in batch]

    def process_sentence(self, phrases: List[PhraseLabel]):

        def _process(phrase: PhraseLabel) -> Tuple[int, float]:
            if 'baike' in phrase.labels:
                return phrase.labels['baike'], 1.0
            if 'hanlp' in phrase.labels:
                return phrase.labels['hanlp'], 0.5
            if 'distant' in phrase.labels:
                return phrase.labels['distant'], 0.2
            if 'unlabel' in phrase.labels:
                return phrase.labels['unlabel'], 0.1

            return None, None

        results = []
        for phrase in phrases:
            flag, weight = _process(phrase)
            if flag is not None:
                results.append((phrase.begin + 1, phrase.end + 1, flag, weight))

        return results

    def make_noise(self, phrases: List[PhraseLabel]):
        seq_length = max([phrase.end for phrase in phrases])
        baike = [phrase for phrase in phrases if 'baike' in phrase]
        baike = self._baike_negative(baike, seq_length)
        offsets = [ for phrase in baike]

    def _baike_negative(self, labels: List[PhraseLabel], length):
        labels = sorted(labels, key=lambda p: (p.begin, p.end))
        negatives = []

        negatives.extend(self._pairwise(labels))
        negatives.extend(self._intersect(labels, length))

        return labels + negatives

    def _pairwise(self, phrases: List[PhraseLabel]):
        first_it, second_it = itertools.tee(phrases)
        next(second_it, None)
        for first, second in zip(first_it, second_it):
            for left in range(first.begin, first.end):
                for right in range(second.begin, second.end):
                    if left != first.begin or right != second.end:
                        yield PhraseLabel(left, right, baike=False)

    def _intersect(self, phrases: List[PhraseLabel], length):
        for phrase in phrases:
            for mid in range(phrase.begin + 1, phrase.end - 1):
                if mid - phrase.begin > 0:
                    yield PhraseLabel(phrase.begin, mid, unlabel=False)
                if phrase.end - mid > 0:
                    yield PhraseLabel(mid, phrase.end, unlabel=False)
                if phrase.begin > 0:
                    yield PhraseLabel(random.randint(max(0, phrase.begin - self.max_length), phrase.begin), mid, baike=False)
                if phrase.end < length:
                    yield PhraseLabel(mid, random.randint(phrase.end + 1, min(phrase.end + self.max_length + 1, length)), baike=False)


filter_pattern = re.compile('(merge_red.sh|merge_map.sh)')

close_pattern = re.compile('(（[^）]+）|【[^】]+】|《[^》]+》)')
noise_max_length = 10

class BaikeDataset(Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path: str, fields: List, **kwargs):
        self.extractor = kwargs.pop('extractor')
        noise_count = 0
        good_count = 0
        examples = []
        with gzip.open(path, mode='rt', compresslevel=6) as file:
            for line in tqdm(file, desc=('load dataset from %s' % path)):
                text, *labels = line.split('\t\t')
                if filter_pattern.search(text):
                    continue

                # words = words.split(' ')
                chars = list(text)
                try:
                    labels = [PhraseLabel.from_json(label) for label in labels if len(label) > 0]
                    for label in labels:
                        label.labels['baike'] = True

                    labels = self.extractor.extract(text, labels)

                    if 10 < len(chars) < 300:
                        chars = np.array(chars, dtype=np.str)
                        # labels = np.array([l.to_np() for l in labels], dtype=PhraseLabel.get_type())
                        examples.append(data.Example.fromlist([chars, labels], fields))

                except Exception as e:
                    print(e.with_traceback())
                    pass

                if len(examples) > 200000:
                    break
        print('%d sentence have %d noise phrases and %d good phrases.' % (len(examples), noise_count, good_count))
        super(BaikeDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls,
               fields: List,
               extractor,
               path=None, root='.data',
               train=None, validation=None, test=None,
               **kwargs):
        return super(BaikeDataset, cls).splits(
            path=path,
            root=root,
            train=train, validation=None, test=test,
            extractor=extractor,
            fields=fields,
            **kwargs)


def lazy_iter(fields,
              path: str, data_prefix: str, valid_file: str,
              distant_dict: str,
              batch_size=16, batch_size_fn=None,
              device=torch.device('cpu'),
              repeat=True,
              **kwargs):

    files = [file[len(path)+1:] for file in listfile(os.path.join(path, data_prefix))]
    random.shuffle(files)

    extractor = Extractor.build_from_dict(os.path.join(path, distant_dict))
    valid, *_ = BaikeDataset.splits(fields, extractor, path=path, train=valid_file)

    while True:
        for file in files:
            print(os.path.join(path, file))

            train, *_ = BaikeDataset.splits(fields, extractor, path=path, train=file)
            yield data.BucketIterator.splits(
                [train, valid],
                batch_sizes=[batch_size, batch_size*2],
                batch_size_fn=batch_size_fn,
                shuffle=True,
                sort_within_batch=True,
                device=device, **kwargs)
        if not repeat:
            break
