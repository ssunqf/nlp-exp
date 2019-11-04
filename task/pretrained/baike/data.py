#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import gzip
import itertools
import json
import random
import re
from collections import Counter
from collections import OrderedDict
from typing import List
from typing import Tuple, Union

from nltk.tree import Tree
from nltk.corpus import LazyCorpusLoader, BracketParseCorpusReader

import numpy as np
import torch
from pyhanlp import *
from torchtext import data
from torchtext.data import Dataset
from tqdm import tqdm
import multiprocessing

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

    def make_negative(self, spans: List[Tuple[int, int]], tag):
        def _pairwise(first_it, second_it):
            for first, second in zip(first_it, second_it):
                if second.begin - first.end < self.max_length:
                    for left in range(first.begin, first.end):
                        for right in range(second.begin + 1, second.end):
                            if left != first.begin or right != second.end:
                                yield PhraseLabel(left, right, **{tag: False})

        first_it, second_it = itertools.tee(spans)
        _pairwise(first_it, second_it)




filter_pattern = re.compile('(merge_red.sh|merge_map.sh)')

close_pattern = re.compile('(（[^）]+）|【[^】]+】|《[^》]+》)')
noise_max_length = 10


class BaikeDataset(Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, dataset: Union[str, List], fields: List, **kwargs):
        self.extractor = kwargs.pop('extractor')
        noise_count = 0
        good_count = 0
        examples = []

        def _source(path):
            with gzip.open(path, mode='rt', compresslevel=6) as file:
                for line in tqdm(file, desc=('load dataset from %s' % path)):
                    yield line

        source = _source(dataset) if isinstance(dataset, str) else dataset

        for line in source:
            text, *labels = line.split('\t\t')
            text = text.replace(u'\u3000', ' ')

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

            #if len(examples) > 500000:
            #    break

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
              path: str, train_prefix: str, valid_file: str,
              distant_dict: str,
              batch_size=16, batch_size_fn=None,
              device=torch.device('cpu'),
              repeat=True,
              **kwargs):

    files = [file[len(path)+1:] for file in listfile(os.path.join(path, train_prefix))]
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


corpus = LazyCorpusLoader('ctb8', BracketParseCorpusReader, r'chtb_.*\.txt')


def is_eng_num(c):
    return 'a' <= c <= 'z' or 'A' <= c <= 'Z' or '0' <= c <= '9'


def is_phrase(tree: Tree):

    if tree.height() == 2:
        return tree.label() in {'NN', 'NR', 'NT', 'VV', 'CD', 'M'}
    labels = tree.label().split('-')
    if tree.height() >= 3:
        if labels[0] in {'QP'}:
            return True
        elif labels[0] == 'NP':
            tags = set(child.label() for child in tree)
            if len(tags.intersection({'ETC', 'DEC', 'DNP'})) > 0:
                return False
            if len(tags.difference({'NN', 'NR'})) == 0:
                return True
        elif labels[0] in {'DNP', 'VP', 'LCP', 'DVP', 'PP', 'DP'}:
            return False

    return None


def with_offset(tree: Tree):
    text = ''
    for subtree in tree.subtrees(lambda t: t.height() == 2):
        if subtree.label() == '-NONE-':
            subtree[:] = [(len(text), len(text), subtree[0])]
            continue

        offset = len(text)
        if len(text) > 0 and is_eng_num(text[-1]) and is_eng_num(subtree[0]):
            text += ' '
            offset += 1
        text += subtree[0]

        subtree[:] = [(offset, len(text), subtree[0])]

    spans = {}
    for subtree in tree.subtrees(lambda t: t.height() >= 2):
        if subtree.label() == '-NONE-':
            continue
        if subtree.height() > 2 and len(subtree) == 1:
            continue

        subleaves = subtree.leaves()
        spans[(subleaves[0][0], subleaves[-1][1])] = is_phrase(subtree)

    negatives = {}
    for i in range(len(spans)):
        start = random.randint(0, len(text))
        end = random.randint(0, len(text))
        if start == end:
            continue
        if start > end:
            start, end = end, start
        if (start, end) not in spans:
            negatives[(start, end)] = False
    spans.update(negatives)
    return text, spans


def load_ctb8():
    for sent in tqdm(corpus.parsed_sents(), desc='loading ctb8 dateset'):
        text, spans = with_offset(sent)
        labels = [PhraseLabel(begin, end, **{'baike': label}) for (begin, end), label in spans.items() if label]
        yield text, labels


class CTBDataset(Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, fields: List, **kwargs):
        examples = []
        for text, labels in load_ctb8():
            chars = np.array(list(text), dtype=np.str)
            examples.append(data.Example.fromlist([chars, labels], fields))

        super(CTBDataset, self).__init__(examples, fields, **kwargs)


class DatasetIterator:
    BUFFER_SIZE = 100000

    def __init__(self, fields,
                 path: str, train_prefix: str, valid_file: str, distant_dict: str,
                 batch_size=16, batch_size_fn=None,
                 device=torch.device('cpu'),
                 repeat=True):
        self.fields = fields
        self.path = path
        self.train_prefix = train_prefix
        self.valid_file = valid_file
        self.extractor = Extractor.build_from_dict(os.path.join(path, distant_dict))

        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn

        self.device = device
        self.repeat = repeat

        self.offset = 0

        valid, *_ = BaikeDataset.splits(self.fields, self.extractor, path=path, train=valid_file)
        self.valid, *_ = data.BucketIterator.splits(
                [valid],
                batch_sizes=[self.batch_size * 2],
                batch_size_fn=self.batch_size_fn,
                shuffle=False,
                sort_within_batch=True,
                device=self.device)
        '''
        ctb = CTBDataset(self.fields)
        ctb8, *_ = data.BucketIterator.splits(
            [ctb],
            batch_sizes=[self.batch_size],
            batch_size_fn=self.batch_size_fn,
            shuffle=True,
            sort_within_batch=True,
            repeat=True,
            device=self.device)
        self.ctb8 = iter(ctb8)
        '''

    def _read_line(self, prefix):
        offset = 0
        for path in listfile(os.path.join(self.path, prefix)):
            with gzip.open(path, mode='rt', compresslevel=6) as file:
                for line in tqdm(file, desc='reading %s' % path):
                    line = line.strip()
                    offset += 1
                    if len(line) > 0 and offset > self.offset:
                        yield line

    def __iter__(self):
        def _sub_iter(examples):
            dataset = BaikeDataset(examples, self.fields, extractor=self.extractor)

            dataset_it, *_ = data.BucketIterator.splits(
                [dataset],
                batch_sizes=[self.batch_size],
                batch_size_fn=self.batch_size_fn,
                shuffle=True,
                sort_within_batch=True,
                device=self.device)

            for batch in dataset_it:
                yield batch
                # yield next(self.ctb8)

        buffers = []
        for line in self._read_line(self.train_prefix):
            buffers.append(line)
            if len(buffers) > self.BUFFER_SIZE:
                yield from _sub_iter(buffers)
                buffers = []

        if len(buffers) > 0:
            yield from _sub_iter(buffers)

        if self.repeat:
            yield from self.__iter__()

class ProcessIterator:
    def __init__(self, iterator):
        self.iterator = iterator

    def __iter__(self):
        queue = multiprocessing.Queue(1000)

        def __producer(iterator):
            for item in iterator:
                queue.put(item)

        process = multiprocessing.Process(target=__producer, args=(self.iterator,))
        process.start()

        while process.is_alive() or not queue.empty():
            yield queue.get()