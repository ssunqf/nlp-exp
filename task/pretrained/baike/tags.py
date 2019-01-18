#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import math
from typing import List, Tuple, Dict
from collections import Counter, OrderedDict
from jieba import posseg
from .base import to_BMES, mixed_open
import torch
from torch import nn
from torchtext import data
from torchtext.data import Dataset


class Tagger:
    def tag(self, sentence: List[str]) -> Tuple[List[str], List[float]]:
        raise NotImplementedError()

class JiebaTagger(Tagger):
    def __init__(self):
        self.name = 'jieba_pos'

    def tag(self, sentence: List[str]) -> Tuple[List[str], List[float]]:
        tags = []
        for word, type in posseg.cut(''.join(sentence)):
            tags.extend(to_BMES(len(word), type))
        return tags, [1.0] * len(sentence)


class DictTagger(Tagger):
    def __init__(self, name, words, max_length):
        super(DictTagger, self).__init__()
        self.name = name
        self.words = words
        self.max_length = max_length

    def tag(self, sentence: List[str]) -> Tuple[List[str], List[float]]:
        tags = ['O_%s' % self.name] * len(sentence)
        sentence = ''.join(sentence)
        for begin in range(len(sentence)):
            for end in range(min(begin+self.max_length, len(sentence)), begin, -1):
                if sentence[begin:end] in self.words:
                    for id, tag in zip(range(begin, end), to_BMES(end-begin, self.name)):
                        if tags[id] is None or tags[id][0:2] in {'M_', 'O_'}:
                            tags[id] = tag

        return tags, [1.0] * len(sentence)

    @classmethod
    def load(cls, name, path, sep='\t'):
        words = set()
        max_length = 0
        with mixed_open(path) as file:
            for lineno, line in enumerate(file, 1):
                line = line.strip()
                if sep is None:
                    word = line
                else:
                    word, *_ = line.split(sep)

                words.add(word)
                max_length = max(max_length, len(word))

        return cls(name, words, max_length)


class DictWithTypeTagger(Tagger):
    def __init__(self, name, words, max_length):
        super(DictWithTypeTagger, self).__init__()
        self.name = name
        self.words = words
        self.max_length = max_length

    def tag(self, sentence: List[str]) -> Tuple[List[str], List[float]]:
        tags = ['O_%s' % self.name] * len(sentence)
        probs = [0.] * len(sentence)
        sentence = ''.join(sentence)
        for begin in range(len(sentence)):
            for end in range(min(begin + self.max_length, len(sentence)), begin, -1):
                prob_with_type = self.words.get(sentence[begin:end], None)
                if prob_with_type:
                    prob, type = prob_with_type
                    for id, tag in zip(range(begin, end), to_BMES(end - begin, type)):
                        if tags[id] is None or tags[id][0:2] in {'M_', 'O_'}:
                            tags[id] = tag
                            probs[id] = prob

        return tags, probs

    @classmethod
    def load(cls, name, path):
        words = {}
        max_length = 0
        max_freq = 0
        with mixed_open(path) as file:
            for lineno, line in enumerate(file, 1):
                line = line.strip()
                word, freq, type = line.rsplit(maxsplit=2)
                freq = float(freq)
                words[word] = (freq, type)
                max_length = max(max_length, len(word))
                max_freq = max(max_freq, freq)

        return cls(name,
                   {word: (math.sqrt(freq/max_freq), type) for word, (freq, type) in words.items()},
                   max_length)


class RadicalTagger(Tagger):
    name = 'radical'
    def __init__(self, data: Dict[str, str]):
        super(RadicalTagger, self).__init__()
        self.data = data

    def tag(self, sentence: List[str]) -> Tuple[List[str], List[float]]:
        return [self.data.get(token, 'None') for token in sentence]

    @classmethod
    def load(cls, path):
        data = {}
        with open(path) as file:
            for line in file:
                line = line.strip()
                if len(line) > 0:
                    char, radical = line.split(':')
                    data[char] = radical
        return RadicalTagger(data)


class TagField(data.Field):
    def __init__(self, tagger: Tagger, has_init=True, has_eos=True, *args, **kwargs):
        super(TagField, self).__init__(*args, **kwargs)
        self.tagger = tagger
        self.sequential = True
        self.has_init=has_init
        self.has_eos=has_eos

    @property
    def name(self):
        return self.tagger.name

    def build_vocab(self, *args, **kwargs):
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                tags, probs = self.tagger.tag(x)
                counter.update(tags)
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None))

        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)
        self.unk_token_idx = self.vocab.stoi[self.unk_token]
        self.pad_token_idx = self.vocab.stoi[self.pad_token]
        self.init_token_idx = self.vocab.stoi[self.init_token]
        self.eos_token_idx = self.vocab.stoi[self.eos_token]

    def process(self, batch: List[List[str]], device=None):

        batch_size = len(batch)
        lens = [len(sen) for sen in batch]
        seq_len = max(lens) + 2
        batch_tags = torch.LongTensor(seq_len, batch_size, device=device)
        batch_probs = torch.FloatTensor(seq_len, batch_size, device=device)
        for bid, sentence in enumerate(batch):
            tags, probs = self.tagger.tag(sentence)
            batch_tags[:, bid] = torch.LongTensor(
                [self.unk_token_idx] + \
                [self.vocab.stoi.get(tag, self.unk_token_idx) for tag in tags] + \
                [self.unk_token_idx] +
                [self.unk_token_idx] * (seq_len - len(sentence) - 2),
                device=device)

            batch_probs[:, bid] = torch.FloatTensor(
                [1.0] + probs + [1.0] + [1.0] * (seq_len - len(sentence) - 2),
                device=device)

        return batch_tags, batch_probs


class TagEmbedding(nn.Module):
    def __init__(self, name, tag_size, tag_dim, padding_idx):
        super(TagEmbedding, self).__init__()
        self.name = name
        self.embedding = nn.Embedding(tag_size, tag_dim, padding_idx=padding_idx)

    def forward(self, tags, probs):
        emb = self.embedding(tags)
        return emb * probs.unsqueeze(-1)


jieba_pos = JiebaTagger()
digit = DictTagger.load('digit', './gazetteers/chnDigit.dic')
person = DictTagger.load('name', './gazetteers/chnName.dic')
place = DictTagger.load('place', './gazetteers/chnPlace.dic')
quantifier = DictTagger.load('quantifier', './gazetteers/chnQuantifier.dic')
stopword = DictTagger.load('stopword', './gazetteers/chnStopWord.dic')
phrase = DictTagger.load('phrase', './gazetteers/phrase.dic')
posdict = DictWithTypeTagger.load('posdict', './gazetteers/jieba.pos.dic')
idioms = DictTagger.load('idioms', './gazetteers/idioms.txt')
organizations = DictTagger.load('organization', './gazetteers/org.all', None)

radical = RadicalTagger.load('./gazetteers/radicals.txt')
