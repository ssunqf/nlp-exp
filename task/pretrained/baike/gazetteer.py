#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import math
from typing import Dict, Iterable, List, Tuple
from collections import defaultdict, Counter, OrderedDict
import json
from .base import listfile, smart_open
import os
from torch import nn
from torchtext import data
from torchtext.data import Dataset
from .data import Field

import torch

class Node:
    def __init__(self, tag=None):
        self.tags = set()
        if tag:
            self.tags.add(tag)
        self.children = {}

    def add_child(self, k):
        node = self.children.get(k, None)
        if node is None:
            node = Node()
            self.children[k] = node
        return node

    def get_child(self, k):
        return self.children.get(k, None)

class Trie:
    def __init__(self):
        self.root = Node()

    def add(self, key: Iterable, value):
        node = self.root
        for k in key:
            node = node.add_child(k)

        node.tags.add(value)

    def find(self, key: Iterable):
        node = self.root
        for i, k in enumerate(key):
            node = node.get_child(k)
            if node is None:
                return
            if len(node.tags) > 0:
                yield node.tags, i+1


def to_bmes(length, tag):
    if length == 1:
        return ['S_%s' % tag]
    elif length > 1:
        return ['B_%s' % tag] + ['M_%s' % tag] * (length - 2) + ['E_%s' % tag]
    else:
        raise RuntimeError('length must be big than 0.')


class Gazetteer:
    def __init__(self, data: Trie, type_counter: Counter, max_length=15):
        self.data = data
        self.type_counter = Counter
        self.max_length = max_length

    def find(self, words: List) -> List[List[str]]:
        token_tags = [set() for _ in range(len(words))]
        for begin in range(len(words)):
            length = min(len(words)-begin, self.max_length)
            # 同类型保留最长的
            for phrase_tags, phrase_len in self.data.find(words[begin:begin+length]):
                for tag in phrase_tags:
                    for offset, token_tag in enumerate(to_bmes(phrase_len, tag)):
                        token_tags[begin+offset].add(token_tag)

        return [list(tags) for tags in token_tags]

    @classmethod
    def from_file(cls, path: str, max_length=-1):
        type_stats = Counter()
        trie = Trie()
        token_max_length = 0
        with open(path) as file:
            for line in file:
                line = line.strip()
                if len(line) > 0:
                    phrase, type = line.rsplit('\t', maxsplit=1)
                    if max_length <= 0 or len(phrase) < max_length:
                        type_stats['type_%d' % min(len(phrase), 3)] += 1
                        trie.add(phrase, type)
                        token_max_length = max(token_max_length, len(phrase))

        return cls(trie, type_stats, token_max_length)

    @classmethod
    def from_files(cls, pathes: List[Tuple[str, str]], max_length=-1):
        type_stats = Counter()
        trie = Trie()
        token_max_length = 0
        print(pathes)
        for name, path in pathes:
            with open(path) as file:
                for line in file:
                    line = line.strip()
                    if len(line) > 0:
                        phrase, *_ = line.split('\t')
                        if max_length <= 0 or len(phrase) < max_length:
                            type_stats['type_%d' % min(len(phrase), 3)] += 1
                            trie.add(phrase, name)
                            token_max_length = max(token_max_length, len(phrase))

        return cls(trie, type_stats, token_max_length)

    @classmethod
    def from_jieba_pos(cls, path, max_length=-1):
        type_stats = Counter()
        trie = Trie()
        token_max_length = 0
        with open(path) as file:
            for line in file:
                line = line.strip()
                if len(line) > 0:
                    phrase, count, type = line.rsplit(maxsplit=2)
                    if max_length <= 0 or len(phrase) < max_length:
                        type_stats['type_%d' % min(len(phrase), 3)] += 1
                        trie.add(phrase, type)
                        token_max_length = max(token_max_length, len(phrase))

        return cls(trie, type_stats, token_max_length)


class GazetteerField(Field):
    def __init__(self, name, gazetteer, *args, **kwargs):
        self.name = name
        self.gazetteer = gazetteer
        super(GazetteerField, self).__init__(sequential=True, *args, **kwargs)

    def process(self, batch: List[List[str]], device=None):

        batch_size = len(batch)
        lens = [len(sen) for sen in batch]
        seq_len = max(lens) + 2
        var = torch.zeros(seq_len, batch_size, len(self.vocab), dtype=torch.float, device=device)

        for bid, sen in enumerate(batch):
            for sid, token in enumerate(self.gazetteer.find(sen)):
                for tag in token:
                    tid = self.vocab.stoi[tag]
                    if tid:
                        var[sid+1, bid, tid] = 1

        return var


class WordDictionary(Field):
    def __init__(self, word_probs: Dict[str, float], max_word_length, min_freq, *args, **kwargs):
        super(WordDictionary, self).__init__(sequential=True, *args, **kwargs)
        self.word_probs = word_probs
        self.max_word_length = max_word_length
        self.min_freq = min_freq
        self.flag_length = 5

    def process_sentence(self, sentence: List[str]):
        sen_len = len(sentence)
        flags = [[0.] * (self.flag_length * 2) for _ in range(sen_len)]
        for begin in range(sen_len):
            for end in range(begin+2, min(sen_len, begin+self.max_word_length)):
                phrase = ''.join(sentence[begin:end])
                probs = self.word_probs.get(phrase, None)
                if probs:
                    index = min(end - begin, 6) - 2
                    print(phrase, probs, index)
                    flags[begin][2*index+1] = max(probs, flags[begin][2*index+1])
                    flags[end-1][2*index] = max(probs, flags[end-1][2*index])

        return flags

    def process(self, batch: List[List[str]], device=None):
        batch_size = len(batch)
        lens = [len(sen) for sen in batch]
        seq_len = max(lens) + 2
        var = torch.zeros(seq_len, batch_size, self.flag_length * 2, dtype=torch.float, device=device)
        for bid, sentence in enumerate(batch):
            sen_len = len(sentence)
            for begin in range(sen_len):
                for end in range(begin+2, min(sen_len, begin+self.max_word_length)):
                    phrase = ''.join(sentence[begin:end])
                    prob = self.word_probs.get(phrase, None)
                    if prob:
                        index = min(end - begin, self.flag_length+1) - 2
                        var[begin+1, bid, 2*index+1] = max(prob, var[begin+1, bid, 2*index+1])
                        var[end, bid, 2*index] = max(prob, var[end, bid, 2*index])
        return var

    @staticmethod
    def load(path: str, **kwargs):
        with smart_open(path) as file:
            lfreq = {}
            ltotal = 0.
            max_freq = 0
            max_length = 0
            for lineno, line in enumerate(file, 1):
                try:
                    line = line.strip()
                    word, freq, type = line.rsplit(' ', maxsplit=2)
                    freq = float(freq)
                    lfreq[word] = freq
                    ltotal += freq
                    max_freq = max(freq, max_freq)
                    max_length = max(max_length, len(word))
                except ValueError:
                    raise ValueError(
                        'invalid dictionary entry in %s at Line %s: %s' % (file, lineno, line))

            return WordDictionary({k: v/max_freq for k, v in lfreq.items()}, max_length, min_freq=1/max_freq, **kwargs)


class TagEmbedding(nn.Module):
    def __init__(self, type_size, *args, **kwargs):
        super(TagEmbedding, self).__init__()
        self.type_size = type_size
        self.embed = nn.Embedding(*args, **kwargs)

    def forward(self, tags: torch.Tensor, probs: torch.Tensor):
        seq_len, batch_size, tag_len = tags.size()
        assert tag_len == self.type_size
        embed = self.embed(tags) * probs.unsqueeze(-1).view_as(seq_len, batch_size, -1)

        return embed


if __name__ == '__main__':

    dict = WordDictionary.load('./gazetteers/jieba.pos.dic')

    print(dict.process([list('电流是指一群电荷的流动')]))

