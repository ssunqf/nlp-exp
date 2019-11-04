#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import sys
from tqdm import tqdm
from typing import Iterable, List, Tuple
from collections import Counter
from task.util import utils
import json


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


class Gazetteer:
    def __init__(self, data: Trie, max_length=15):
        self.data = data
        self.max_length = max_length

    def find(self, words: List) -> List[List[str]]:
        token_tags = [set() for _ in range(len(words))]
        for begin in range(len(words)):
            length = min(len(words)-begin, self.max_length)
            # 同类型保留最长的
            for phrase_tags, phrase_len in self.data.find(words[begin:begin+length]):
                for tag in phrase_tags:
                    yield (begin, begin+phrase_len, tag)

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

        return cls(trie, token_max_length)

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

        return cls(trie, token_max_length)

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

        return cls(trie, token_max_length)


def ChineseSplitter():
    ends = '。！？\n'
    pairs = {'(': ')', '{': '}', '[': ']', '<': '>', '《': '》', '（': '）', '【': '】', '“': '”'}
    left2id = {}
    right2id = {}
    sames = {'"', '\', '}
    same2id = {}
    for i, (k, v) in enumerate(pairs.items()):
        left2id[k] = i
        right2id[v] = i

    for i, s in enumerate(sames):
        same2id[s] = i

    def split_sentence(data: str):
        same_count = [0] * len(same2id)
        pair_count = [0] * len(left2id)

        begin = 0
        for pos, char in enumerate(data):
            if char in ends:
                if sum(same_count) == 0 and sum(pair_count) == 0:
                    if pos - begin > 1:
                        yield ''.join(data[begin:pos + 1])
                    begin = pos + 1
            elif char in left2id:
                pair_count[left2id[char]] += 1
            elif char in right2id:
                pair_count[right2id[char]] -= 1
            elif char in same2id:
                count = same_count[same2id[char]]
                same_count[same2id[char]] = (count + 1) % 2

        if begin < len(data) - 1:
            yield ''.join(data[begin:])

    return split_sentence


splitter = ChineseSplitter()


if __name__ == '__main__':

    gazetteer = Gazetteer.from_files([
        ('@@digit@@', './gazetteers/chnDigit.dic'),
        ('@@person@@', 'gazetteers/chnName.dic'),
        ('@@place@@', 'gazetteers/chnPlace.dic'),
        ('@@quantifier@@', 'gazetteers/chnQuantifier.dic'),
        ('@@stopword@@', 'gazetteers/chnStopWord.dic'),
    ])
    '''
    gazetteer = Gazetteer.from_jieba_pos('./gazetteers/jieba.pos.dic')
    '''

    gazetteer_counter = Counter()

    for line in sys.stdin:
        for sentence in splitter(line):
            if len(sentence) > 0:
                tags = list(gazetteer.find(sentence))
                phrase = [(begin, end, sentence[begin:end]) for begin, end, _ in tags]
                tokens = utils.replace_entity(sentence)
                unigram = [(begin, begin+1, sentence[begin:begin+1]) for begin in range(len(sentence))]

                lattice = unigram + phrase + tags
                print(json.dumps(unigram + phrase + tags, ensure_ascii=False))

                gazetteer_counter.update([str for _, _, str in lattice])

    print(gazetteer_counter.most_common())





