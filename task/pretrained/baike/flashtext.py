#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import gzip
import os
import json
import re
import numpy as np
from typing import List
from collections import defaultdict
from tqdm import tqdm

from .tokenizer import hanlp_tokenizer


class KeywordProcessor:
    value_name = 'END_WITH_VALUE'

    def __init__(self, tokenize_func=lambda text: text, case_sensitive=False):
        self.root = {}
        self.tokenize_func = tokenize_func
        self.case_sensitive = case_sensitive
        self.number_keyword = 0
        self.unk_id = 0

    def __setitem__(self, key: str, value=None):

        status = False
        if value is None:
            value = key
        if key and value:
            if not self.case_sensitive:
                key = key.lower()
            node = self.root
            for word in self.tokenize_func(key):
                node = node.setdefault(word, {})
            if self.value_name not in node:
                self.number_keyword += 1
                status = True
                node[self.value_name] = value
        return status

    def __len__(self):
        return self.number_keyword

    def __getitem__(self, key):
        node = self.root
        if not self.case_sensitive:
            key = key.lower()

        node = self.root
        for w in self.tokenize_func(key):
            node = node.get(w)
            if node is None:
                return None
        return node.get(self.value_name, None)

    def __contains__(self, key):
        if not self.case_sensitive:
            key = key.lower()

        node = self.root
        for w in self.tokenize_func(key):
            node = node.get(w)
            if node is None:
                return False
        return self.value_name in node

    def add_keyword(self, key: str, value=None):
        self.__setitem__(key, value)

    def extract_keywords(self, text: str, words: List[str]=None, span_info=False, longest=True):
        if words is None:
            words = self.tokenize_func(text)

        extracted = []
        begin = 0
        while begin < len(words):
            end = begin
            matched_ends = []
            node = self.root
            while end < len(words):
                node = node.get(words[end], None)

                if not node:
                    break
                if self.value_name in node:
                    matched_ends.append((node[self.value_name], (begin, end)))
                end += 1

            if longest and len(matched_ends) > 0:
                extracted.append(matched_ends[-1])
                # begin = matched_ends[-1][1][1]
            else:
                extracted.extend(matched_ends)

            begin += 1

        if span_info:
            offsets = []
            offset = 0
            for word in words:
                try:
                    offset = text.index(word, offset)
                except Exception as e:
                    pass
                offsets.append((offset, offset+len(word)))
                offset += len(word)

            extracted = [(value, offsets[begin][0], offsets[end][1])
                         for value, (begin, end) in extracted]
        else:
            extracted = [value for value, _ in extracted]

        return extracted

    def from_list(self, file, sep='\t'):
        """
        :param file: 优先从json文件中读取，修改字典必须删掉json文件
        :param sep:
        :return:
        """
        if os.path.exists(file + '.json'):
            with gzip.open(file + '.json', mode='rt', compresslevel=6) as input:
                self.root = json.load(input)
        else:
            with open(file) as input:
                for line in tqdm(input, desc='load keyword from %s' % file):
                    keyword, *_ = line.split(sep, maxsplit=1)
                    if len(keyword) > 1:
                        self.add_keyword(keyword)

            with gzip.open(file + '.json', mode='wt', compresslevel=6) as output:
                json.dump(self.root, output, ensure_ascii=False)


class TripleProcessor:
    SEP = re.compile(', ')
    def __init__(self):
        self.stopwords = {}
        self.keyword = KeywordProcessor()
        self.src2id = {}
        self.id2src = []
        self.rel2id = {}
        self.id2rel = []
        self.tgt2id = {}
        self.id2tgt = []

        self.src2triple = defaultdict(set)
        self.tgt2triple = defaultdict(set)
        self.triples = []

    def extract_triple(self, text: str, words: List[str]):
        extracted = self.keyword.extract_keywords(text, words, span_info=True, longest=False)

        keywords = set([word for word, _, _ in extracted])
        src_triples = [self.src2triple[self.src2id[word]] if word in self.src2id else set() for word in keywords]
        tgt_triples = [self.tgt2triple[self.tgt2id[word]] if word in self.tgt2id else set() for word in keywords]

        triples = set()
        for src, src_str in zip(src_triples, keywords):
            for tgt, tgt_str in zip(tgt_triples, keywords):
                if src_str != tgt_str:
                    triples.update(src.intersection(tgt))

        triples = [self.triples[triple_id] for triple_id in triples]
        return extracted, [(self.id2src[src], self.id2rel[rel], self.id2tgt[tgt]) for src, rel, tgt in triples]

    def optimize(self):
        '''
        optimize memory usage
        :return:
        '''
        self.id2src = np.array(self.id2src)
        self.id2rel = np.array(self.id2rel)
        self.id2tgt = np.array(self.id2tgt)
        self.triples = np.array(self.triples)

        return self

    def to_json_file(self, file:str):
        with open(file, 'w') as output:
            json.dump(
                {
                    'stopwords': self.stopwords,
                    'keyword': self.keyword,
                    'id2src': self.id2src,
                    'id2rel': self.id2rel,
                    'id2tgt': self.id2tgt,
                    'src2triple': self.src2triple,
                    'tgt2triple': self.tgt2triple,
                    'triples': self.triples
                },

                output, ensure_ascii=False)

    def from_json_file(self, file: str):
        with open(file) as input:
            data = json.load(input)
            self.stopwords = data['stopwords']
            self.keyword = data['keyword']
            self.id2src = data['id2src']
            self.src2id = {src:i for i, src in enumerate(self.id2src)}
            self.id2rel = data['id2rel']
            self.rel2id = {rel:i for i, rel in enumerate(self.id2rel)}
            self.id2tgt = data['id2tgt']
            self.tgt2id = {tgt:i for i, tgt in enumerate(self.id2tgt)}
            self.src2triple = data['src2triple']
            self.tgt2triple = data['tgt2triple']
            self.triples = data['triples']

    def build(self, stopword_file: str, triple_files: List[str], entity_files: List[str]):
        self._load_stopword(stopword_file)
        for file in triple_files:
            self._from_csv(file)
        for file in entity_files:
            self._from_dict(file)

    def _load_stopword(self, file):
        with open(file) as input:
            self.stopwords = {line.strip() for line in input.readlines()}

    def _from_csv(self, file):
        with open(file, 'r') as input:
            names = self.SEP.split(input.readline())
            for line in tqdm(input, desc='load triples from %s' % file):
                row = self.SEP.split(line.strip())
                src, rel, *tgts = row
                pos = src.find('[')
                if pos >= 0:
                    attr = src[pos + 1:-1]
                    self.keyword.add_keyword(attr)
                    src = src[:pos]

                for tgt in tgts:
                    if src == tgt or src in self.stopwords or rel in self.stopwords or tgt in self.stopwords:
                        continue
                    src_id = self.src2id.setdefault(src, len(self.src2id))
                    if src_id == len(self.id2src):
                        self.id2src.append(src)
                    rel_id = self.rel2id.setdefault(rel, len(self.rel2id))
                    if rel_id == len(self.id2rel):
                        self.id2rel.append(rel)
                    tgt_id = self.tgt2id.setdefault(tgt, len(self.tgt2id))
                    if tgt_id == len(self.id2tgt):
                        self.id2tgt.append(tgt)

                    self.triples.append((src_id, rel_id, tgt_id))
                    triple_id = len(self.triples) - 1
                    self.src2triple[src_id].add(triple_id)
                    self.tgt2triple[tgt_id].add(triple_id)

                    self.keyword.add_keyword(src)
                    self.keyword.add_keyword(tgt)
                    self.keyword.add_keyword(rel)

                if len(self.triples) > 1e6:
                    break

            print('src: %d, rel: %d, tgt: %d, triple: %d' %
                  (len(self.src2id), len(self.rel2id), len(self.tgt2id), len(self.triples)))

    def _from_dict(self, file, sep='\t'):
        with open(file) as input:
            for line in tqdm(input, desc='load keyword from %s' % file):
                keyword, *_ = line.split(sep)
                if len(keyword) > 1:
                    self.keyword.add_keyword(keyword)


if __name__ == '__main__':
    import sys
    extractor = KeywordProcessor(tokenize_func=lambda p: p)
    extractor.from_list(sys.argv[1])
    #extractor.from_csv(sys.argv[2])
    print("finished loading %s" % sys.argv[1])
    for line in sys.stdin:
        line = line.strip()
        print(line in extractor)
        print(extractor.extract_keywords(line))