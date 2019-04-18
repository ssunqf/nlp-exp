#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import gzip
import os
import json
from typing import List
from tqdm import tqdm
from pyhanlp import HanLP


def hanlp_tokenizer(text: str) -> List[str]:
    return [term.word for term in HanLP.segment(text)]


class KeywordProcessor:
    value_name = 'END_WITH_VALUE'
    def __init__(self, case_sensitive=False):
        self.root = {}
        self.tokenizer = hanlp_tokenizer
        self.case_sensitive = case_sensitive
        self.number_keyword = 0
        self.unk_id = 0

    def __setitem__(self, key, value=None):

        status = False
        if value is None:
            value = key
        if key and value:
            key = key.strip()
            value = value.strip()
            if not self.case_sensitive:
                key = key.lower()
            node = self.root
            for word in self.tokenizer(key):
                node = node.setdefault(word, {})
            if self.value_name not in node:
                self.number_keyword += 1
                status = True
                node[self.value_name] = value
        return status

    def __len__(self):
        return self.number_keyword

    def add_keyword(self, key: str, value=None):
        self.__setitem__(key, value)

    def extract_keywords(self, text: str, span_info=False, longest=True):
        words = self.tokenizer(text)
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
                    keyword, *_ = line.split(sep)
                    if len(keyword) > 1:
                        self.add_keyword(keyword)

            with gzip.open(file + '.json', mode='wt', compresslevel=6) as output:
                json.dump(self.root, output, ensure_ascii=False)


if __name__ == '__main__':
    import sys
    extractor = KeywordProcessor()
    extractor.from_list(sys.argv[1])
    for line in sys.stdin:
        print(line)
        for value, (begin, end) in extractor.extract_keywords(line, span_info=True):
            print(value, line[begin:end])