#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup, PageElement, Tag, NavigableString
import sys
from task.util import utils
from typing import Generator


def label(html):
    def _label(node: PageElement):
        if isinstance(node, NavigableString):
            chars = [word for t, word in utils.replace_entity(node.strip())]
            if node.parent.name == 'a' and len(chars) > 0:
                tags = ['S_'] if len(chars) == 1 else ['B_'] + ['M_'] * (len(chars) - 2) + ['E_']
                yield from zip(chars, tags)
            else:
                yield from zip(chars, [''] * len(chars))
        elif isinstance(node, Tag):
            for child in node.children:
                yield from _label(child)

    html = BeautifulSoup(html, 'html.parser')
    # 过滤链接较少的数据

    return list(_label(html))


def ChineseSplitter():
    ends = '。！？'
    pairs = {'(': ')', '{': '}', '[': ']', '<': '>',  '《': '》', '（': '）', '【': '】', '“': '”'}
    left2id = {}
    right2id = {}
    sames = {'"', '\''}
    same2id = {}
    for i, (k, v) in enumerate(pairs.items()):
        left2id[k] = i
        right2id[v] = i

    for i, s in enumerate(sames):
        same2id[s] = i

    def split_sentence(data: list):
        same_count = [0] * len(same2id)
        pair_count = [0] * len(left2id)

        begin = 0
        for pos, (word, type) in enumerate(data):
            if word in ends:
                if sum(same_count) == 0 and sum(pair_count) == 0:
                    yield data[begin:pos + 1]
                    begin = pos + 1
            elif word in left2id:
                pair_count[left2id[word]] += 1
            elif word in right2id:
                pair_count[right2id[word]] -= 1
            elif word in same2id:
                count = same_count[same2id[word]]
                same_count[same2id[word]] = (count + 1) % 2

        if begin < len(data):
            yield data[begin:]

    return split_sentence


splitter = ChineseSplitter()


def ratio(sentence):
    count = 0
    for word, type in sentence:
        if len(type) > 0:
            count += 1

    return count > 0


if __name__ == '__main__':
    with open(sys.argv[1]) as reader, open(sys.argv[2], 'w') as writer:
        for line in reader:
            import json
            line = json.loads(line)
            for sentence in splitter(label(line)):
                if len(sentence) > 0 and ratio(sentence) > 0.2:
                    sentence = ' '.join([word + '#' + type for word, type in sentence])
                    writer.write(sentence + '\n')



