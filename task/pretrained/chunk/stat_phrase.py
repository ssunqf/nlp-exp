#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import sys
import re

from tqdm import tqdm
from typing import List
from nltk.tree import Tree
from collections import Counter
from nltk.corpus import LazyCorpusLoader, BracketParseCorpusReader

corpus = LazyCorpusLoader(
    'ctb8', BracketParseCorpusReader, r'chtb_.*\.txt')


def load_dict(files: List[str], sep='\t'):
    words = set()
    for file in files:
        with open(file) as input:
            for line in tqdm(input, desc='load %s' % file):
                word, *_ = line.strip().split(sep)
                if len(word) > 1:
                    words.add(word)
    print('word size: %s' % len(words))
    assert '大学生运动会' in words
    return words


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
    return text, spans


if __name__ == '__main__':
    words = {} # load_dict(sys.argv[1:])

    counter = Counter()
    for id, sent in enumerate(corpus.parsed_sents()):
        print(id, sent)
        text, spans = with_offset(sent)
        print(text)
        for span, label in spans.items():
            print(span, label, text[span[0]:span[1]])

