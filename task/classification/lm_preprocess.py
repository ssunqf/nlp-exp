#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from pyhanlp import HanLP
from .utils import splitter
import sys
from collections import defaultdict

counts = defaultdict(int)


def preprocess(para: str):
    for sen in splitter(para):
        words = [term.word for term in HanLP.segment(sen)]
        for w in words:
            counts[w] += 1
        if len(words) > 5:
            yield ' '.join(words)


if __name__ == '__main__':
    for line in sys.stdin:
        line = line.strip()
        if len(line) > 0:
            for s in preprocess(line):
                print(s)

    with open('word.counts', 'w') as f:
        for w, c in sorted(counts.items(), key=lambda i: i[1], reverse=True):
            f.write('%s\t%d' % (w, c))
