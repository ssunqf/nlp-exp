#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import sys
from typing import List


def detokenize(words: List[str]):
    def _isalphnum(char):
        return '0' <= char <= '9' or 'a' <= char <= 'z' or 'A' <= char <= 'Z'

    for i in range(len(words)):
        if i > 0 and _isalphnum(words[i - 1][-1]) and _isalphnum(words[i][0]):
            yield ' '

        yield from words[i]


for line in sys.stdin:
    text, count, mi, nmi = line.split('\t')
    phrase = detokenize(text.split(' '))
    if nmi < 0.2:
        print('%s\t%d\t%f\t%f\n' % (phrase, count, mi, nmi))