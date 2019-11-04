#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import sys
import math
from tqdm import tqdm
from nltk import Tree
from collections import Counter


def words(file):
    with open(file) as input:
        for lid, line in tqdm(enumerate(input)):
            line = line.strip()
            if len(line) == 0:
                continue
            id, *words = line.strip().split('  ')

            new_words = []
            for word in words:
                items = word.rsplit('/')

                word, pos = items[0:2]

                begin = 0
                while begin < len(word) and word[begin] == '[':
                    begin += 1
                word = word[begin:]

                pos = pos.split(']')[0]

                new_words.append('%s/%s' % (word, pos))
            yield new_words


def calc_grams(sentences):

    grams = Counter()
    for sentence in sentences:
        for length in range(1, 6):
            grams.update('  '.join(sentence[begin:begin+length])
                         for begin in range(max(0, len(sentence) - length + 1)))

    return grams


def calc_pmi(counter: Counter):
    total = sum(counter.values())

    for phrase, count in tqdm(counter.items()):
        words = phrase.split('  ')
        if len(words) == 1:
            continue

        max_xy = 0
        for begin in range(1, len(words)):
            px = '  '.join(words[0:begin])
            py = '  '.join(words[begin:])
            cx = counter.get(px)
            cy = counter.get(py)
            max_xy = max(max_xy, cx * cy)

        pxy = count / total
        pxpy = max_xy / (total * total)
        pmi = math.log(pxy / pxpy)
        npmi = pmi / -math.log(pxy)
        yield phrase, count, pmi, npmi


if __name__ == '__main__':
    counter = calc_grams(words(sys.argv[1]))
    with open(sys.argv[2] + '.count', 'w') as output:
        output.write('\n'.join(['%s\t%d' % (k, v) for k, v in counter.most_common()]))

    pmi = list(calc_pmi(counter))
    with open(sys.argv[2] + '.pmi', 'w') as output:
        output.write('\n'.join(['%s\t%d\t%f\t%f' % (p, c, pmi, npmi) for p, c, pmi, npmi in pmi]))
