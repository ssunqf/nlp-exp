#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import sys
import gzip
import math
import re
import argparse
import statistics
from typing import List, Union, Dict
from tqdm import tqdm
from collections import Counter

def count_all(path: str):
    total_count = 0
    with open(path) as input:
        for line in input:
            total_count += int(line)

    return total_count


def hash(phrase: Union[List[str], str]):
    def _detokeize(words: List[str]):
        for i in range(len(words)):
                if i > 0 and words[i-1][-1] < chr(255) and words[i][-1] < chr(255):
                    yield ' '
                yield from words[i]
    key = 1
    for c in _detokeize(phrase) if isinstance(phrase, List) else phrase:
        key = (key * 8978948897894561157) ^ (ord(c) * 17894857484156487943)

    return key


def make_hash(pathes: List[str], total_count: int):
    counter = Counter()
    hash2prob = {}
    for path in pathes:
        with gzip.open(path, mode='rt', compresslevel=6) as file:
            for line in tqdm(file, desc=('mask hash from %s' % path)):
                phrase, count = line.rsplit('\t', maxsplit=1)
                words = phrase.split()
                if words[0] != '<s>' and words[-1] != '</s>' and counter[len(words)] < 10000000:
                    hashid = hash(words)
                    hash2prob[hashid] = int(count) / total_count
                    counter[len(words)] += 1

                    if counter[10] >= 10000000:
                        break

    return hash2prob


def score(output: str, pathes: List[str], id2count: Dict[str, float]):
    files = [open('%s.%d' % (output, i), 'w') for i in range(10)]
    for path in pathes:
        with gzip.open(path, mode='rt', compresslevel=6) as file:
            for line in tqdm(file, desc=('score ngram from %s' % path)):

                def _score(text: List[str]):
                    prob_xy = id2count.get(hash(text), 0)
                    if prob_xy > 0:
                        max_x_y = -1e5
                        for mid in range(1, len(text)):
                            x = id2count.get(hash(text[0:mid]), 0)
                            y = id2count.get(hash(text[mid:]), 0)
                            max_x_y = max(max_x_y, x * y)

                        if max_x_y > 0:
                            mi = math.log(prob_xy/max_x_y)
                            nmi = mi / -math.log(prob_xy)
                            return mi, nmi

                    return None, None

                phrase, prob = line.rsplit('\t', maxsplit=1)
                words = phrase.split()
                if words[0] != '<s>' and words[-1] != '</s>':
                    mi, nmi = _score(words)
                    if mi:
                        files[len(words)-1].write('%s\t%s\t%f\t%f\n' % (' '.join(words), prob.strip(), mi, nmi))

    for f in files:
        f.close()


def make_entity(file: str):
    entities = {}
    with open(file) as input:
        for line in input:
            line = line.strip()
            entities[hash(line.strip())] = line

    return entities


def analyse(file, entities: Dict[int, str]):
    mi_list, nmi_list = [], []
    entity_mi_list, entity_nmi_list = [], []
    with open(file) as input:
        for line in input:
            phrase, count, mi, nmi = line.rsplit('\t', maxsplit=3)
            words = phrase.split()
            count, mi, nmi = int(count), float(mi), float(nmi)
            mi_list.append(mi)
            nmi_list.append(nmi)
            hashid = hash(words)
            if hashid in entities:
                entity_mi_list.append(mi)
                entity_nmi_list.append(nmi)

    print('mi: all mean=%f, std=%f\t\tentity mean=%f, std=%f',
          statistics.mean(mi_list), statistics.stdev(mi_list),
          statistics.mean(entity_mi_list), statistics.stdev(entity_mi_list))

    print('nmi: all mean=%f, std=%f\t\tentity mean=%f, std=%f',
          statistics.mean(nmi_list), statistics.stdev(nmi_list),
          statistics.mean(entity_nmi_list), statistics.stdev(entity_nmi_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='filter ngram')
    parser.add_argument('--stage', type=int, help='1: score, 2: analyse')
    parser.add_argument('--count', type=str, help='ngram count file')
    parser.add_argument('--ngram', type=str, help='ngram file')
    parser.add_argument('--entity', type=str, help='entity file')
    parser.add_argument('--input', type=str, help='input file')
    parser.add_argument('--output', type=str, help='output file')

    args = parser.parse_args()

    if args.stage == 1:
        assert args.count and args.ngram and args.input and args.output
        total_count = count_all(args.count)
        hash2prob = make_hash([args.ngram], total_count)
        score(args.output, [args.input], hash2prob)
    elif args.stage == 2:
        entities = make_entity(args.entity)
        analyse(args.input, entities)