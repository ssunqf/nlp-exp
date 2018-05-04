#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-


import argparse
import sys

import jieba
from spacy.lang.en import English

arg_parser = argparse.ArgumentParser()

#arg_parser.add_argument('--lang', type=str, help='language type')
#arg_parser.add_argument('--replace', type=str, default=False, help='replace named entity')

def tokenize_zh(sen):
    return jieba.cut(sen, cut_all=False)

english = English()
def tokenize_en(sen):
    return [str(w) for w in english(sen)]

if __name__ == '__main__':

    args = arg_parser.parse_args()

    for line in sys.stdin:
        items = line.strip().split('\t')
        if len(items) >= 2:
            en_sen, zh_sen, *_ = line.strip().split('\t')

            en_sen = ' '.join(tokenize_en(en_sen))
            zh_sen = ' '.join(tokenize_zh(zh_sen))

            print(en_sen + '\t' + zh_sen)