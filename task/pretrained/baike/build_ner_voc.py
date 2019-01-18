#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import sys
from tqdm import tqdm
from collections import Counter
from .gazetteer import Gazetteer
from .base import bio_to_bmeso, save_counter
import itertools


def get_line(file):
    chars, types = [], []
    for line in tqdm(file, desc='load data from %s ' % (file.name)):
        line = line.strip()
        if len(line) == 0:
            yield chars, types
            chars, types = [], []
        else:
            char, type = line.rsplit(maxsplit=1)
            chars.append(char)
            types.append(type)

    if len(chars) > 0:
        yield chars, types


if __name__ == '__main__':


    gazetteer = Gazetteer.from_files([
        ('digit', './gazetteers/chnDigit.dic'),
        ('person', 'gazetteers/chnName.dic'),
        ('place', 'gazetteers/chnPlace.dic'),
        ('quantifier', 'gazetteers/chnQuantifier.dic'),
        ('stopword', 'gazetteers/chnStopWord.dic'),
        ('phrase', 'gazetteers/phrase.dic')
    ])
    '''
    gazetteer = Gazetteer.from_jieba_pos('./gazetteers/jieba.pos.dic')
    '''
    char_counter = Counter()
    tag_counter = Counter()
    gazetteer_counter = Counter()

    for chars, tags in get_line(sys.stdin):
        chars, tags = bio_to_bmeso(chars, tags)
        char_counter.update(chars)
        tag_counter.update(tags)
        gazetteer_counter.update([tag for g_tags in gazetteer.find(chars) for tag in g_tags ])

    save_counter('./ner/data/char.voc.gz', char_counter)
    save_counter('./ner/data/tag.voc.gz', tag_counter)
    save_counter('./ner/data/gazetteer.voc.gz', gazetteer_counter)