#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import argparse
import gzip
import os
import json
import re
import sys
from collections import namedtuple, Counter
from typing import Dict, List, Tuple

from tqdm import tqdm

from task.util import utils
from .base import PhraseLabel, mixed_open, save_counter

# LINK_PREFIX = 'link::'
LINK_PREFIX = ''
ATTR_PREFIX = 'attr::'
prefix = 'https://baike.baidu.com/item'

key_blacklist = {'中文名', '中文名称'}
subtitle_blacklist = {'内容简介', '简介', '目录'}


class Voc:
    def __init__(self, counter: Dict):
        self.counter = counter
        self.itos = list(counter.keys())
        self.stoi = dict((s, i) for i, s in enumerate(self.itos))

    def get_index(self, name):
        return self.stoi.get(name)

    def get_str(self, id):
        assert id < len(self.itos)
        return self.itos[id]

    @staticmethod
    def create(path: str, mincount=1, blacklist=None):
        counter = {}
        with mixed_open(path, mode='rt') as file:
            for line in tqdm(file, desc='load voc'):
                line = line.strip()
                if len(line) > 0:
                    fields = line.rsplit('\t', maxsplit=1)
                    if len(fields) != 2:
                        print(line)
                        print(fields)
                    else:
                        text, count = fields[0], int(fields[1])
                        if len(text) > 0 and count < mincount:
                            break
                        if blacklist is None or text not in blacklist:
                            counter[text] = int(count)
        print('length = ', len(counter))
        return Voc(counter)


class Entity:
    def __init__(self, url: str, names: List[str],
                 keys: List[int], attrs: List[int], subtitles: List[int]):
        self.url = url
        self.names = names
        self.keys = keys
        self.attrs = attrs
        self.subtitles = subtitles

    def valid(self, name: str) -> bool:
        return name in self.names

    def get_labels(self) -> Tuple[List[int], List[int], List[int]]:
        return self.keys, self.attrs, self.subtitles

    def len_label(self):
        return len(self.keys) + len(self.attrs) + len(self.subtitles)

    def tojson(self):
        print(self.__dict__)
        return json.dumps(self.__dict__, ensure_ascii=False)

    @staticmethod
    def create(raws, key_voc: Voc, attr_voc: Voc, subtitle_voc: Voc):
        url = raws['url'][len(prefix):]
        names = set(raws['names'])
        keys, attrs = set(), set()
        for k, vs in raws['attrs'].items():
            if k[-2:] in {'又称', '又名', '别名', '别称', '另称', '简称', '缩写'}:
                names.update(vs)
            elif len(k) > 0:
                kid = key_voc.get_index(k)
                if kid:
                    keys.add(kid)
                for v in vs:
                    if len(v) > 0:
                        aid = attr_voc.get_index('%s:%s' % (k, v))
                        if aid:
                            attrs.add(aid)

        subtitles = set()
        for s in raws['subtitle']:
            sid = subtitle_voc.get_index(s)
            if len(s) > 0 and sid:
                subtitles.add(sid)

        return Entity(url, list(names), list(keys), list(attrs), list(subtitles))


def split(sentence: str):
    start = 0
    for m in re.finditer(r'\[\[\[([^[^|^\]]+)\|\|\|([^[^|^\]]+)\]\]\]', sentence):
        if start < m.start():
            yield sentence[start:m.start()]
        yield m.group(1), m.group(2)

        start = m.end()

    if start < len(sentence):
        yield sentence[start:]


class Labeler:
    def __init__(self, entities: Dict, key_voc: Voc, attr_voc: Voc, subtitle_voc: Voc):
        self.text_counter = Counter()
        self.key_counter = Counter()
        self.attr_counter = Counter()
        self.subtitle_counter = Counter()
        self.entity_counter = Counter()
        self.entities = entities
        self.key_voc = key_voc
        self.attr_voc = attr_voc
        self.subtitle_voc = subtitle_voc

    def label(self, sentence: str):
        words, labels = [], []
        for phrase in split(sentence.strip()):
            if isinstance(phrase, str):
                # words.extend(w for t, w in utils.replace_entity(phrase))
                words.extend(phrase)
            elif isinstance(phrase, tuple):
                assert len(phrase) == 2
                name, info = phrase
                begin = len(words)
                # words.extend(w for t, w in utils.replace_entity(name))
                words.extend(name)
                end = len(words)
                if info.startswith(LINK_PREFIX):
                    url = info[len(LINK_PREFIX) + len(prefix):]
                    entity = self.entities.get(url)
                    if entity and entity.valid(name):
                        keys, attrs, subtitles = entity.get_labels()
                        keys = [self.key_voc.get_str(k) for k in keys]
                        attrs = [self.attr_voc.get_str(a) for a in attrs]
                        subtitles = [self.subtitle_voc.get_str(s) for s in subtitles]
                        self.key_counter.update(keys)
                        self.attr_counter.update(attrs)
                        self.subtitle_counter.update(subtitles)
                        self.entity_counter.update([url])
                        labels.append(PhraseLabel(begin, end, keys=keys, attrs=attrs, subtitles=subtitles, entity=[url]))
                    elif len(name) > 1:
                        labels.append(PhraseLabel(begin, end))
                elif info.startswith(ATTR_PREFIX):
                    attr_name = info[len(ATTR_PREFIX):]
                    labels.append(PhraseLabel(begin, end, attr_name=[attr_name]))

        if len(labels) > 0:
            self.text_counter.update(words)
        return words, labels

    @staticmethod
    def create(entity_path, key_voc, attr_voc, subtitle_voc, mincount):
        key_voc = Voc.create(key_voc, mincount=mincount, blacklist=key_blacklist)
        attr_voc = Voc.create(attr_voc, mincount=mincount)
        subtitle_voc = Voc.create(subtitle_voc, mincount=mincount, blacklist=subtitle_blacklist)

        entities = dict()
        with mixed_open(entity_path, mode='rt') as file:
            for line in tqdm(file):
                try:
                    entity = Entity.create(json.loads(line), key_voc, attr_voc, subtitle_voc)
                    if entity.len_label() > 0:
                        entities[entity.url] = entity

                except Exception as e:
                    print(e.with_traceback())

        print('entity dict length = ', len(entities))
        return Labeler(entities, key_voc, attr_voc, subtitle_voc)


def listfile(path: str):
    if os.path.isdir(path):
        for name in os.listdir(path):
            yield from listfile(os.path.join(path, name))
    else:
        dir, prefix = os.path.split(path)
        if len(dir) == 0:
            dir = './'
        for name in os.listdir(dir):
            if name.startswith(prefix):
                yield os.path.join(dir, name)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='Preprocess baike corpus and save vocabulary')
    argparser.add_argument('--entity', type=str, help='entity dir, must include entity.gz, key.gz, attr.gz, subtitle.gz')
    argparser.add_argument('--corpus', type=str, help='corpus dir or prefix')
    argparser.add_argument('--output', type=str, help='output dir')
    argparser.add_argument('--mincount', type=int, default=10, help='vocabulary mincount')
    argparser.add_argument('--sample', type=int, help='sample for validation.')

    args = argparser.parse_args()

    entity_path = os.path.join(args.entity, 'entity.gz')
    key_path = os.path.join(args.entity, 'key.gz')
    attr_path = os.path.join(args.entity, 'attr.gz')
    subtitle_path = os.path.join(args.entity, 'subtitle.gz')

    labeler = Labeler.create(entity_path, key_path, attr_path, subtitle_path, mincount=args.mincount)

    os.makedirs(args.output, exist_ok=True)

    for corpus_file in listfile(args.corpus):
        print(corpus_file)
        with mixed_open(corpus_file, mode='rt') as data:
            output_path = os.path.join(args.output, os.path.split(corpus_file)[1])
            with mixed_open(output_path, 'wt') as output:
                for line in tqdm(data):
                    words, labels = labeler.label(line)
                    if 10 < len(words) < 300 and (len(labels) == 0 or labels[0].end - labels[0].begin < len(words)):
                        output.write('%s\t\t%s\n' % (
                            ''.join(words),
                            '\t\t'.join(l.to_json() for l in labels)))

    save_counter(os.path.join(args.output, 'text.voc.gz'), labeler.text_counter)

    save_counter(os.path.join(args.output, 'key.voc.gz'), labeler.key_counter)

    save_counter(os.path.join(args.output, 'attr.voc.gz'), labeler.attr_counter)

    save_counter(os.path.join(args.output, 'subtitle.voc.gz'), labeler.subtitle_counter)

    save_counter(os.path.join(args.output, 'entity.voc.gz'), labeler.entity_counter)
