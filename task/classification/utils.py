#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import re
from bs4 import Tag, BeautifulSoup
from typing import List, Optional, Dict
import zlib


def dbConnect():
    import asyncpg
    return asyncpg.connect(host='localhost',
                           user='sunqf', password='840422',
                           database='sunqf',
                           command_timeout=60)


def compress(html: str) -> bytes:
    return zlib.compress(html.encode())


def decompress(data: bytes) -> str:
    return zlib.decompress(data).decode()


def ChineseSplitter():
    ends = '。！？\n'
    pairs = {'(': ')', '{': '}', '[': ']', '<': '>', '《': '》', '（': '）', '【': '】', '“': '”'}
    left2id = {}
    right2id = {}
    sames = {'"', '\', '}
    same2id = {}
    for i, (k, v) in enumerate(pairs.items()):
        left2id[k] = i
        right2id[v] = i

    for i, s in enumerate(sames):
        same2id[s] = i

    def split_sentence(data: str):
        same_count = [0] * len(same2id)
        pair_count = [0] * len(left2id)

        begin = 0
        for pos, char in enumerate(data):
            if char in ends:
                if sum(same_count) == 0 and sum(pair_count) == 0:
                    if pos - begin > 1:
                        yield ''.join(data[begin:pos + 1])
                    begin = pos + 1
            elif char in left2id:
                pair_count[left2id[char]] += 1
            elif char in right2id:
                pair_count[right2id[char]] -= 1
            elif char in same2id:
                count = same_count[same2id[char]]
                same_count[same2id[char]] = (count + 1) % 2

        if begin < len(data) - 1:
            yield ''.join(data[begin:])

    return split_sentence


splitter = ChineseSplitter()


def clean_tag(tag: Tag):
    for img in tag.select('div.lemma-picture'):
        img.decompose()

    for img in tag.select('div.lemma-album'):
        img.decompose()

    for img in tag.select('a.lemma-album'):
        img.decompose()

    for img in tag.select('a.lemma-picture'):
        img.decompose()

    for sup in tag.select('sup'):
        sup.decompose()

    for sup_ref in tag.select('a.sup-anchor'):
        sup_ref.decompose()

    for table in tag.select('table'):
        table.decompose()

    return tag


def dictWalk(knowledge: dict, path: list):
    curr = knowledge
    for key in path:
        if isinstance(curr, dict) and key in curr:
            curr = curr[key]
        else:
            return None
    return curr


def html2text(data):
    if isinstance(data, str):
        return BeautifulSoup(data, 'html.parser').text.replace('\xa0', ' ').strip()
    elif isinstance(data, dict):
        return dict((html2text(k), html2text(v)) for k, v in data.items())
    elif isinstance(data, list):
        return [html2text(i) for i in data]
    else:
        return data


class Entity:
    def __init__(self, knowledge: Dict, label: Dict):
        self.knowledge = knowledge
        self.label = label

    def title(self):
        return dictWalk(self.knowledge, ['title'])

    def type(self) -> Optional[str]:
        title = dictWalk(self.knowledge, ['attrs', 'lemma_title'])
        if title:
            res = re.findall('(?<=（).*(?=）_百度百科)', title)
            if len(res) > 0:
                return res[0]
        return None

    def summary(self) -> List[str]:
        paras = html2text(dictWalk(self.knowledge, ['attrs', 'lemma_summary']))
        return [t for p in paras for t in splitter(p) if len(t) > 0] if paras and len(paras) > 0 else []

    def infobox(self, isText=True) -> Dict[str, List[str]]:
        box = dictWalk(self.knowledge, ['attrs', 'infobox'])
        if isText:
            box = html2text(box)
        return box if box else {}

    def open_tags(self) -> List[str]:
        tags = html2text(dictWalk(self.knowledge, ['attrs', 'open_tags']))
        return tags if tags else []

    def keywords(self) -> List[str]:
        t = dictWalk(self.knowledge, ['attrs', 'lemma_title'])
        text = dictWalk(self.knowledge, ['keywords'])
        raws = [word.replace('###', ' ').replace(t, '') if t else word.replace('###', ' ')
                for word in re.sub('(?<=[A-Za-z.])[ ]+(?=[A-Za-z.])', '###', text if text else '').split()]
        return [i for i in raws if len(i) > 0]

    def labels(self) -> List[str]:

        def _filter():
            for k, v in self.label.items():
                if 'flag' in v and v['flag'] == 'manual' and 'mask' in v and v['mask'] == 1:
                    yield k

        return list(_filter())

