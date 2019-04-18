#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import random
import numpy as np
import nltk
import itertools
import sys
import re
from tqdm import tqdm
from pyhanlp import *
from .flashtext import KeywordProcessor
from typing import List, Tuple

from .base import PhraseLabel


url = re.compile("((?:(http|https|Http|Https|rtsp|Rtsp)://(?:(?:[a-zA-Z0-9$\\-_.+!*\\'(),;\\?\\&\\=]|(?:\\%[a-fA-F0-9]{2})){1,64}(?:\\:(?:[a-zA-Z0-9\\$\\-\\_\\.\\+\\!\\*\\'\\(\\)\\,\\;\\?\\&\\=]|(?:\\%[a-fA-F0-9]{2})){1,25})?\\@)?)?(?:(((([a-zA-Z0-9][a-zA-Z0-9\\-]*)*[a-zA-Z0-9]\\.)+((aero|arpa|asia|a[cdefgilmnoqrstuwxz])|(biz|b[abdefghijmnorstvwyz])|(cat|com|coop|c[acdfghiklmnoruvxyz])|d[ejkmoz]|(edu|e[cegrstu])|f[ijkmor]|(gov|g[abdefghilmnpqrstuwy])|h[kmnrtu]|(info|int|i[delmnoqrst])|(jobs|j[emop])|k[eghimnprwyz]|l[abcikrstuvy]|(mil|mobi|museum|m[acdeghklmnopqrstuvwxyz])|(name|net|n[acefgilopruz])|(org|om)|(pro|p[aefghklmnrstwy])|qa|r[eosuw]|s[abcdeghijklmnortuvyz]|(tel|travel|t[cdfghjklmnoprtvwz])|u[agksyz]|v[aceginu]|w[fs]|(δοκιμή|испытание|рф|срб|טעסט|آزمایشی|إختبار|الاردن|الجزائر|السعودية|المغرب|امارات|بھارت|تونس|سورية|فلسطين|قطر|مصر|परीक्षा|भारत|ভারত|ਭਾਰਤ|ભારત|இந்தியா|இலங்கை|சிங்கப்பூர்|பரிட்சை|భారత్|ලංකා|ไทย|テスト|中国|中國|台湾|台灣|新加坡|测试|測試|香港|테스트|한국|xn\\-\\-0zwm56d|xn\\-\\-11b5bs3a9aj6g|xn\\-\\-3e0b707e|xn\\-\\-45brj9c|xn\\-\\-80akhbyknj4f|xn\\-\\-90a3ac|xn\\-\\-9t4b11yi5a|xn\\-\\-clchc0ea0b2g2a9gcd|xn\\-\\-deba0ad|xn\\-\\-fiqs8s|xn\\-\\-fiqz9s|xn\\-\\-fpcrj9c3d|xn\\-\\-fzc2c9e2c|xn\\-\\-g6w251d|xn\\-\\-gecrj9c|xn\\-\\-h2brj9c|xn\\-\\-hgbk6aj7f53bba|xn\\-\\-hlcj6aya9esc7a|xn\\-\\-j6w193g|xn\\-\\-jxalpdlp|xn\\-\\-kgbechtv|xn\\-\\-kprw13d|xn\\-\\-kpry57d|xn\\-\\-lgbbat1ad8j|xn\\-\\-mgbaam7a8h|xn\\-\\-mgbayh7gpa|xn\\-\\-mgbbh1a71e|xn\\-\\-mgbc0a9azcg|xn\\-\\-mgberp4a5d4ar|xn\\-\\-o3cw4h|xn\\-\\-ogbpf8fl|xn\\-\\-p1ai|xn\\-\\-pgbs0dh|xn\\-\\-s9brj9c|xn\\-\\-wgbh1c|xn\\-\\-wgbl6a|xn\\-\\-xkc2al3hye2a|xn\\-\\-xkc2dl3a5ee0h|xn\\-\\-yfro4i67o|xn\\-\\-ygbi2ammx|xn\\-\\-zckzah|xxx)|y[et]|z[amw]))|((25[0-5]|2[0-4][0-9]|[0-1][0-9]{2}|[1-9][0-9]|[1-9])\\.(25[0-5]|2[0-4][0-9]|[0-1][0-9]{2}|[1-9][0-9]|[1-9]|0)\\.(25[0-5]|2[0-4][0-9]|[0-1][0-9]{2}|[1-9][0-9]|[1-9]|0)\\.(25[0-5]|2[0-4][0-9]|[0-1][0-9]{2}|[1-9][0-9]|[0-9]))))(?:\\:\\d{1,5})?)(\\/(?:(?:[a-zA-Z0-9\\;\\/\\?\\:\\@\\&\\=\\#\\~\\-\\.\\+\\!\\*\\'\\(\\)\\,\\_])|(?:\\%[a-fA-F0-9]{2}))*)?")

# http://www.hankcs.com/nlp/part-of-speech-tagging.html#h2-8
class RulePhraseExtractor:
    unused = {
        'a', 'ad', 'ag', 'al', 'begin', 'end', 'bg', 'bl',
        'c', 'cc', 'd', 'dg', 'dl', 'e', 'end', 'f', 'h', 'i',
        'k', 'l', 'nl', 'o', 'p', 'pba', 'pbei', 's', 'u', 'ud', 'ude1',
        'ude2', 'ued3', 'udeng', 'udh', 'ug', 'uguo', 'uj', 'ul', 'ule',
        'ulian', 'uls', 'usuo', 'uv', 'uyy', 'uz', 'uzhe', 'uzhi', 'v', 'vd', 'vf',
        'vg', 'vi', 'vl', 'vn', 'vshi', 'vx', 'vyou',
        'w', 'wb', 'wd', 'wf', 'wd', 'wt', 'ww', 'wyy', 'wyz',
        'y', 'yg', 'z', 'zg'
    }

    specials = {
        'g', 'gb', 'gbc', 'gc', 'gg', 'gi', 'gm', 'gp', 'j', 'm', 'mg', 'Mg', 'mq',
        'nb', 'nba', 'nbc', 'nbp', 'nf', 'ng', 'nh', 'nhb', 'nhm',
        'ni', 'nic', 'nis', 'nit', 'nm', 'nmc', 'nn', 'nnd', 'nnt', 'nr', 'nr1', 'nr2', 'nrf', 'nrj',
        'ns', 'nsf', 'nt', 'ntc', 'ntcb', 'ntcf', 'ntch', 'nth', 'nto', 'nts', 'ntu', 'nx', 'nz',
        'q', 'qg', 'qt', 'qv', 'r', 'rg', 'Rg', 'rr', 'ry' 'rys', 'ryt', 'ryv', 'rz', 'rzs',
        't', 'tg', 'wb', 'xu', 'x', 'wh'
    }

    def __init__(self):
        # rule compound word
        self.entity_grammar = """
            NAME1: {<nr.*><nx><n.*>*<nr.*>|<nr.*>}
            NAME2: {<POSITION.*><NAME1>}
                   {<NAME1><nnt|nnd>}
            TIME: {<m><qt>(<m|q.*|t|b>*<qt|t|b>)*}
                  {<t>+}
            NUMEX: {<m><wb|mq>}
                   {<m.*>+<q>?}
            MEASUREX: {<m><a>?<q.*|n_人>}
            ADDRESS: {<url>}
            LOCATION: {<ns.*>+<n>*<f|d|s>}
                      {<ns.*>+<n.*>*<m><q>}
                      {<ns.*>+}

            ORGANIZATION1: {<nt.*>}
                           {<nz|LOCATION.*><nis|v_学会>}
            ORGANIZATION2: {<ORGANIZATION1><vn|m|n.*><ORGANIZATION1>}
                           {<ORGANIZATION1|LOCATION.*><n.*|m.*>*<ni.*|v_学会>}
                           {<LOCATION.*><ORGANIZATION.*>}
                           {<ORGANIZATION1><n><cc><v>*<n><ni.*|v_学会>}<nnt>

            POSITION1: {<nnt|nnd>}
            POSITION2: {<ORGANIZATION.*|LOCATION.*|nz><POSITION1>}

            FOOD: {<LOCATION.*|ORGANIZATION.*|nz|a>?<nf>}
            COMPOUND: {<ns.*|nr|nt.*><n>+<nis|nt.*>}
                      {<ns>*<nt*><nnt>}
                      {<nr|ns><nz>}
                      {<ns.*><n.*|vn|z>*<nnt>}
                      <w-、>{<f><n.*|vn>+}<w-、>
            TITLE: <w_《|w_【>{<.*>+}<w_》|w_】>
        """

        self.noise_grammar = """
            IGNORE: {<a|ad|ag|al|b|c|d.*|e|f|k|l|o|p.*|u.*|vshi|vyou|w.*|y|vi>}   
            IGNORE1: {<ad><v.*>}
                    {<p><n.*>+<v.*>}
                    {<p><n.*>+<f>}
                    {<v.*><.*>+<v.*>}
                    {<n.*>+<ude.*><n.*>+}
                    {<cc><n.*>+<a>}
                    {<d><v.*><ule><a>?<n.*>}
            NOISE: {<.*><IGNORE><.*>}
                   {<.*><IGNORE>}
                   {<IGNORE><.*>}

        """

        self.entity_parser = nltk.RegexpParser(self.entity_grammar)
        self.noise_parser = nltk.RegexpParser(self.noise_grammar)

    def extract(self, words: List[str], tags: List[str], offsets: List[Tuple[int, int]]) -> List[PhraseLabel]:
        labels = [PhraseLabel(begin, end, hanlp=(tag in self.specials))
                  for (begin, end), word, tag in zip(offsets, words, tags)
                  if tag in self.specials or tag in self.unused]

        for (begin, end), word, tag in zip(offsets, words, tags):
            if tag in self.specials and end - begin > 1:
                mid = random.randint(begin + 1, end - 1)
                labels.append(PhraseLabel(begin, mid, hanlp=False))
                labels.append(PhraseLabel(mid, end, hanlp=False))

        terms = [(
            (begin, end, word),
            '%s_%s' % (tag, word) if word in ['《', '》', '（', '）', '【', '】', '学会', '人'] else tag)
            for (begin, end), word, tag in zip(offsets, words, tags)]

        chunks = self.entity_parser.parse(terms)
        for node in chunks.subtrees(filter=lambda n: n.height() != chunks.height()):
            leaves = node.leaves()
            # print('good', ' '.join([leaf[0][2] for leaf in leaves]))
            if len(leaves) > 1:
                labels.append(PhraseLabel(leaves[0][0][0], leaves[-1][0][1], hanlp=True))

        chunks = self.noise_parser.parse(terms)
        for node in chunks.subtrees(filter=lambda n: n.height() != chunks.height()):
            leaves = node.leaves()
            # print('bad', ' '.join([leaf[0][2] for leaf in leaves]))
            if len(leaves) > 1:
                labels.append(PhraseLabel(leaves[0][0][0], leaves[-1][0][1], hanlp=False))

        return labels


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

'''
class PhraseNode:
    __slots__ = ['begin', 'end', 'children']
    def __init__(self, begin, end, children=None):
        self.begin = begin
        self.end = end
        self.children = children if children else []

class Tree:
    def __init__(self, length):
        self.root = PhraseLabel(0, length)

    def insert(self, begin, end):
        def _insert(node, begin, end):
            if node.begin == begin and node.end == end:
                return node
            if node.begin <= begin < node.end <= node.end:
                if len(node.children) == 0:
                    new_node = PhraseNode(begin, end)
                    node.children.append(new_node)
                    return new_node

                left = 0
                while left < len(node.children) and node.children[left].begin < begin:
                    left += 1
                right = len(node.children) - 1
                while right >= 0 and node.children[right].end > end:
                    right -= 1

                if left == right:

                if left < right:
                    new_node = PhraseNode(begin, end, children=node.children[begin:end])
                    node.children = node.children[:left] + [new_node] + node.children[right:]
                    return new_node
                else:
                    new_node = PhraseNode(begin, end)
                    if left == 0:
                        node.children = [new_node] + node.children
                    elif left >= len(node.children):
                        node.children.append(new_node)
'''


class Extractor:
    max_length = 15
    def __init__(self, keyword_extractor: KeywordProcessor):
        self.distant_extractor = keyword_extractor
        self.rule_extractor = RulePhraseExtractor()

    def extract(self, text: str, labels: List[PhraseLabel]):
        labels = self._baike_negative(text, labels)
        labeled_offsets = set((label.begin, label.end) for label in labels)

        terms = HanLP.segment(text)
        words = [term.word for term in terms]
        tags = [str(term.nature) for term in terms]
        offsets = []
        begin = 0
        for word in words:
            try:
                begin = text.index(word, begin)
            except Exception as e:
                print(e)

            offsets.append((begin, begin + len(word)))
            begin += len(word)

        rule_labels = [label for label in self.rule_extractor.extract(words, tags, offsets)
                       if (label.begin, label.end) not in labeled_offsets]
        labeled_offsets.update((label.begin, label.end) for label in rule_labels)

        distant_labels = [label for label in self.distant(text) if (label.begin, label.end) not in labeled_offsets]

        labeled_offsets.update((label.begin, label.end) for label in distant_labels)
        unlabels = []
        for begin in range(len(text)-2):
            end = random.randint(begin + 2, min(begin+self.max_length, len(text)))
            if (begin, end) not in labeled_offsets:
                unlabels.append(PhraseLabel(begin, end, unlabel=False))

        return np.array(labels + rule_labels + distant_labels + unlabels)

    '''
    def distant(self, text, words, offsets):

        seg_text = ' '.join(words)
        offset_mapping = {}
        seg_begin = 0
        for (begin, end), word in zip(offsets, words):
            try:
                seg_begin = seg_text.index(word, seg_begin)
                offset_mapping[seg_begin] = begin
                offset_mapping[seg_begin + len(word)] = end
            except Exception as e:
                print(e)

            seg_begin += len(word)

        distant_labels = []
        for keyword, seg_begin, seg_end in self.distant_extractor.extract_keywords(text, span_info=True):
            if seg_begin in offset_mapping and seg_end in offset_mapping:
                begin = offset_mapping[seg_begin]
                end = offset_mapping[seg_end]
                distant_labels.append(PhraseLabel(begin, end, distant=True))

        return distant_labels
    '''
    def distant(self, text):
        return [PhraseLabel(begin, end, distant=True)
                for keyword, begin, end in self.distant_extractor.extract_keywords(text, span_info=True)]

    def _baike_negative(self, text: str, labels: List[PhraseLabel]):
        labels = sorted(labels, key=lambda p: (p.begin, p.end))
        negatives = []

        negatives.extend(self._pairwise(labels))
        negatives.extend(self._intersect(labels, len(text)))

        return labels + negatives

    def _pairwise(self, phrases: List[PhraseLabel]):
        first_it, second_it = itertools.tee(phrases)
        next(second_it, None)
        for first, second in zip(first_it, second_it):
            for left in range(first.begin, first.end):
                for right in range(second.begin, second.end):
                    if left != first.begin or right != second.end:
                        yield PhraseLabel(left, right, baike=False)

    def _intersect(self, phrases: List[PhraseLabel], length):
        for phrase in phrases:
            for mid in range(phrase.begin + 1, phrase.end - 1):
                if mid - phrase.begin > 0:
                    yield PhraseLabel(phrase.begin, mid, unlabel=False)
                if phrase.end - mid > 0:
                    yield PhraseLabel(mid, phrase.end, unlabel=False)
                if phrase.begin > 0:
                    yield PhraseLabel(random.randint(max(0, phrase.begin - self.max_length), phrase.begin), mid, baike=False)
                if phrase.end < length:
                    yield PhraseLabel(mid, random.randint(phrase.end + 1, min(phrase.end + self.max_length + 1, length)), baike=False)

    @classmethod
    def build_from_dict(cls, path: str):
        processor = KeywordProcessor()
        processor.from_list(path)

        return cls(processor)


if __name__ == '__main__':
    extractor = Extractor.build_from_dict(sys.argv[1])
    for line in sys.stdin:
        print(line)
        result = extractor.extract(line, [])
        print([(phrase.begin, phrase.end, line[phrase.begin:phrase.end], phrase.labels ) for phrase in result])