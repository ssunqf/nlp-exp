#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import argparse
import itertools
import os
import random
import re
import time
from collections import defaultdict, deque
import numpy as np

from nltk import Tree
from opencc import OpenCC
from tabulate import tabulate
from tensorboardX import SummaryWriter
from torch import optim
from torch.nn import functional as F
from torchtext import vocab
from torchtext.data.iterator import BucketIterator
from tqdm import tqdm

from task.pretrained.baike.flashtext import KeywordProcessor
from .base import INIT_TOKEN, EOS_TOKEN, PAD_TOKEN, bio_to_bmeso, PhraseLabel
from .classifier import ContextClassifier
from .data import LabelField
from .crf import LinearCRF, MaskedCRF
from .encoder import ElmoEncoder, StackRNN
from .tags import *
from ..transformer.field import PartialField, BracketField

t2s_cc = OpenCC('t2s')


double2single = {
    '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
    '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
    'Ａ': 'A', 'Ｂ': 'B', 'Ｃ': 'C', 'Ｄ': 'D', 'Ｅ': 'E',
    'Ｆ': 'F', 'Ｇ': 'G', 'Ｈ': 'H', 'Ｉ': 'I', 'Ｊ': 'J',
    'Ｋ': 'K', 'Ｌ': 'L', 'Ｍ': 'M', 'Ｎ': 'N', 'Ｏ': 'O',
    'Ｐ': 'P', 'Ｑ': 'Q', 'Ｒ': 'R', 'Ｓ': 'S', 'Ｔ': 'T',
    'Ｕ': 'U', 'Ｖ': 'V', 'Ｗ': 'W', 'Ｘ': 'X', 'Ｙ': 'Y',
    'Ｚ': 'Z', 'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd',
    'ｅ': 'e', 'ｆ': 'f', 'ｇ': 'g', 'ｈ': 'h', 'ｉ': 'i',
    'ｊ': 'j', 'ｋ': 'k', 'ｌ': 'l', 'ｍ': 'm', 'ｎ': 'n',
    'ｏ': 'o', 'ｐ': 'p', 'ｑ': 'q', 'ｒ': 'r', 'ｓ': 's',
    'ｔ': 't', 'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｘ': 'x',
    'ｙ': 'y', 'ｚ': 'z',

    '➀': '1', '➁': '2', '➂': '3', '➃': '4', '➄': '5',
    '➅': '6', '➆': '7', '➇': '8', '➈': '9', '①': '1',
    '②': '2', '③': '3', '④': '4', '⑤': '5', '⑥': '6',
    '⑦': '7', '⑧': '8', '⑨': '9', '❶': '1', '❷': '2',
    '❸': '3', '❹': '4', '❺': '5', '❻': '6', '❼': '7',
    '❽': '8', '❾': '9', '➊': '1', '➋': '2', '➌': '3',
    '➍': '4', '➎': '5', '➏': '6', '➐': '7', '➑': '8',
    '➒': '9', 'Ⓐ': 'A', 'Ⓑ': 'B', 'Ⓒ': 'C', 'Ⓓ': 'D',
    'Ⓔ': 'E', 'Ⓕ': 'F', 'Ⓖ': 'G', 'Ⓗ': 'H', 'Ⓘ': 'I',
    'Ⓙ': 'J', 'Ⓚ': 'K', 'Ⓛ': 'L', 'Ⓜ': 'M', 'Ⓝ': 'N',
    'Ⓞ': 'O', 'Ⓟ': 'P', 'Ⓠ': 'Q', 'Ⓡ': 'R', 'Ⓢ': 'S',
    'Ⓣ': 'T', 'Ⓤ': 'U', 'Ⓥ': 'V', 'Ⓦ': 'W', 'Ⓧ': 'X',
    'Ⓨ': 'Y', 'Ⓩ': 'Z', 'ⓐ': 'a', 'ⓑ': 'b', 'ⓒ': 'c',
    'ⓓ': 'd', 'ⓔ': 'e', 'ⓕ': 'f', 'ⓖ': 'g', 'ⓗ': 'h',
    'ⓘ': 'i', 'ⓙ': 'j', 'ⓚ': 'k', 'ⓛ': 'l', 'ⓜ': 'm',
    'ⓝ': 'n', 'ⓞ': 'o', 'ⓟ': 'p', 'ⓠ': 'q', 'ⓡ': 'r',
    'ⓢ': 's', 'ⓣ': 't', 'ⓤ': 'u', 'ⓥ': 'v', 'ⓦ': 'w',
    'ⓧ': 'x', 'ⓨ': 'y', 'ⓩ': 'z',
    '㊀': '一', '㊁': '二', '㊂': '三', '㊃': '四',
    '㊄': '五', '㊅': '六', '㊆': '七', '㊇': '八',
    '㊈': '九',

    '⑴': '1', '⑵': '2', '⑶': '3', '⑷': '4', '⑸': '5',
    '⑹': '6', '⑺': '7', '⑻': '8', '⑼': '9',
    '🄐': 'A', '🄑': 'B', '🄒': 'C', '🄓': 'D', '🄔': 'E',
    '🄕': 'F', '🄖': 'G', '🄗': 'H', '🄘': 'I', '🄙': 'J',
    '🄚': 'K', '🄛': 'L', '🄜': 'M', '🄝': 'N', '🄞': 'O',
    '🄟': 'P', '🄠': 'Q', '🄡': 'R', '🄢': 'S', '🄣': 'T',
    '🄤': 'U', '🄥': 'V', '🄦': 'W', '🄧': 'X', '🄨': 'Y',
    '🄩': 'Z', '⒜': 'a', '⒝': 'b', '⒞': 'c', '⒟': 'd',
    '⒠': 'e', '⒡': 'f', '⒢': 'g', '⒣': 'h', '⒤': 'i',
    '⒥': 'j', '⒦': 'k', '⒧': 'l', '⒨': 'm', '⒩': 'n',
    '⒪': 'o', '⒫': 'p', '⒬': 'q', '⒭': 'r', '⒮': 's',
    '⒯': 't', '⒰': 'u', '⒱': 'v', '⒲': 'w', '⒳': 'x',
    '⒴': 'y', '⒵': 'z',
    '㈠': '一', '㈡': '二', '㈢': '三', '㈣': '四',
    '㈤': '五', '㈥': '六', '㈦': '七', '㈧': '八',
    '㈨': '九',

    '⒈': '1',
    '⒉': '2',
    '⒊': '3',
    '⒋': '4',
    '⒌': '5',
    '⒍': '6',
    '⒎': '7',
    '⒏': '8',
    '⒐': '9',

    '．': '.',
    '•': '·',
    '／': '/',
    '％': '%',
    '：': ':',
    '－': '-',
    '＇': '\'',
    '｛': '{',
    '｝': '}',
    '（': '(',
    '）': ')',
    '丶': '、',
    '！': '!',
    '「': '“',
    '」' : '”',
    '『': '‘',
    '』': '’',
    '【': '[',
    '】': ']',
    '〈': '《',
    '〉': '》'
}

def is_eng_alpha(ch):
    return 'a' <= ch <= 'z' or 'ａ' <= ch <= 'ｚ' \
           'A' <= ch <= 'Z' or 'Ａ' <= ch <= 'Ｚ' \
           '0' <= ch <='9' or '０' <= ch <= '９'

def ChineseSplitter():
    ends = '。！？\n'
    pairs = {'(': ')', '{': '}', '[': ']', '<': '>', '《': '》', '『': '』', '（': '）', '【': '】', '“': '”', '‘': '’'}
    left2id = {}
    right2id = {}
    sames = {'"', '\''}
    same2id = {}
    for i, (k, v) in enumerate(pairs.items()):
        left2id[k] = i
        right2id[v] = i

    for i, s in enumerate(sames):
        same2id[s] = i

    def split_sentence(data: List[str]):
        same_count = [0] * len(same2id)
        pair_count = [0] * len(left2id)

        begin, end = 0, 0
        while end < len(data):
            if data[end] in ends:
                while end < len(data) and data[end] in ends:
                    end += 1
                if sum(same_count) == 0 and sum(pair_count) == 0:
                    if end - begin > 1:
                        yield ''.join(data[begin:end])
                        begin = end

            elif data[end] in left2id:
                pair_count[left2id[data[end]]] += 1
                end += 1
            elif data[end] in right2id:
                pair_count[right2id[data[end]]] -= 1
                if end > 0 and data[end-1] in ends and sum(same_count) == 0 and sum(pair_count) == 0:
                    if end - begin > 1:
                        yield ''.join(data[begin:end+1])
                        begin = end + 1

                end += 1
            elif data[end] in same2id:
                same_count[same2id[data[end]]] = (same_count[same2id[data[end]]] + 1) % 2
                end += 1
            else:
                end += 1

        if begin < len(data) - 1:
            yield ''.join(data[begin:])

    return split_sentence


splitter = ChineseSplitter()

def double_to_single(text):
    return [double2single.get(s, s) for s in text]


def preprocess(text, *args):
    text = t2s_cc.convert(''.join(text) if isinstance(text, list) else text)

    offset = 0
    for sentence in splitter(text):
        for arg in args:
            assert len(sentence) == len(arg[offset:offset+len(sentence)])
        yield tuple([double_to_single(sentence)] + [arg[offset:offset+len(sentence)] for arg in args])
        offset += len(sentence)


class NamedEntityData(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path: str, fields: List, **kwargs):

        examples = []
        for chars, types in self.get_line(path):
            chars, types = bio_to_bmeso(chars, types)
            examples.append(data.Example.fromlist([chars, types], fields))
        super(NamedEntityData, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def get_line(path):
        with open(path) as file:
            chars, types = [], []
            for line in tqdm(file, desc='load data from %s ' % (path)):
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


class BiCharField(data.Field):

    def preprocess(self, xs):
        """Preprocess a single example.
        Arguments:
            xs (list or str): The input to preprocess.

        Returns:
            list: The preprocessed list.
        """

        def pairwise(words: List[str]):
            import itertools
            a, b = itertools.tee(words)
            next(b, None)
            return zip(a, b)
        return ['<s>#<s>'] + ['%s#%s' % (a, b) for a, b in pairwise(xs)]


class KeywordField(data.Field):
    """
        前向后向最大匹配
    """
    def __init__(self, pretrained_vocab, **kwargs):
        self.vocab = pretrained_vocab
        self.keywords = KeywordProcessor(tokenize_func=lambda text: list(text))

        for s in self.vocab.itos:
            self.keywords.add_keyword(s)

        super(KeywordField, self).__init__(**kwargs)

    def preprocess(self, xs):
        return list(sorted(self.keywords.extract_keywords(xs, xs, span_info=True),
                           key=lambda x: x[2]-x[1])), len(xs)

    def process(self, batch, device=None):
        """ Process a list of examples to create a torch.Tensor.

        Pad, numericalize, and postprocess a batch and create a tensor.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            torch.autograd.Variable: Processed object given the input
            and custom postprocessing Pipeline.
        """
        forwards, backwards = [], []
        for sentence, length in batch:
            forwards.append([self.unk_token] * length)
            backwards.append([self.unk_token] * length)
            for word, begin, end in sentence:
                if end <= length:
                    forwards[-1][end-1] = word
                    backwards[-1][begin] = word

        forwards = self.numericalize(self.pad(forwards), device=device)
        backwards = self.numericalize(self.pad(backwards), device=device)
        return forwards, backwards

    @classmethod
    def from_pretrained(cls, vectors, **kwargs):
        pretrained_vocab = vocab.Vocab(Counter())

        def _extend(vocab, words):
            for w in words:
                if w not in vocab.stoi:
                    vocab.itos.append(w)
                    vocab.stoi[w] = len(vocab.itos) - 1

        _extend(pretrained_vocab, vectors.itos[0:1000000])
        pretrained_vocab.set_vectors(vectors.stoi, vectors.vectors, dim=vectors.dim)

        return cls(pretrained_vocab, **kwargs)


class POSData(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path: str, fields: List, **kwargs):
        examples = [data.Example.fromlist((chars, types), fields)
                    for chars, types in self.get_line(path)]
        super(POSData, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def get_line(path):
        with open(path) as file:
            for line in tqdm(file, desc='load data from %s ' % (path)):
                chars, types = [], []
                for tokens in line.split():
                    try:
                        _word, _type = tokens.rsplit('/', maxsplit=1)
                    except Exception as e:
                        print(line)
                        print(e)
                    chars.extend(_word)
                    if len(_word) == 1:
                        types.extend(['S_%s' % _type])
                    elif len(_word) >= 2:
                        types.extend(['B_%s' % _type] + ['M_%s' % _type] * (len(_word) - 2) + ['E_%s' % _type])
                yield chars, types


class People2014(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, fields: List, **kwargs):
        raws = []
        char_counter = Counter()
        tag_counter = Counter()
        for line in tqdm(self.read_line(path), desc='load data from %s' % path):
            try:
                chars, tags = [], []
                for word, tag in self.fix_tokenize(line):
                    chars.extend(list(word))
                    tags.extend(to_bmes(len(word), tag))

                char_counter.update(chars)
                tag_counter.update(tags)
                raws.append((chars, tags))
            except Exception as e:
                print(e)
                print(line)
                print(self.fix_tokenize(line))

        print(tag_counter.most_common()[-100:])
        examples = [data.Example.fromlist((chars, types), fields)
                    for chars, types in raws]
        super(People2014, self).__init__(examples, fields, **kwargs)

    def fix_tokenize(self, raw_line):
        tokens = []
        line = re.sub(r'\[([^\]]+)\]/[a-z]+[0-9]?', r'\1', raw_line)
        for token in line.split():
            items = token.rsplit('/', maxsplit=1)
            if len(items) == 1:
                tokens.extend([(c, 'w') for c in items])
            else:
                word, tag = items
                for i in range(len(word)):
                    if word[i] in {'，', '：', '、', '。', '“', '”', '《', '》', '（', '）', '(', ')', '……', '‘', '’'}:
                        tokens.append((word[i], 'w'))
                    else:
                        if tag in ['ude1', 'ude2', 'ude3', 'w', 'wb'] and len(word[i:]) > 1:
                            print(raw_line)
                            print(line)
                            print(items)
                        tokens.append((word[i:], tag))
                        break
        return tokens

    def read_line(self, path: str):
        if os.path.isdir(path):
            for name in os.listdir(path):
                child = os.path.join(path, name)
                yield from self.read_line(child)
        elif path.endswith('.txt'):
            with open(path) as file:
                for line in file:
                    line = line.strip()
                    if len(line) > 0:
                        yield line
        else:
            print('%s is not loaded.' % path)

    @classmethod
    def splits(cls,
               text_field,
               path=None, root='.data',
               train=None, validation=None, test=None,
               **kwargs):

        fields = [
            ('text', text_field),
            ('people2014_pos', PartialField(init_token=INIT_TOKEN, eos_token=EOS_TOKEN))
        ]

        train_data = None if train is None else cls(
            os.path.join(path, train), fields, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), fields, **kwargs)

        if val_data is None and test_data is None:
            train_data, val_data, test_data = train_data.split(split_ratio=[0.85, 0.05, 0.1])

        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


class People98(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, fields: List, **kwargs):
        examples = [data.Example.fromlist((chars, types), fields)
                    for chars, types in self.read_sentence(path)]
        super(People98, self).__init__(examples, fields, **kwargs)

    def read_sentence(self, path: str):
        with open(path) as input:
            for line in tqdm(input, desc='load data from %s' % path):
                line = line.strip()
                if len(line) == 0:
                    continue
                try:
                    chars, tags = [], []
                    left = 0
                    for token in line.split()[1:]:
                        word, tag = token.rsplit('/', maxsplit=1)
                        if len(word) > 1 and word.startswith('['):
                            while len(word) > 1 and word.startswith('['):
                                left += 1
                                word = word[1:]
                        elif left > 0 and len(word) > 3 and word.endswith(']'):
                            while left > 0 and len(word) > 3 and word.endswith(']'):
                                left -= 1
                                word = word[0:-1]
                            word, tag = word.rsplit('/', maxsplit=1)

                        if len(chars) > 0 and is_eng_alpha(chars[-1]) and is_eng_alpha(word[0]):
                            chars.append(' ')
                            tags.append('S_space')
                        chars.extend(word)
                        tags.extend(to_bmes(len(word), tag))

                    # tags = [tag[0:2] for tag in tags]
                    yield from preprocess(chars, tags)
                except Exception as e:
                    print(line)

    @classmethod
    def splits(cls,
               text_fields,
               bracket_field,
               path=None, root='.data',
               train=None, validation=None, test=None,
               **kwargs):

        fields = [
            text_fields,
            # bracket_field,
            ('people98_pos', PartialField(init_token=INIT_TOKEN, eos_token=EOS_TOKEN, is_target=True))
        ]

        train_data = None if train is None else cls(
            os.path.join(path, train), fields, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), fields, **kwargs)

        if val_data is None:
            train_data, val_data = train_data.split(split_ratio=[0.9, 0.1])

        if test_data is None:
            train_data, test_data = train_data.split(split_ratio=[0.85, 0.15])

        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


class MSRA(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, fields: List, **kwargs):
        examples = [data.Example.fromlist(sentence, fields)
                    for sentence in self.read_xml(path)]
        super(MSRA, self).__init__(examples, fields, **kwargs)

    def read_xml(self, path: str):

        def _to_bmes(length):
            if length >= 3:
                return ['*B*'] + ['*M*'] * (length - 2) + ['*E*']
            else:
                return ['*'] * length

        with open(path) as input:
            from bs4 import BeautifulSoup, element
            xml_file = BeautifulSoup(input, features='lxml')
            for sentence in tqdm(xml_file.find_all('sentence')):
                chars, bracket_tags, ne_tags = [], [], []
                # phrase = []
                for word in sentence.find_all('w'):
                    assert word.name == 'w'
                    _chars, _bracket_tags, _ne_tags = [], [], []
                    num_child = 0
                    for child in word.children:
                        num_child += 1
                        if isinstance(child, element.Tag):
                            if len(_chars) > 0 and is_eng_alpha(_chars[-1]) and is_eng_alpha(child.text):
                                _chars.append(' ')
                                _bracket_tags.append('*S*')
                                _ne_tags.append('S_O')
                            _chars.extend(child.text)
                            _bracket_tags.extend(_to_bmes(len(child.text)))
                            _ne_tags.extend(to_bmes(len(child.text), child['type']))
                            # phrase.append(len(chars) + len(_chars) - len(child.text), len(chars) + len(_chars), child['type'])
                        else:
                            if len(_chars) > 0 and is_eng_alpha(_chars[-1]) and is_eng_alpha(child):
                                _chars.append(' ')
                                _bracket_tags.append('*S*')
                                _ne_tags.append('S_O')
                            _chars.extend(child)
                            _bracket_tags.extend(_to_bmes(len(child)))
                            _ne_tags.extend(to_bmes(len(child), 'O'))

                    if num_child > 1:
                        _bracket_tags = [t2 + t1 if t1 != '*' else t2 for t1, t2 in zip(_bracket_tags, _to_bmes(len(_chars)))]

                    chars.extend(_chars)
                    bracket_tags.extend(_bracket_tags)
                    ne_tags.extend(_ne_tags)

                # tags = [tag[0:2] for tag in tags]
                yield double_to_single(chars), bracket_tags, ne_tags # , phrase
                # yield from preprocess(chars, ne_tags)

    @classmethod
    def splits(cls,
               text_fields,
               bracket_field: Tuple[str, data.Field],
               path=None, root='.data',
               train=None, validation=None, test=None,
               **kwargs):

        fields = [
            text_fields,
            bracket_field,
            ('msra_ne', PartialField(init_token=INIT_TOKEN, eos_token=EOS_TOKEN, is_target=True))
        ]

        train_data = None if train is None else cls(
            os.path.join(path, train), fields, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), fields, **kwargs)


        return (train_data, val_data, test_data)


class Bracket(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, fields: List, **kwargs):
        examples = [data.Example.fromlist((double_to_single(chars), bracket_tags, pos_tags), fields)
                    for chars, bracket_tags, pos_tags in self.read_sentence(path)]
        super(Bracket, self).__init__(examples, fields, **kwargs)

    def read_sentence(self, path: str):

        def _to_bmes(length):
            if length == 1:
                return ['S']
            else:
                return ['B'] + ['M'] * (length - 2) + ['E']

        def tree_2_bmes(node: Tree):
            chars, bracket_tags, pos_tags = [], [], []

            num_child = 0
            for child in node:
                if isinstance(child, Tree):
                    if '-NONE-' in child.label():
                        continue

                    num_child += 1

                    if child.height() == 2:
                        if len(chars) > 0 and is_eng_alpha(chars[-1]) and is_eng_alpha(child[0][0]):
                            chars.append(' ')
                            pos_tags.append('S_SPACE')
                            bracket_tags.append('S')
                        chars.extend(child[0])
                        pos_tags.extend(to_bmes(len(child[0]), child.label().split('-')[0]))
                        bracket_tags.extend(_to_bmes(len(child[0])))
                    else:
                        _chars, _bracket_tags, _pos_tags = tree_2_bmes(child)
                        if len(chars) > 0 and is_eng_alpha(chars[-1]) and is_eng_alpha(_chars[0]):
                            chars.append(' ')
                            pos_tags.append('S_SPACE')
                            bracket_tags.append('S')
                        chars.extend(_chars)
                        pos_tags.extend(_pos_tags)
                        bracket_tags.extend(_bracket_tags)

            if node.label() != 'S' and num_child > 1 and node.height() <= 6 and len(chars) > 1:
                bracket_tags = [prefix + tag for prefix, tag in zip(_to_bmes(len(chars)), bracket_tags)]

            return chars, bracket_tags, pos_tags

        with open(path) as input:
            for line in tqdm(input, desc='load data from %s' % path):
                try:
                    root = Tree.fromstring(line)
                    if not isinstance(root, Tree) or root.label() != 'S':
                        continue

                    chars, bracket_tags, pos_tags = tree_2_bmes(root)
                    chars = list(t2s_cc.convert(''.join(chars)))

                    assert len(chars) == len(bracket_tags) == len(pos_tags)
                    yield chars, bracket_tags, pos_tags

                except Exception as e:
                    print(line)
                    print(root)
                    print(e)

    @classmethod
    def splits(cls,
               text_fields,
               bracket_field,
               path=None, root='.data',
               train=None, validation=None, test=None,
               **kwargs):

        fields = [
            text_fields,
            bracket_field,
            ('ctb_pos', PartialField(init_token=INIT_TOKEN, eos_token=EOS_TOKEN, is_target=True)),
        ]

        train_data = None if train is None else cls(
            os.path.join(path, train), fields, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), fields, **kwargs)

        if val_data is None:
            train_data, val_data = train_data.split(split_ratio=[0.9, 0.1])

        if test_data is None:
            train_data, test_data = train_data.split(split_ratio=[0.85, 0.15])

        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


class Baike(data.Dataset):
    filter_pattern = re.compile('(merge_red.sh|merge_map.sh)')

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path: str, fields: List, **kwargs):
        examples = [data.Example.fromlist(example, fields)
                    for example in self.read_sentence(path)]
        super(Baike, self).__init__(examples, fields, **kwargs)

    def read_sentence(self, path):
        with smart_open(path, mode='rt') as file:

            for id, line in enumerate(tqdm(file, desc=('load dataset from %s' % path))):
                if id > 500000:
                    break

                text, *labels = line.split('\t\t')
                text = text.replace(u'\u3000', ' ')

                if self.filter_pattern.search(text):
                    continue

                # words = words.split(' ')
                chars = list(text)
                try:
                    labels = [PhraseLabel.from_json(label) for label in labels if len(label) > 0]

                    if 10 < len(chars) < 300 and len(labels) > 0:
                        # chars = np.array(chars, dtype=np.str)
                        # labels = np.array([l.to_np() for l in labels], dtype=PhraseLabel.get_type())
                        bracket_tags = ['*'] * len(chars)
                        for label in labels:
                            if label.end - label.begin > 4:
                                bracket_tags[label.begin:label.end] = ['*B*'] + ['*M*'] * (label.end - label.begin - 2) + ['*E*']
                        yield double_to_single(chars), bracket_tags, labels

                except Exception as e:
                    print(e.with_traceback())
                    pass

    @staticmethod
    def load_voc(config):
        KEY_LABEL = LabelField('keys', is_target=True)
        ATTR_LABEL = LabelField('attrs', is_target=True)
        SUB_LABEL = LabelField('subtitles', is_target=True)
        # ENTITY_LABEL = LabelField('entity')

        KEY_LABEL.build_vocab(os.path.join(config.root, 'key.voc.gz'),
                              max_size=config.key_max_size,
                              min_freq=config.key_min_freq)
        ATTR_LABEL.build_vocab(os.path.join(config.root, 'attr.voc.gz'),
                               max_size=config.attr_max_size,
                               min_freq=config.attr_min_freq)
        SUB_LABEL.build_vocab(os.path.join(config.root, 'subtitle.voc.gz'),
                              max_size=config.subtitle_max_size,
                              min_freq=config.subtitle_min_freq)
        # ENTITY_LABEL.build_vocab(os.path.join(config.root, 'entity.voc.gz'),
        #                         max_size=config.entity_max_size,
        #                         min_freq=config.entity_min_freq)

        print('key vocab size = %d' % len(KEY_LABEL.vocab))
        print('attr vocab size = %d' % len(ATTR_LABEL.vocab))
        print('subtitle vocab size = %d' % len(SUB_LABEL.vocab))
        # print('entity vocab size = %d' % len(ENTITY_LABEL.vocab))

        return KEY_LABEL, ATTR_LABEL, SUB_LABEL

    @classmethod
    def splits(cls,
               text_fields,
               bracket_field,
               config,
               path=None, root='.data',
               train=None, validation=None, test=None,
               **kwargs):

        key, attr, sub = cls.load_voc(config)

        fields = [
            text_fields,
            bracket_field,
            (('infobox_key', 'infobox_value', 'subtitle'), (key, attr, sub))
        ]

        train_data = None if train is None else cls(
            os.path.join(path, train), fields, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), fields, **kwargs)

        if val_data is None:
            train_data, val_data = train_data.split(split_ratio=[0.9, 0.1])

        if test_data is None:
            train_data, test_data = train_data.split(split_ratio=[0.85, 0.15])

        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


class ChunkDataset(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, fields: List, **kwargs):
        examples = [data.Example.fromlist((double_to_single(chars), pos_tags, chunk_tags), fields)
                    for chars, pos_tags, chunk_tags in self.read_sentence(path)]
        super(ChunkDataset, self).__init__(examples, fields, **kwargs)

    def read_sentence(self, path: str):
        with open(path) as input:
            for line in tqdm(input, desc='load data from %s' % path):
                try:
                    root = Tree.fromstring(line)
                    chars, pos_tags, chunk_tags = [], [], []
                    for child in root:
                        for word in child:
                            pos_tag = word.label().split('-')[0]
                            chars.extend(word[0])
                            pos_tags.extend(to_bmes(len(word[0]), pos_tag))
                        chunk_tags.extend(to_bmes(sum(len(word[0]) for word in child), child.label()))
                    # pos_tags = [tag[0:2] for tag in pos_tags]
                    yield chars, pos_tags, chunk_tags

                except Exception as e:
                    print(line)
                    print(root)
                    print(e)

    @classmethod
    def splits(cls,
               text_fields, path=None, root='.data',
               train=None, validation=None, test=None,
               **kwargs):

        fields = [
            text_fields,
            ('ctb_pos', PartialField(init_token=INIT_TOKEN, eos_token=EOS_TOKEN, is_target=True)),
            ('ctb_chunk', PartialField(init_token=INIT_TOKEN, eos_token=EOS_TOKEN, is_target=True))
        ]
        return super(ChunkDataset, cls).splits(
            path=path, root=root,
            train=train, validation=validation, test=test,
            fields=fields, **kwargs)

    '''
    def read_line(self, path: str):
        with open(path) as file:
            words, poses, tags = [], [], []
            for id, line in tqdm(enumerate(file)):
                line = strQ2B(line.strip())
                if len(line) == 0:
                    if len(words) > 0:
                        if set(tags) == {'O'}:
                            print(list(zip(words, tags)))
                        else:
                            try:
                                yield self.to_flat(words, poses, tags)
                            except Exception as e:
                                print(e)
                                print(list(zip(words, tags)))
                        words, poses, tags = [], [], []
                    continue

                tokens = line.split()
                words.append(tokens[5])
                poses.append(tokens[4].split('-')[0])
                tags.append(tokens[3] if tokens[4] != 'URL' else 'URL')

            if len(words) > 0:
                try:
                    yield self.to_flat(words, poses, tags)
                except Exception as e:
                    print(e)
                    print(zip(words, poses, tags))

    def isalnum(self, char):
        return 'a' < char < 'z' or 'A' < char < 'Z' or '0' < char < '9'

    def add_space(self, words, poses, tags):
        fixed_words, fixed_poses, fixed_tags = [], [], []
        for word, pos, tag in zip(words, poses, tags):
            if len(fixed_words) > 0 and self.isalnum(fixed_words[-1][-1]) and self.isalnum(word[0]):
                fixed_words.append(' ')
                fixed_poses.append('SPACE')
                if tag.startswith('I-'):
                    fixed_tags.append(tag)
                else:
                    fixed_tags.append('O')
            fixed_words.append(word)
            fixed_poses.append(pos)
            fixed_tags.append(tag)
        return fixed_words, fixed_poses, fixed_tags

    def to_flat(self, words, poses, tags):
        chunks, chunk_poses, chunk_types = [], [], []
        for word, pos, tag in zip(*self.add_space(words, poses, tags)):
            if tag.startswith('B-'):
                chunks.append([word])
                chunk_poses.append([pos])
                chunk_types.append(tag[2:])
            elif tag.startswith('I-'):
                chunks[-1].append(word)
                chunk_poses[-1].append(pos)
                assert chunk_types[-1] == tag[2:]
            else:
                chunks.append([word])
                chunk_poses.append([pos])
                chunk_types.append(tag)

        chars, char_poses, char_tags = [], [], []
        for chunk, poses, type in zip(chunks, chunk_poses, chunk_types):
            chunk_chars = []
            for word, pos in zip(chunk, poses):
                chunk_chars.extend(word)
                char_poses.extend(to_bmes(len(word), pos))

            chars.extend(chunk_chars)
            char_tags.extend(to_bmes(len(chunk_chars), type))

        terms = [(term.word, str(term.nature)) for term in HanLP.segment(''.join(chars))]

        return chars, char_poses, char_tags
    '''

class LMClassifier(nn.Module):
    def __init__(self, voc_size, hidden_size, embedding_size, shared_weight=None):
        super(LMClassifier, self).__init__()

        self.voc_size = voc_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.context2token = nn.Sequential(
            nn.Linear(hidden_size, embedding_size),
            nn.Sigmoid()
        )

        self.token_linear = nn.Linear(embedding_size, self.voc_size)
        if shared_weight is not None:
            print(shared_weight.size())
            print(self.token_linear.weight.size())
            assert shared_weight.size() == self.token_linear.weight.size()
            self.token_linear.weight = shared_weight

        self.inverse_temperature = nn.Parameter(torch.tensor([1.0], dtype=torch.float))

    def forward(self, hidden, lens, token):

        contexts, tokens = [], []

        for i, l in enumerate(lens):
            context = torch.cat((hidden[:l - 2, i, :self.hidden_size // 2],
                                 hidden[2:l, i, self.hidden_size // 2:]),
                                dim=-1)
            contexts.append(context)
            tokens.append(token[1:l - 1, i])

        contexts = torch.cat(contexts, dim=0)
        tokens = torch.cat(tokens, dim=0)

        return F.cross_entropy(self.token_linear(self.context2token(contexts)) * self.inverse_temperature,
                               tokens)

    def predict(self, hidden, lens):

        contexts = []
        for i, l in enumerate(lens):
            context = torch.cat((hidden[:l - 2, i, :self.hidden_size // 2],
                                 hidden[2:l, i, self.hidden_size // 2:]),
                                dim=-1)
            contexts.append(context)

        contexts = torch.cat(contexts, dim=0)

        probs, tokens = (self.token_linear(self.context2token(contexts)) * self.inverse_temperature).softmax(-1).max(-1)

        probs = probs.split((lens - 2).tolist(), dim=0)
        tokens = tokens.split((lens - 2).tolist(), dim=0)

        return [s.tolist() for s in tokens]


class Task(nn.Module):
    def __init__(self, task_name, field_name, hidden_layer):
        super(Task, self).__init__()
        self.task_name = task_name
        self.field_name = field_name
        self.hidden_layer = hidden_layer

    def loss(self, hiddens, lens, data: data.Batch):
        raise NotImplementedError

    def predict(self, hiddens, lens):
        raise NotImplementedError

    def verbose(self, hiddens, lens, data: data.Batch):
        raise NotImplementedError

    def evaluation(self, hiddens, lens, data: data.Batch):
        raise NotImplementedError

    def score(self):
        raise NotImplementedError


class SequenceLabelScorer:
    def __init__(self):
        self.correct_counts = defaultdict(float)
        self.gold_counts = defaultdict(float)
        self.pred_counts = defaultdict(float)

    def stat(self, preds, golds):
        begin = 0
        while begin < len(golds):
            end = begin
            while end < len(golds):
                if golds[end][0:2] in ['S_', 'E_'] or golds[end] in ['O', '*']:
                    end += 1
                    break
                else:
                    end += 1
            if not golds[begin].endswith('*') and not golds[begin].endswith('_O'):
                tag_type = golds[begin][2:]
                self.gold_counts[tag_type] += 1
                if preds[begin:end] == golds[begin:end]:
                    self.correct_counts[tag_type] += 1

            begin = end

        for t in preds:
            if t[0:2] in ['B_', 'S_'] and not t.endswith('_O'):
                self.pred_counts[t[2:]] += 1
            # elif t in ['O', '*']:
            #    pred_counts[t] += 1

    def score(self):
        total_correct = sum(self.correct_counts.values())
        total_gold = sum(self.gold_counts.values())
        total_pred = sum(self.pred_counts.values())

        results = {
            'total_count': total_gold,
            'total_f1': total_correct * 2 / (total_gold + total_pred + 1e-5),
            'total_prec': total_correct / (total_gold + 1e-5),
            'total_recall': total_correct / (total_pred + 1e-5)
        }
        for name in self.gold_counts.keys():
            results['%s_count' % name] = self.gold_counts[name]
            results['%s_f1' % name] = self.correct_counts[name] * 2 / (
                        self.pred_counts[name] + self.gold_counts[name] + 1e-5)
            results['%s_prec' % name] = self.correct_counts[name] / (self.pred_counts[name] + 1e-5)
            results['%s_recall' % name] = self.correct_counts[name] / (self.gold_counts[name] + 1e-5)

        return results

    def reset(self):
        self.correct_counts = defaultdict(float)
        self.gold_counts = defaultdict(float)
        self.pred_counts = defaultdict(float)


class BracketSequenceLabelScorer:
    def __init__(self):
        self.correct_counts = defaultdict(float)
        self.gold_counts = defaultdict(float)
        self.pred_counts = defaultdict(float)

    def stat(self, preds, golds):
        pred_spans = {span: level for span, level in self.bmes_to_span(preds)}
        gold_spans = {span: level for span, level in self.bmes_to_span(golds)}

        for span, height in pred_spans.items():
            self.pred_counts['span'] += 1
            self.pred_counts['span-%d' % height] += 1
            if span in gold_spans:
                self.correct_counts['span'] += 1
                self.correct_counts['span-%d' % height] += 1

        for span, height in gold_spans.items():
            self.gold_counts['span'] += 1
            self.gold_counts['span-%d' % height] += 1

    def bmes_to_span(self, tags):
        def _to_tree(begin, end, level=0):
            child_begin = begin
            while child_begin < end and level < len(tags[child_begin]):
                if tags[child_begin][level] == 'S':
                    yield ((child_begin, child_begin + 1), len(tags[child_begin]) - level)
                    child_begin += 1
                else:
                    while child_begin < end and level < len(tags[child_begin]) and tags[child_begin][level] != 'B':
                        child_begin += 1

                    child_end = child_begin + 1
                    while child_end < end and level < len(tags[child_end]) and tags[child_end][level] == 'M':
                        child_end += 1

                    if child_end < end and level < len(tags[child_end]) and tags[child_end][level] == 'E':
                        yield ((child_begin, child_end + 1), len(tags[child_begin]) - level)

                        yield from _to_tree(child_begin, child_end + 1, level + 1)

                    child_begin = child_end + 1

        return _to_tree(0, len(tags), 0)

    def score(self):
        total_correct = sum(self.correct_counts.values())
        total_gold = sum(self.gold_counts.values())
        total_pred = sum(self.pred_counts.values())

        results = {
            'total_count': total_gold,
            'total_f1': total_correct * 2 / (total_gold + total_pred + 1e-5),
            'total_prec': total_correct / (total_gold + 1e-5),
            'total_recall': total_correct / (total_pred + 1e-5)
        }
        for name in self.gold_counts.keys():
            results['%s_count' % name] = self.gold_counts[name]
            results['%s_f1' % name] = self.correct_counts[name] * 2 / (
                        self.pred_counts[name] + self.gold_counts[name] + 1e-5)
            results['%s_prec' % name] = self.correct_counts[name] / (self.pred_counts[name] + 1e-5)
            results['%s_recall' % name] = self.correct_counts[name] / (self.gold_counts[name] + 1e-5)

        return results

    def reset(self):
        self.correct_counts = defaultdict(float)
        self.gold_counts = defaultdict(float)
        self.pred_counts = defaultdict(float)


class LMScorer:
    def __init__(self):
        self.total = 0
        self.correct = 0

    def stat(self, pred, gold):
        for p, g in zip(pred, gold):
            if p == g:
                self.correct += 1
            self.total += 1
        return self

    def score(self):
        return {'lm_prec': self.correct / (self.total + 1e-5)}

    def reset(self):
        self.total = 0
        self.correct = 0


class SpanScorer:
    def __init__(self):
        self.total = 0
        self.acc = 0
        self.prec = 0
        self.recall = 0

    def stat(self, pred, gold):
        self.total += 1
        gold = set(gold)
        pred = set(pred)
        inter = gold.intersection(pred)
        union = gold.union(pred)
        self.acc += len(inter) / len(union)
        self.prec += len(inter) / len(pred)
        self.recall += len(inter) / len(gold)

    def score(self):
        return {
            'total_acc': self.acc / (self.total + 1e-5),
            'total_prec': self.prec / (self.total + 1e-5),
            'total_recall': self.recall / (self.total + 1e-5),
            'total_total': self.total
        }

    def reset(self):
        self.total = 0
        self.acc = 0
        self.prec = 0
        self.recall = 0


class SpanTask(Task):
    def __init__(self,
                 task_name: str, field_name: str, hidden_layer: int,
                 text_field, task_field,
                 model: ContextClassifier):
        super(SpanTask, self).__init__(task_name, field_name, hidden_layer)
        self.text_field = text_field
        self.task_field = task_field
        self.model = model

        self.scorer = SpanScorer()

    def loss(self, hiddens, lens, data: data.Batch):
        if hasattr(data, self.field_name):
            labels = getattr(data, self.field_name)
            return self.model(hiddens[self.hidden_layer], labels)['loss']
        return 0

    def predict(self, hiddens, lens):
        pass

    def verbose(self, hiddens, lens, data: data.Batch):
        if hasattr(data, self.field_name):
            text, _ = data.text
            labels = getattr(data, self.field_name)
            results = self.model.predict(hiddens[self.hidden_layer], labels)[self.model.name]['phrase']

            sentences = [[self.text_field.vocab.itos[w] for w in text[:lens[i], i]] for i in range(lens.size(0))]

            outputs = []
            for sentence, sen_labels in zip(sentences, results):
                output = ''
                for label, pred in sen_labels:
                    gold = set(self.task_field.vocab.itos[i] for i in label.tags.tolist())
                    pred = set(self.task_field.vocab.itos[i] for i in pred)
                    output += '(%d,%d,%s): (%s, %s, %s)\n' % (
                        label.begin, label.end,
                        ''.join(sentence[label.begin:label.end]), self.task_name, gold, pred)
                if len(sen_labels) > 0:
                    outputs.append(''.join(sentence) + '\n' + output + '\n')
                else:
                    outputs.append('')

            return outputs
        else:
            return [[]] * len(lens)

    def evaluation(self, hiddens, lens, data: data.Batch):
        if hasattr(data, self.field_name):
            labels = getattr(data, self.field_name)
            results = self.model.predict(hiddens[self.hidden_layer], labels)[self.model.name]['phrase']

            for sen_res in results:
                for label, pred in sen_res:
                    if label.tags.size(0) > 0:
                        self.scorer.stat(pred, label.tags.tolist())

    def score(self):
        return self.scorer.score()

    def reset_scorer(self):
        self.scorer.reset()


class SequenceLabel(Task):
    def __init__(self,
                 task_name: str, field_name: str, hidden_layer: int,
                 text_field, task_field,
                 model: LinearCRF):
        super(SequenceLabel, self).__init__(task_name, field_name, hidden_layer)
        self.text_field = text_field
        self.task_field = task_field
        self.model = model

        self.scorer = SequenceLabelScorer()

    def loss(self, hiddens, lens, data: data.Batch):
        if hasattr(data, self.field_name):
            mask_tags, golds = getattr(data, self.field_name)
            return self.model.neg_log_likelihood(hiddens[self.hidden_layer], lens, mask_tags)
        return 0

    def predict(self, hiddens, lens):
        return [([self.task_field.vocab.itos[w] for w in words], score)
                for words, score in self.model(hiddens[self.hidden_layer], lens)]

    def verbose(self, hiddens, lens, data: data.Batch):
        def tostr(words: List[str], tags: List[str]):
            for word, tag in zip(words, tags):
                if tag == 'E_O' or tag == 'S_O':
                    yield word + ' '
                elif tag.startswith('E_'):
                    yield '%s/%s ' % (word, tag[2:])
                elif tag.startswith('S_'):
                    yield '%s/%s ' % (word, tag[2:])
                else:
                    yield word

        text, _ = data.text
        pred_results = self.model.predict_with_prob(hiddens[self.hidden_layer], lens)

        sentences = [[self.text_field.vocab.itos[w] for w in text[:lens[i], i]] for i in range(lens.size(0))]
        pred_tags = [([self.task_field.vocab.itos[tag] for tag in tags], score)
                     for (tags, score), length in zip(pred_results, lens)]

        results = ['%s_pred: %s\nscore: %f' % (self.task_name, ''.join(tostr(sentence, tags)), score)
                   for sentence, (tags, score) in zip(sentences, pred_tags)]

        if hasattr(data, self.field_name):
            gold_masks, gold_tags = getattr(data, self.field_name)
            str_tags = [tags[:length] for tags, length in zip(gold_tags, lens)]

            gold_sentences = [''.join(tostr(sentence, tags)) for sentence, tags in zip(sentences, str_tags)]

            results = ['%s_gold: %s\n%s' % (self.task_name, gold, pred)
                       for gold, pred in zip(gold_sentences, results)]

        return results

    def evaluation(self, hiddens, lens, data: data.Batch):
        if hasattr(data, self.field_name):
            _, golds = getattr(data, self.field_name)
            golds = [gold[:length] for gold, length in zip(golds, lens)]
            for (tags, score), gold in zip(self.predict(hiddens, lens), golds):
                self.scorer.stat(tags[1:-1], gold[1:-1])

    def score(self):
        return self.scorer.score()

    def reset_scorer(self):
        self.scorer.reset()


class BracketSequenceLabel(Task):
    def __init__(self,
                 task_name: str, field_name: str, hidden_layer: int,
                 text_field, task_field,
                 model: LinearCRF):
        super(BracketSequenceLabel, self).__init__(task_name, field_name, hidden_layer)
        self.text_field = text_field
        self.task_field = task_field
        self.model = model

        self.scorer = BracketSequenceLabelScorer()

    def loss(self, hiddens, lens, data: data.Batch):
        if hasattr(data, self.field_name):
            (mask_tags, mask_flag), golds = getattr(data, self.field_name)
            return self.model.neg_log_likelihood(hiddens[self.hidden_layer], lens, mask_tags)
        return 0

    def predict(self, hiddens, lens):
        return [([self.task_field.vocab.itos[w] for w in words], score)
                for words, score in self.model(hiddens[self.hidden_layer], lens)]

    def verbose(self, hiddens, lens, data: data.Batch):
        def tostr(words: List[str], tags: List[str]):
            for word, tag in zip(words, tags):
                text = word
                for t in reversed(tag):
                    if t == 'B':
                        text = '[' + text
                    elif t == 'E':
                        text = text + ']'
                    elif t == 'S':
                        text = '[%s]' % text

                if text.endswith(']'):
                    text += ' '
                yield text

        text, _ = data.text
        pred_results = self.model.predict_with_prob(hiddens[self.hidden_layer], lens)

        sentences = [[self.text_field.vocab.itos[w] for w in text[:lens[i], i]] for i in range(lens.size(0))]
        pred_tags = [([self.task_field.vocab.itos[tag] for tag in tags], score)
                     for (tags, score), length in zip(pred_results, lens)]

        results = ['%s_pred: %s\nscore: %f' % (self.task_name, ''.join(tostr(sentence, tags)), score)
                   for sentence, (tags, score) in zip(sentences, pred_tags)]

        if hasattr(data, self.field_name):
            (gold_masks, mask_flag), gold_tags = getattr(data, self.field_name)
            if not mask_flag:
                str_tags = [tags[:length] for tags, length in zip(gold_tags, lens)]

                gold_sentences = [''.join(tostr(sentence, tags)) for sentence, tags in zip(sentences, str_tags)]

                results = ['%s_gold: %s\n%s' % (self.task_name, gold, pred)
                           for gold, pred in zip(gold_sentences, results)]

        return results

    def evaluation(self, hiddens, lens, data: data.Batch):
        if hasattr(data, self.field_name):
            (_, mask_flag), golds = getattr(data, self.field_name)
            if mask_flag:
                return
            golds = [gold[:length] for gold, length in zip(golds, lens)]
            for (tags, score), gold in zip(self.predict(hiddens, lens), golds):
                self.scorer.stat(tags[1:-1], gold[1:-1])

    def score(self):
        return self.scorer.score()

    def reset_scorer(self):
        self.scorer.reset()



class LMTask(Task):
    def __init__(self, name, field_name, hidden_layer, vocab, model: LMClassifier):
        super(LMTask, self).__init__(name, field_name, hidden_layer)
        self.vocab = vocab
        self.model = model

        self.scorer = LMScorer()

    def loss(self, hiddens, lens, data: data.Batch):
        text, lens = getattr(data, self.field_name)
        return self.model(hiddens[self.hidden_layer], lens, text)

    def predict(self, hiddens, lens):
        return [[self.vocab.itos[w] for w in words]
                for words in self.model.predict(hiddens[self.hidden_layer], lens)]

    def evaluation(self, hiddens, lens, data: data.Batch):
        text, lens = getattr(data, self.field_name)
        text = [[self.vocab.itos[w] for w in words[:length]]
                for words, length in zip(text.tolist(), lens.tolist())]
        for words, gold in zip(self.predict(hiddens, lens), text):
            self.scorer.stat(words[1:-1], gold[1:-1])

    def verbose(self, hiddens, lens, data: data.Batch):
        results = self.predict(hiddens, lens)
        results = ['%s:  %s' % (self.task_name, ' '.join([INIT_TOKEN] + words + [EOS_TOKEN]))
                   for words in results]
        return results

    def score(self):
        return self.scorer.score()

    def reset_scorer(self):
        self.scorer.reset()


class Tagger(nn.Module):
    def __init__(self,
                 name2embedding: nn.ModuleDict,
                 encoder: nn.Module,
                 name2task: nn.ModuleDict):
        super(Tagger, self).__init__()

        self.name2embedding = name2embedding
        self.encoder = encoder
        self.name2task = name2task

    def _encode(self, batch: data.Batch):

        embeds, lens = [], None
        for name, embedding in self.name2embedding.items():
            if name == 'text':
                sens, lens = getattr(batch, 'text')
                embeds.append(embedding(sens))
            else:
                data = getattr(batch, name)
                if isinstance(data, Tuple):
                    for seq in data:
                        embeds.append(embedding(seq))
                else:
                    embeds.append(embedding(data))

        embed = torch.cat(embeds, dim=-1)

        if isinstance(self.encoder, ElmoEncoder):
            forwards, backwards = self.encoder(embed, lens)
            hiddens = [torch.cat((f, b), dim=-1) for f, b in zip(forwards, backwards)]
        else:
            hiddens, _ = self.encoder(embed)

        return hiddens, lens

    def loss(self, data: data.Batch):
        hiddens, lens = self._encode(data)
        return {name: task.loss(hiddens, lens, data)
                for name, task in self.name2task.items() if hasattr(data, task.field_name)}

    def predict(self, data: data.Batch):
        hiddens, lens = self._encode(data)

        return {name: task.predict(hiddens, lens) for name, task in self.name2task.items()}

    def verbose(self, data: data.Batch):
        hiddens, lens = self._encode(data)
        results = [task.verbose(hiddens, lens, data) for name, task in self.name2task.items()]

        for i in range(lens.size(0)):
            for res in results:
                print(res[i])

    def evaluation(self, datasets):
        self.eval()

        for task in self.name2task.values():
            task.reset_scorer()

        for dataset in datasets:
            for batch in tqdm(dataset, desc='eval', total=len(dataset)):
                hiddens, lens = self._encode(batch)

                for task in self.name2task.values():
                    task.evaluation(hiddens, lens, batch)

        return {name: task.score() for name, task in self.name2task.items()}

    def print(self, batch: data.Batch):
        text, text_len = batch.text
        for name in self.name2task.keys():
            gold_masks, gold_tags = batch[name]
            for i in range(len(text_len)):
                length = text_len[i]
                print(name + ': ' + ' '.join([self.words.itos[w] + '#' + t
                                              for w, t in zip(text[0:length, i].data.tolist(), gold_tags[i])]))


def RandomIterator(its):
    # cum_weights = list(itertools.accumulate([len(it) for it in its]))
    its = [iter(it) for it in its]

    while True and len(its) > 0:
        for index in random.choices(list(range(len(its)))):
            yield next(its[index])


class FineTrainer:
    def __init__(self,
                 config,
                 model: Tagger,
                 trains, valids, tests):
        self.config = config
        self.model = model
        # self.optimizer = optim.SGD(self.model.parameters(), lr=3e-2, momentum=0.95)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-3)

        self.task_optimizers = {name: optim.Adam(task.parameters(), lr=1e-3)
                                for name, task in self.model.name2task.items()}

        self.shared_optimizer = optim.Adam(list(self.model.name2embedding.parameters()) + list(self.model.encoder.parameters()),
                                           lr=1e-3)

        self.trains, self.valids, self.tests = \
            trains, valids, tests

        self.train_its = [iter(t) for t in trains]

        self.valid_step = config.valid_step
        self.warm_up_step = config.warm_up_step
        self.checkpoint_dir = config.checkpoint_path

        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir)

    def state_dict(self, train=True, optimizer=True):

        states = OrderedDict()
        states['model'] = self.model.state_dict()
        if optimizer:
            states['shared_optimizer'] = self.shared_optimizer.state_dict()
            for name, optim in self.task_optimizers.items():
                states[name] = optim.state_dict()
        # if train:
        #     states['train_it'] = self.train_it.state_dict()
        return states

    def load_state_dict(self, states, strict):

        self.model.load_state_dict(states['model'], strict=strict)
        # if 'optimizer' in states:
            # self.optimizer.load_state_dict(states['optimizer'])

        # if 'train_it' in states:
        #     self.train_it.load_state_dict(states['train_it'])

    def load_checkpoint(self, path, strict=True):
        states = torch.load(path)
        self.load_state_dict(states, strict=strict)

    def train_one(self) -> Dict[str, float]:
        self.model.train()
        self.model.zero_grad()

        losses = defaultdict(float)
        for it in self.train_its:
            batch = next(it)
            for name, loss in self.model.loss(batch).items():
                losses[name] += loss / len(batch)

            del batch

        sum(losses.values()).backward()

        # Step 3. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        for name, optimzier in self.task_optimizers.items():
            optimzier.step()
        self.shared_optimizer.step()

        return {name: loss.item() for name, loss in losses.items()}

    def valid(self, valids) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            total_losses = Counter()
            num_samples = 0

            sample = False
            for valid_it in valids:
                for _, valid_batch in tqdm(enumerate(valid_it), desc='valid', total=len(valid_it)):
                    losses = self.model.loss(valid_batch)

                    total_losses.update({name: l.item() for name, l in losses.items()})
                    num_samples += len(valid_batch)

                    if not sample and random.random() < 0.02:
                        self.model.verbose(valid_batch)

                    del valid_batch

            return {name: loss / num_samples for name, loss in total_losses.items()}

    @staticmethod
    def tabulate_format(scores: Dict) -> str:
        return tabulate([[key, [(n.rsplit('_', maxsplit=1)[1], '%0.5f' % s) for n, s in values]]
                         for key, values in
                         itertools.groupby(scores.items(), key=lambda item: item[0].rsplit('_', maxsplit=1)[0])])

    def train(self):
        total_losses, total_count, start = Counter(), 1e-10, time.time()
        checkpoint_losses = deque()

        def infinite_stream(start):
            while True:
                yield start
                start += 1

        for step in tqdm(infinite_stream(1), desc='train step'):
            losses = self.train_one()

            total_losses.update(losses)
            total_count += 1

            # if step == self.warm_up_step:
                # self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

            if step % self.valid_step == 0:
                train_speed = total_count / (time.time() - start)

                inference_start = time.time()
                valid_losses = self.valid(self.valids)

                self.checkpoint(checkpoint_losses, sum(valid_losses.values()))

                for name, loss in total_losses.items():
                    if name in valid_losses:
                        print('%s train loss=%.6f\t\tvalid loss=%.6f' % (name, loss / total_count, valid_losses[name]))
                        self.summary_writer.add_scalars(
                            'loss',
                            {'train_%s' % name: loss / total_count, 'valid_%s' % name: valid_losses[name]},
                            step)
                    else:
                        print('%s train loss=%.6f' % (name, loss / total_count))
                        self.summary_writer.add_scalars(
                            'loss',
                            {'train_%s' % name: loss / total_count},
                            step)

                # print('speed:   train %.2f sentence/s  valid %.2f sentence/s\n\n' %
                #      (train_speed, len(self.valid_it.dataset) / (time.time() - inference_start)))

                total_losses, total_count, start = Counter(), 1e-10, time.time()

            if step % (self.valid_step * 2) == 0:
                with torch.no_grad():
                    '''
                    print([(embedding.name, embedding.scale_ratio.item()) for embedding in self.model.tag_embeddings])
                    self.summary_writer.add_scalars(
                        'tag_weights',
                        {embedding.name: embedding.scale_ratio.item() for embedding in self.model.tag_embeddings},
                        step)
                    '''
                    eval_start = time.time()
                    results = self.model.evaluation(self.valids)
                    # print('speed: eval %.2f sentence/s' % (len(self.valid_it.dataset)/(time.time() - eval_start)))
                    for name, result in results.items():
                        print('------- %s -------' % name)
                        print(self.tabulate_format(result))
                        self.summary_writer.add_scalars('%s_eval_valid' % name, result, step)

                    results = self.model.evaluation(self.tests)
                    for name, result in results.items():
                        print('------- %s -------' % name)
                        print(self.tabulate_format(result))
                        self.summary_writer.add_scalars('%s_eval_test' % name, result, step)

    def checkpoint(self, checkpoint_losses, valid_loss):
        if len(checkpoint_losses) == 0 or checkpoint_losses[-1] > valid_loss:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

            checkpoint_losses.append(valid_loss)

            torch.save(self.state_dict(),
                       '%s/model-%0.4f' % (self.checkpoint_dir, valid_loss))

            if len(checkpoint_losses) > 5:
                removed = checkpoint_losses.popleft()
                try:
                    os.remove('%s/model-%0.4f' % (self.checkpoint_dir, removed))
                except:
                    pass

    @staticmethod
    def load_dataset(dataset, text_fields, bracket_field, config):

        train, valid, test = dataset(text_fields, bracket_field)

        tasks = {}
        for field_name, field in train.fields.items():
            if not field.is_target or isinstance(field, BracketField):
                continue

            if isinstance(field, PartialField):
                field.build_vocab(train, min_freq=config.text_min_freq)
                model = LinearCRF(config.encoder_hidden_dim,
                                  len(field.vocab),
                                  field.vocab.transition_constraints,
                                  attention_num_heads=config.attention_num_heads,
                                  dropout=0.3)
                tasks[field_name] = SequenceLabel(field_name, field_name, -2, train.fields['text'], field, model)
            elif isinstance(field, LabelField):
                model = ContextClassifier(field_name, field.vocab, config.encoder_hidden_dim, config.label_dim)
                tasks[field_name] = SpanTask(field_name, field_name, -1, train.fields['text'], field, model)
            else:
                raise NotImplementedError()

        return (train, valid, test), tasks

    @classmethod
    def create(cls, fine_config):
        text_field = data.Field(include_lengths=True, init_token=INIT_TOKEN, eos_token=EOS_TOKEN,
                                pad_token=PAD_TOKEN)

        pretrained_embedding = vocab.Vectors(fine_config.pretrained_path, max_vectors=3000000)
        keyword_field = KeywordField.from_pretrained(
            pretrained_embedding,
            init_token=INIT_TOKEN,
            eos_token=EOS_TOKEN,
            pad_token=PAD_TOKEN)

        bracket_field = BracketField(init_token=INIT_TOKEN, eos_token=EOS_TOKEN, is_target=True)

        datasets = []
        tasks = {}

        for name, dataset in fine_config.datasets.items():
            (train, valid, test), _tasks = cls.load_dataset(
                dataset,
                (('text', 'keyword'), (text_field, keyword_field)),
                ('ctb_bracket', bracket_field),
                fine_config)
            datasets.append((train, valid, test))
            tasks.update(_tasks)

        text_field.build_vocab(*(train for train, _, _ in datasets),
                               min_freq=fine_config.text_min_freq)

        bracket_field.build_vocab(*(train for train, _, _ in datasets),
                               min_freq=fine_config.text_min_freq)

        model = MaskedCRF(fine_config.encoder_hidden_dim,
                          len(bracket_field.vocab),
                          bracket_field.vocab.transition_constraints,
                          attention_num_heads=fine_config.attention_num_heads,
                          dropout=0.3)

        tasks['ctb_bracket'] = BracketSequenceLabel('ctb_bracket', 'ctb_bracket', -1, train.fields['text'], bracket_field, model)

        if fine_config.embedding_dim == fine_config.embedding_dim:
            # only character token
            def _extend(vocab, words):
                for w in words:
                    if len(w) == 1 and w not in vocab.stoi:
                        vocab.itos.append(w)
                        vocab.stoi[w] = len(vocab.itos) - 1

            _extend(text_field.vocab, pretrained_embedding.itos)
            text_field.vocab.set_vectors(pretrained_embedding.stoi,
                                         pretrained_embedding.vectors,
                                         dim=fine_config.embedding_dim)

            embedding = nn.Embedding.from_pretrained(
                text_field.vocab.vectors,
                freeze=False,
                padding_idx=text_field.vocab.stoi[PAD_TOKEN])
        else:
            embedding = nn.Embedding(
                len(text_field.vocab),
                fine_config.embedding_dim,
                padding_idx=text_field.vocab.stoi[PAD_TOKEN],
                scale_grad_by_freq=False
            )

        keyword_embedding = nn.Embedding.from_pretrained(
            keyword_field.vocab.vectors,
            freeze=True)

        embeddings = nn.ModuleDict({
            'text': embedding,
            'keyword': keyword_embedding
        })

        encoder = StackRNN(
            'LSTM',
            embedding.embedding_dim + keyword_embedding.embedding_dim * 2,
            fine_config.encoder_hidden_dim // 2,
            fine_config.encoder_num_layers,
            bidirectional=True,
            # residual=fine_config.encoder_residual,
            dropout=0.3)

        # lm_cls = LMClassifier(len(text_field.vocab), fine_config.encoder_hidden_dim, fine_config.embedding_dim, embedding.weight)

        tasks = nn.ModuleDict({
            **tasks,
            # 'lm': LMTask('lm', 'text', 1, text_field.vocab, lm_cls)
        })

        fine_model = Tagger(embeddings,
                            encoder,
                            tasks)

        fine_model.to(fine_config.device)

        trains, valids, tests = [], [], []

        for train, valid, test in datasets:
            if valid is None:
                train_it, test_it = BucketIterator.splits([train, test],
                                      batch_sizes=fine_config.batch_sizes,
                                      shuffle=True,
                                      device=fine_config.device,
                                      sort_within_batch=True)
                train_it.repeat = True
                trains.append(train_it)
                tests.append(test_it)
            else:
                train_it, valid_it, test_it = \
                    BucketIterator.splits([train, valid, test],
                                          batch_sizes=fine_config.batch_sizes,
                                          shuffle=True,
                                          device=fine_config.device,
                                          sort_within_batch=True)
                train_it.repeat = True
                trains.append(train_it)
                valids.append(valid_it)
                tests.append(test_it)

        return cls(fine_config,
                   fine_model,
                   trains, valids, tests)


class NERConfig:
    def __init__(self, model_dir: str = None):
        self.data_dir = './data/ner/data'
        self.model_dir = model_dir if model_dir else './ner/model'

        os.makedirs(self.model_dir, exist_ok=True)

        self.train = 'example.train'
        self.valid = 'example.dev'
        self.test = 'example.test'

        self.dataset_class = NamedEntityData

        self.batch_sizes = [16, 32, 32]

        # vocabulary
        self.text_min_freq = 5
        # text_min_size = 50000

        self.tag_min_freq = 5
        # tag_min_size = 50000

        self.common_size = 1000

        self.char_tag_dim = 10

        self.taggers = []  # [(radical, 32)]

        self.ngram_taggers = [place_ngram, person_ngram, digit_ngram, quantifier_ngram, idioms_ngram, org_ngram]
        # (jieba_pos, 64), (place, 8), (person, 8), (idioms, 8), (organizations, 8), ]

        # model
        self.embedding_dim = 256
        self.encoder_hidden_dim = 256
        self.encoder_num_layers = 2
        self.encoder_residual = False
        self.attention_num_heads = 8

        self.loss = 'nll'  # ['nll', 'focal_loss']

        self.device = torch.device('cpu')  # torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.cached_dataset_prefix = os.path.join(self.data_dir, 'dataset')
        self.checkpoint_path = os.path.join(self.model_dir, 'checkpoint')
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # summary parameters
        self.summary_dir = os.path.join(self.model_dir, 'summary')
        os.makedirs(self.summary_dir, exist_ok=True)

        self.valid_step = 400


class CTBPOSConfig:
    def __init__(self, model_dir: str = None):
        self.data_dir = './data/pos/data'
        self.model_dir = model_dir if model_dir else './pos/model'
        os.makedirs(self.model_dir, exist_ok=True)

        self.train = 'train'
        self.valid = 'valid'
        self.test = 'gold'

        self.dataset_class = POSData

        self.batch_sizes = [16, 32, 32]

        # vocabulary
        self.text_min_freq = 5
        # text_min_size = 50000

        self.tag_min_freq = 1
        # tag_min_size = 50000

        self.common_size = 1000

        # model
        self.vocab_size = 100
        self.embedding_dim = 64
        self.encoder_hidden_dim = 64
        self.encoder_num_layers = 2
        self.encoder_residual = False
        self.attention_num_heads = None

        self.taggers = []  # [(radical, 32)]

        self.ngram_taggers = []  # [place_ngram, person_ngram, digit_ngram, quantifier_ngram, idioms_ngram, org_ngram]
        # (jieba_pos, 64), (place, 8), (person, 8), (idioms, 8), (organizations, 8), ]

        self.loss = 'nll'  # ['nll', 'focal_loss']

        self.device = torch.device('cpu')  # torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.cached_dataset_prefix = os.path.join(self.data_dir, 'dataset')
        self.checkpoint_path = os.path.join(self.model_dir, 'checkpoint')
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # summary parameters
        self.summary_dir = os.path.join(self.model_dir, 'summary')
        os.makedirs(self.summary_dir, exist_ok=True)

        self.valid_step = 200


class CNC_config:
    def __init__(self, model_dir: str = None):
        self.data_dir = './data/cnc/data'
        self.model_dir = model_dir if model_dir else os.path.join(self.data_dir, 'model')
        os.makedirs(self.model_dir, exist_ok=True)

        self.train = 'cnc_train.txt'
        self.valid = 'cnc_dev.txt'
        self.test = 'cnc_test.txt'

        self.dataset_class = POSData

        self.batch_sizes = [16, 64, 64]

        # vocabulary
        self.text_min_freq = 5
        # text_min_size = 50000

        self.tag_min_freq = 1
        # tag_min_size = 50000

        self.common_size = 1000

        # model
        self.vocab_size = 100
        self.embedding_dim = 64
        self.encoder_hidden_dim = 64
        self.encoder_num_layers = 2
        self.encoder_residual = False
        self.attention_num_heads = None

        self.taggers = []  # [(radical, 32)]

        self.ngram_taggers = []  # [place_ngram, person_ngram, digit_ngram, quantifier_ngram, idioms_ngram, org_ngram]
        # (jieba_pos, 64), (place, 8), (person, 8), (idioms, 8), (organizations, 8), ]

        self.loss = 'nll'  # ['nll', 'focal_loss']

        self.device = torch.device('cpu')  # torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.cached_dataset_prefix = os.path.join(self.data_dir, 'dataset')
        self.checkpoint_path = os.path.join(self.model_dir, 'checkpoint')
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # summary parameters
        self.summary_dir = os.path.join(self.model_dir, 'summary')
        os.makedirs(self.summary_dir, exist_ok=True)

        self.valid_step = 500


class PeopleConfig:
    def __init__(self, model_dir: str = None):
        self.data_dir = 'data/people'
        self.model_dir = model_dir if model_dir else './ner/model'
        os.makedirs(self.model_dir, exist_ok=True)

        self.train = '2014'
        self.valid = None
        self.test = None

        self.dataset_class = People2014

        self.batch_sizes = [16, 32, 32]

        # vocabulary
        self.text_min_freq = 5
        # text_min_size = 50000

        self.tag_min_freq = 1
        # tag_min_size = 50000

        self.common_size = 1000

        # model
        self.vocab_size = 100
        self.embedding_dim = 64
        self.encoder_hidden_dim = 64
        self.encoder_num_layers = 2
        self.encoder_residual = False
        self.attention_num_heads = None

        self.taggers = []  # [(radical, 32)]

        self.ngram_taggers = []  # [place_ngram, person_ngram, digit_ngram, quantifier_ngram, idioms_ngram, org_ngram]
        # (jieba_pos, 64), (place, 8), (person, 8), (idioms, 8), (organizations, 8), ]

        self.loss = 'nll'  # ['nll', 'focal_loss']

        self.device = torch.device('cpu')  # torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.cached_dataset_prefix = os.path.join(self.data_dir, 'dataset')
        self.checkpoint_path = os.path.join(self.model_dir, 'checkpoint')
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # summary parameters
        self.summary_dir = os.path.join(self.model_dir, 'summary')
        os.makedirs(self.summary_dir, exist_ok=True)

        self.valid_step = 200


class ChunkConfig:
    def __init__(self, model_dir: str = None):
        self.data_dir = './'
        self.model_dir = model_dir if model_dir else 'chunk/model'
        os.makedirs(self.model_dir, exist_ok=True)

        self.train = 'chunk/train.txt.manual'
        self.valid = 'chunk/dev.txt.manual'
        self.test = 'chunk/test.txt.manual'

        self.dataset_class = ChunkDataset

        self.batch_sizes = [16, 32, 32]

        # vocabulary
        self.text_min_freq = 1
        # text_min_size = 50000

        self.tag_min_freq = 1
        # tag_min_size = 50000

        self.common_size = 1000

        # model
        self.vocab_size = 100
        self.embedding_dim = 128
        self.encoder_hidden_dim = 128
        self.encoder_num_layers = 2
        self.encoder_residual = False
        self.attention_num_heads = None

        self.taggers = []  # [(radical, 32)]

        self.ngram_taggers = []  # [place_ngram, person_ngram, digit_ngram, quantifier_ngram, idioms_ngram, org_ngram]
        # (jieba_pos, 64), (place, 8), (person, 8), (idioms, 8), (organizations, 8), ]

        self.loss = 'nll'  # ['nll', 'focal_loss']

        self.device = torch.device('cpu')  # torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.cached_dataset_prefix = os.path.join(self.data_dir, 'dataset')
        self.checkpoint_path = os.path.join(self.model_dir, 'checkpoint')
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # summary parameters
        self.summary_dir = os.path.join(self.model_dir, 'summary')
        os.makedirs(self.summary_dir, exist_ok=True)

        self.valid_step = 200


class BaikeConfig:
    def __init__(self):
        self.root = './baike/preprocess-char'
        self.train_prefix = 'sentence.url.gz'
        self.valid_file = 'valid.gz'
        self.distant_dict = 'distant.dic'
        self.stop_dict = 'stopword.all'

        self.voc_max_size = 50000
        self.voc_min_freq = 50
        self.key_max_size = 150000
        self.key_min_freq = 50
        self.subtitle_max_size = 150000
        self.subtitle_min_freq = 50
        self.attr_max_size = 150000
        self.attr_min_freq = 50
        self.entity_max_size = 150000
        self.entity_min_freq = 50

        self.label_dim = 128


baike_config = BaikeConfig()


class MultiConfig:
    def __init__(self, model_dir: str = None):
        self.data_dir = './'
        self.model_dir = model_dir if model_dir else 'bracket/model'
        os.makedirs(self.model_dir, exist_ok=True)

        self.pretrained_path = './word2vec/Tencent_AILab_ChineseEmbedding.txt.gz'
        self.pretrained_dim = 200


        self.datasets = {
            '''
            'ctb': lambda text_field: ChunkDataset.splits(text_field,
                                                          path='chunk',
                                                          train='train.txt.manual',
                                                          validation='dev.txt.manual',
                                                          test='test.txt.manual'),
            
            '''
            'msra': lambda text_field, bracket_field: MSRA.splits(text_field,
                                                   bracket_field,
                                                   path='data/msra/data',
                                                   train='msra_bakeoff3_training.utf8.xml',
                                                   test='msra_bakeoff3_test.utf8.xml'),

            'people98': lambda text_field, bracket_field: People98.splits(text_field,
                                                           bracket_field,
                                                           path='data/people/1998f',
                                                           train='all.txt'),

            'bracket': lambda text_field, bracket_field: Bracket.splits(text_field,
                                                                        bracket_field,
                                                                        path='bracket',
                                                                        train='train.txt'),

            'baike': lambda text_field, bracket_field: Baike.splits(text_field,
                                                         bracket_field,
                                                         baike_config,
                                                         path=baike_config.root,
                                                         train=baike_config.train_prefix,
                                                         test=baike_config.valid_file)
        }


        self.batch_sizes = [16, 32, 32]

        # vocabulary
        self.text_min_freq = 10
        # text_min_size = 50000

        self.tag_min_freq = 1
        # tag_min_size = 50000

        self.common_size = 1000

        # model
        self.vocab_size = 100
        self.embedding_dim = 200

        self.encoder_hidden_dim = 256
        self.encoder_num_layers = 2
        self.encoder_residual = False
        self.attention_num_heads = 8

        self.label_dim = 128


        self.loss = 'nll'  # ['nll', 'focal_loss']

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.cached_dataset_prefix = os.path.join(self.data_dir, 'dataset')
        self.checkpoint_path = os.path.join(self.model_dir, 'checkpoint')
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # summary parameters
        self.summary_dir = os.path.join(self.model_dir, 'summary')
        os.makedirs(self.summary_dir, exist_ok=True)

        self.valid_step = 500
        self.warm_up_step = 10000
        self.max_epoch = 10


configs = {'ctb': CTBPOSConfig, 'ner': NERConfig, 'people': PeopleConfig, 'cnc': CNC_config, 'chunk': ChunkConfig}

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Preprocess baike corpus and save vocabulary')
    argparser.add_argument('--checkpoint', type=str, help='checkpoint path')
    argparser.add_argument('--task', type=str, help='task type, [ctb, ner, people, cnc, chunk]')

    args = argparser.parse_args()

    assert args.task in configs

    SEED = 1234
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    trainer = FineTrainer.create(MultiConfig())

    trainer.train()
