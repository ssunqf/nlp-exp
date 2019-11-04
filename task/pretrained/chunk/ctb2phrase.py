#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import sys
import re
import itertools
from typing import List
from collections import Counter, defaultdict
from tqdm import tqdm
import argparse
import subprocess
from os import listdir, makedirs
from os.path import isfile, join, isdir, exists


def is_eng_num(c):
    return 'a' <= c <= 'z' or 'A' <= c <= 'Z' or '0' <= c <= '9'

class Node:
    def __init__(self, tag, level, **kwargs):
        self.tag = tag
        self.level = level
        self.__dict__.update(kwargs)

    def is_leaf(self):
        return 'word' in self.__dict__

    def __str__(self):

        if self.is_leaf():
            return '([%d, %d] %s %d %s)' % (self.begin, self.end, self.tag, self.level, self.word)
        else:
            return '([%d, %d] %s %d %s)' % (self.begin, self.end, self.tag, self.level, ' '.join([str(n) for n in self.children]))

    def path_to_leaves(self, path):

        if self.is_leaf():
            yield self, path.copy()
        else:
            path.append(self)
            for child in self.children:
                yield from child.path_to_leaves(path)
            path.pop()

    def leaves(self):
        if self.is_leaf():
            yield self
        else:
            for child in self.children:
                yield from child.leaves()

    def _boundary_leaf(self):
        assert self.tag == 'S'
        leaves = list(self.leaves())

        offset = 0
        for li, leaf in enumerate(leaves):
            if li > 0 and (leaves[li-1].word[-1] and is_eng_num(leaves[li].word[0])):
                offset += 1

            leaf.begin = offset
            offset += len(leaf.word)
            leaf.end = offset

    def boundary(self):
        if self.tag == 'S':
            self._boundary_leaf()

        if not self.is_leaf():
            for child in self.children:
                child.boundary()
            self.begin = self.children[0].begin
            self.end = self.children[-1].end
        return self

    def traversal(self):
        filtered = {'DNP'}
        child_tags = None
        if not self.is_leaf():
            if len(self.children) > 0:
                for child in self.children:
                    yield from child.traversal()
                child_tags = {child.tag for child in self.children}

        if not child_tags or len(filtered.intersection(child_tags)) == 0:
            yield self


def read_sentence(files: List[str]):
    vocab = Counter()
    counter = Counter()
    children_counter = Counter()

    rule_counter = defaultdict(Counter)

    def _tokenize(file):
        with open(file) as input:
            for line in input:
                line = line.strip()
                if line.startswith('<'):
                    continue

                for token in line.replace('(', ' ( ').replace(')', ' ) ').split():
                    yield token

    def _is_prune_node(child):
        return not child.is_leaf() and child.tag in {
            'IP', 'CP', 'FRAG', 'FLR', 'DNP', 'DVP', 'PRN', 'UCP', 'CLP', 'LCP', 'DFL', 'INC', 'SKIP', 'TYPO', 'OTH',
            'WHPP', 'INTJ'}

    def _is_prune_vp(child):
        return not child.is_leaf() and child.tag in {'VP'} and (
                'VP' in set(grand.tag for grand in child.children)  # VP -> * NP *
        )

    def _is_prune_np(child):
        return not child.is_leaf() and child.tag in {'NP'} and (
                'VP' in set(grand.tag for grand in child.children)
        )

    def _prune_children(children):

        new_children = []
        for child in children:
            if child.is_leaf():
                new_children.append(child)
            elif child.tag in {'IP', 'CP', 'DNP', 'DVP', 'PRN', 'UCP', 'LCP'}:
                new_children.extend(child.children)
            elif child.tag in {'VP'} and 'VP' in [grand_child.tag for grand_child in child.children]:
                pass

    # 合并连续的NT为NP
    # ( (FRAG  (NN 新华社)
    #          (NR 西安)
    #          (NT 十二月)
    #          (NT 三十日)
    #          (NN 电)
    #          (PU （)
    #          (NN 记者)
    #          (NR 张连业)
    #          (PU ）) ))

    # ( (FRAG  (NN 新华社)
    #          (NR 西安)
    #          (NP (NT 十二月)
    #              (NT 三十日))
    #          (NN 电)
    #          (PU （)
    #          (NN 记者)
    #          (NR 张连业)
    #          (PU ）) ))
    def _merge_NT(tag, children: List[Node]):
        if len(children) < 3:
            return children

        results = []
        begin = 0
        while begin < len(children):
            if children[begin].tag != 'NT':
                results.append(children[begin])
                begin = begin + 1
            else:
                end = begin + 1
                while end < len(children) and children[end].tag == 'NT':
                    end = end + 1

                if end - begin == len(children):
                    return children
                elif end - begin < len(children):
                    results.append(Node('NP', children[begin].level - 1, children=children[begin:end]))

                begin = end

        return results

    def _insert_NP(tag, children: List[Node]):
        if tag == 'NP':
            return children

        results = [
            Node('NP', child.level - 1, children=[child]) if child.is_leaf() and child.tag in {'NN', 'NR'} else child
            for child in children]
        return results

    # VP -> ADVP VP
    def _flat_vp(tag, children):
        if tag != 'VP' or len(children) < 2:
            return children

        if children[-2].tag in {'VP', 'VV'} and children[-1].tag in {'VP'}:
            return children[0:-1] + children[-1].children
        else:
            return children

    def _make_node(tokens, level):
        # 只保留第一个类型
        tag = next(tokens)

        # if not (tag == '-NONE-'):
        #    tag = re.split('[-=]', tag)[0]

        # VCD 并列动词复合(VCD (VV 投资)    (VV 办厂))
        # VCP VV+VC 动词+是
        # VNV A不A，A一A
        # VPT V的R，或V不R (VPT (VV 得)   (AD 不)   (VV 到))
        # VRD 动词结果复合(VRD (VV 呈现) (VV 出))
        # VSB 定语+核心复合VSB (VV 加速) (VV 建设))
        if tag in {'VCD', 'VCP', 'VNV', 'VPT', 'VRD', 'VSB'}:
            tag = 'VP'

        second = next(tokens)

        if second == '(':
            children = []
            while second == '(':
                child = _make_node(tokens, level + 1)
                if child:
                    if child.is_leaf():
                        children.append(child)
                    else:
                        children.append(child)

                second = next(tokens)
            assert second == ')'

            if len(children) == 0:
                return None
            else:
                # children = _merge_NT(tag, children)
                # children = _insert_NP(tag, children)
                # children = _flat_vp(tag, children)
                counter[tag] += 1
                children_counter[tag] += len(children)

                return Node(tag, level, children=children)
        else:
            third = next(tokens)
            assert third == ')'

            def _is_filterd(tag, word):
                empty = re.compile('^(\*OP\*|\*pro\*|\*PRO\*|\*PNR\*|\*RNR\*|\*T\*|\*？\*|\*-[0-9]+)')
                return tag in {'-NONE-'} or empty.search(word)

            if not _is_filterd(tag, second):
                vocab.update(second)
            return Node(tag, level, word=second) if not _is_filterd(tag, second) else None

    def _make_sentence(tokens):

        left = next(tokens)
        while left == '(':
            left2 = next(tokens)
            assert left2 == '('
            root = _make_node(tokens, 1)
            root.tag = 'S'
            root.boundary()
            yield root

            right = next(tokens)
            assert right == ')'

            left = next(tokens)

    for file in files:
        yield from _make_sentence(_tokenize(file))

    print('\t'.join(['%s: %d %f' % (k, v, float(children_counter[k]) / v) for k, v in counter.items()]))

    print('vocab size = %d' % len(vocab))
    print(vocab.most_common())
    for tag, counters in rule_counter.items():
        print(tag, counters.most_common())


def split(ctb_root):
    chtbs = [f for f in listdir(ctb_root) if isfile(join(ctb_root, f)) and f.startswith('chtb')]
    folder = {}
    for f in chtbs:
        tag = f[-6:-4]
        if tag not in folder:
            folder[tag] = []
        folder[tag].append(f)
    train, dev, test = [], [], []
    for tag, files in folder.items():
        t = int(len(files) * .8)
        d = int(len(files) * .9)
        train += files[:t]
        dev += files[t:d]
        test += files[d:]
    return train, dev, test


def ctb_to_linear(files, ctb_root, out_path):
    print('Generating ' + out_path)
    with open(out_path, 'w') as out:
        for sentence in tqdm(read_sentence([join(ctb_root, file) for file in files])):
            print('sentence', sentence)
            leaves = list(sentence.leaves())
            print(' '.join(leaf.word for leaf in leaves))
            print('\n'.join(list(str(node) for node in sentence.traversal() if re.split('[-=]', node.tag)[0] in ['NP', 'QP', 'NN', 'NR'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine Chinese Treebank 8 bracketed files into train/dev/test set')
    parser.add_argument("--ctb", required=True,
                        help='The root path to Chinese Treebank 8')
    parser.add_argument("--output", required=True,
                        help='The folder where to store the output train.txt/dev.txt/test.txt')

    args = parser.parse_args()

    training, development, test = split(args.ctb)

    root_path = args.output
    if not exists(root_path):
        makedirs(root_path)

    ctb_to_linear(training, args.ctb, join(root_path, 'train.txt'))

    ctb_to_linear(development, args.ctb, join(root_path, 'dev.txt'))
    ctb_to_linear(test, args.ctb, join(root_path, 'test.txt'))
