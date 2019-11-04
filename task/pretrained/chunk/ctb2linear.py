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


class Node:
    def __init__(self, tag, level, **kwargs):
        self.tag = tag
        self.level = level
        self.__dict__.update(kwargs)

    def is_leaf(self):
        return 'word' in self.__dict__

    def __str__(self):

        if self.is_leaf():
            return '(%s %d %s)' % (self.tag, self.level, self.word)
        else:
            return '(%s %d %s)' % (self.tag, self.level, ' '.join([str(n) for n in self.children]))

    def path_to_leaves(self, path):

        if self.is_leaf():
            yield self, path.copy()
        else:
            path.append(self)
            for child in self.children:
                yield from child.path_to_leaves(path)
            path.pop()


def absolute_scale(leaves):
    def _pair(first, second):
        first_leaf, first_path = first
        second_leaf, second_path = second
        for i in range(min(len(first_path), len(second_path)) - 1, -1, -1):
            if id(first_path[i]) == id(second_path[i]):
                return first_path[i].level, first_path[i].tag, first_path[-1].tag if len(first_path[-1].children) == 1 else None

    num_eng = re.compile('[a-zA-Z0-9]$')
    def _to_char(first, second, scale):
        first_leaf, first_path = first
        second_leaf, second_path = second
        scale, tag, unary_tag = scale

        space = []
        if num_eng.match(first_leaf.word[-1]) and num_eng.match(second_leaf.word[0]):
            space = [(' ', (scale, tag, 'SPACE'))]

        if len(first_leaf.word) == 1:
            return [(first_leaf.word, (scale, tag, 'B-' + first_leaf.tag))] + space
        else:
            return list(zip(first_leaf.word,
                            [(first_path[-1].level, first_path[-1].tag, 'B-' + first_leaf.tag)] +
                            [(first_path[-1].level, first_path[-1].tag, 'I-' + first_leaf.tag)] * (len(first_leaf.word) - 2) +
                            [(scale, tag, 'I-' + first_leaf.tag)])) + space

    def _last(leaf):
        leaf, path = leaf

        if len(leaf.word) == 1:
            return [(leaf.word, (path[0].level, path[0].tag, 'B-' + leaf.tag))]
        else:
            return list(zip(leaf.word,
                            [(path[-1].level, path[-1].tag, 'B-' + leaf.tag)] +
                            [(path[-1].level, path[-1].tag, 'I-' + leaf.tag)] * (len(leaf.word) - 2) +
                            [(path[0].level, path[0].tag, 'I-' + leaf.tag)]))

    return itertools.chain.from_iterable(
        [_to_char(leaves[i], leaves[i + 1], _pair(leaves[i], leaves[i + 1])) for i in range(len(leaves) - 1)] + [_last(leaves[-1])])


def absolute_to_tree(tokens, scales):
    leaves = [Node(unary_tag, -1, word=token, nt=level, ct=tag, ut=unary_tag)
              for token, (level, tag, unary_tag) in zip(tokens[:-1], scales)] \
             + [Node(None, -1, word=tokens[-1], nt=0, ct='$', ut=None)]

    def _one_pass(lower):
        upper = []

        begin = 0
        while begin < len(lower):
            end = begin + 1
            while end < len(lower) and lower[end - 1].nt == lower[end].nt:
                end = end + 1
            if end == len(lower):
                if end - begin > 1:
                    node = Node(lower[begin].ct, lower[begin].nt, children=lower[begin:end],
                                nt=lower[end - 1].nt, ct=lower[end - 1].ct, ut=None)
                    upper.append(node)
                else:
                    upper.extend(lower[begin:end])
            elif lower[begin].nt > lower[end].nt:
                node = Node(lower[begin].ct, lower[begin].nt, children=lower[begin:end + 1],
                            nt=lower[end].nt, ct=lower[end].ct, ut=None)
                upper.append(node)
                end = end + 1
            else:
                upper.extend(lower[begin:end])

            begin = end

        return upper

    def _fix(node):
        if not node.is_leaf():
            for child in node.children:
                child.level = node.level + 1
                _fix(child)

        return node

    while len(leaves) > 1:
        # print([str(l) for l in iterate_leaves])
        leaves = _one_pass(leaves)
        # print([str(l) for l in iterate_leaves])

    return _fix(leaves[0])


def relative_scale(absolute_scales):
    return [
        (absolute_scales[i - 1][0] - absolute_scales[i][0], *absolute_scales[i][1:]) if i > 0 else absolute_scales[i]
        for i in range(len(absolute_scales))]


def dynamic_scale(absolute_scales, relative_scales):
    return [((a[0], 'a') if r[0] < 0 else (r[0], 'r'), *a[1:])
            for a, r in zip(absolute_scales, relative_scales)]


def read_sentence(files: List[str]):

    vocab = Counter()
    counter = Counter()
    children_counter = Counter()

    rule_counter = defaultdict(Counter)

    def _tokenize(file):
        with open(file) as input:
            for line in input:
                print(line)
                line = line.strip()
                if line.startswith('<'):
                    continue

                for token in line.replace('(', ' ( ').replace(')', ' ) ').split():
                    yield token

    def _is_prune_node(child):
        return not child.is_leaf() and child.tag in {
            'IP', 'CP', 'FRAG', 'FLR', 'DNP', 'DVP', 'PRN', 'UCP', 'CLP', 'LCP', 'DFL', 'INC', 'SKIP', 'TYPO', 'OTH', 'WHPP', 'INTJ'}

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
                    results.append(Node('NP', children[begin].level-1, children=children[begin:end]))

                begin = end

        return results

    def _insert_NP(tag, children: List[Node]):
        if tag == 'NP':
            return children

        results = [Node('NP', child.level-1, children=[child]) if child.is_leaf() and child.tag in {'NN', 'NR'} else child
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

        if not(tag == '-NONE-'):
            tag = re.split('[-=]', tag)[0]

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
                    elif _is_prune_node(child) or _is_prune_vp(child) or _is_prune_np(child):
                        children.extend(child.children)
                    #elif _is_prune_vp():

                    else:
                        children.append(child)

                second = next(tokens)
            assert second == ')'

            if len(children) == 0:
                return None
                '''          
                elif len(children) == 1:  # and not children[0].is_leaf():
                    # 合并unary chain结点
                    # if not children[0].tag.startswith(tag) and (len(tag) >= len(children) or children[len(tag)] != ':'):
                    #    children[0].tag = '%s:%s' % (tag, children[0].tag)
    
                    # 如果唯一的子结点是叶结点，丢掉叶结点的pos信息
                    # if children[0].is_leaf():
                    #    children[0].tag = tag
                    return children[0]
                '''
            else:
                children = _merge_NT(tag, children)
                children = _insert_NP(tag, children)
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

    def _fix_level(node):

        if node .is_leaf():
            node.level = 0
        else:
            rule_counter[node.tag][' '.join(t for t, g in itertools.groupby(c.tag for c in node.children))] += 1
            max_level = 0
            for child in node.children:
                _fix_level(child)
                max_level = max(max_level, child.level)

            node.level = max_level + 1

        return node

    def _make_sentence(tokens):

        left = next(tokens)
        while left == '(':
            left2 = next(tokens)
            assert left2 == '('
            root = _fix_level(_make_node(tokens, 1))
            root.tag = 'S'
            yield root

            right = next(tokens)
            assert right == ')'

            left = next(tokens)

    for file in files:
        yield from _make_sentence(_tokenize(file))


    print('\t'.join(['%s: %d %f' % (k, v, float(children_counter[k])/v) for k, v in counter.items()]))

    print('vocab size = %d' % len(vocab))
    print(vocab.most_common())
    for tag, counters in rule_counter.items():
        print(tag, counters.most_common())

'''
if __name__ == '__main__':
    for sentence in read_sentence([sys.argv[1]]):
        print()
        print(sentence)
        iterate_leaves = list(sentence.path_to_leaves([]))
        absolute_scales = list(absolute_scale(iterate_leaves))
        print(absolute_scales)
        print(absolute_to_tree([c for leaf, path in iterate_leaves for c in leaf.word], absolute_scales))
        relative_scales = relative_scale(absolute_scales)
        dynamic_scales = dynamic_scale(absolute_scales, relative_scales)

        print(relative_scales)
        print(dynamic_scales)
'''


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
    nt, ct, ut = Counter(), Counter(), Counter()
    with open(out_path, 'w') as out:
        for sentence in tqdm(read_sentence([join(ctb_root, file) for file in files])):
            print(sentence)
            leaves = list(sentence.path_to_leaves([]))
            absolute_scales = list(absolute_scale(leaves))
            print(absolute_scales)
            # relative_scales = relative_scale(absolute_scales)
            # dynamic_scales = dynamic_scale(absolute_scales, relative_scales)
            # print(absolute_scales)
            nt.update(set([n for token, (n, _, _) in absolute_scales]))
            ct.update([c for token, (_, c, _) in absolute_scales])
            ut.update([u for token, (_, _, u) in absolute_scales])

            # relative_scales = relative_scale(absolute_scales)
            # dynamic_scales = dynamic_scale(absolute_scales, relative_scales)

            # print(relative_scales)
            # print(dynamic_scales)
            out.write(''.join(token for token, _ in absolute_scales) + '\n')
            out.write(' '.join(str(n) for token, (n, c, u) in absolute_scales) + '\n')
            out.write(' '.join(str(c) for token, (n, c, u) in absolute_scales) + '\n')
            out.write(' '.join(str(u) for token, (n, c, u) in absolute_scales) + '\n')
    return nt, ct, ut


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
    nt, ct, ut = ctb_to_linear(training, args.ctb, join(root_path, 'train.txt'))

    ctb_to_linear(development, args.ctb, join(root_path, 'dev.txt'))
    ctb_to_linear(test, args.ctb, join(root_path, 'test.txt'))

    print()
    print(nt.most_common())
    print(ct.most_common())
    print(ut.most_common())
