#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import sys
from typing import List

class Node:
    def __init__(self, tag, **kwargs):
        self.tag = tag
        self.__dict__.update(kwargs)

    def is_leaf(self):
        return 'word' in self.__dict__

    def __str__(self):

        if self.is_leaf():
            return '(%s %s)' % (self.tag, self.word)
        else:
            return '(%s %s)' % (self.tag, ' '.join([str(n) for n in self.children]))


def read_sentence(files: List[str]):
    def _tokenize(file):
        with open(file) as input:
            for line in input:
                line = line.strip()
                if not line.startswith('('):
                    continue

                for token in line.replace('(', ' ( ').replace(')', ' ) ').split():
                    yield token

    def _make_node(tokens):

        tag = next(tokens)

        second = next(tokens)

        if second == '(':
            children = []
            while second == '(':
                children.append(_make_node(tokens))
                second = next(tokens)
            assert second == ')'
            return Node(tag, children=children)
        else:
            third = next(tokens)
            assert third == ')'
            return Node(tag, word=second)

    def _make_sentence(tokens):

        left = next(tokens)
        while left == '(':
            left2 = next(tokens)
            assert left2 == '('
            yield _make_node(tokens)

            right = next(tokens)
            assert right == ')'

            left = next(tokens)

    for file in files:
        yield from _make_sentence(_tokenize(file))


def prune(root: Node):
    pass


if __name__ == '__main__':

    for sentence in read_sentence([sys.argv[1]]):
        print(sentence)

