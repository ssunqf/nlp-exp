#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import itertools
import copy

class Node:
    def __init__(self, index, chars, pos=None, head_index=-1, relation=None, lefts=None, rights=None):
        self.index = index
        self.chars = chars if chars else []
        self.pos = pos
        self.head_index = head_index
        self.relation = relation
        self.lefts = lefts if lefts else []
        self.rights = rights if rights else []

    def __len__(self):
        return len(self.chars)

    def __str__(self):
        return '(%s\t#%s#\t%s)' % ('\t'.join([str(left) for left in self.lefts]),
                                 ''.join(self.chars),
                                 '\t'.join([str(right) for right in self.rights]))

    def __deepcopy__(self, memodict={}):

        index = copy.copy(self.index)
        chars = copy.copy(self.chars)
        pos = copy.copy(self.pos)
        head_index = copy.copy(self.head_index)
        relation = copy.copy(self.relation)
        lefts = copy.deepcopy(self.lefts, memodict)
        rights = copy.deepcopy(self.rights, memodict)
        new_node = Node(index, chars, pos, head_index, relation, lefts, rights)
        memodict[id(self)] = new_node
        return new_node


class Actions:
    SHIFT = 0
    APPEND = 1
    ARC_LEFT = 2
    ARC_RIGHT = 3
    #ROOT = 4
    max_len = 4

class Transition:
    def __init__(self, action, label):
        self.action = action
        self.label = label

    def __eq__(self, other):
        return hash(self) == hash(other) and self.action == other.action and self.label == other.label

    def __hash__(self):
        return hash(self.action) + hash(self.label)

    def __str__(self):
        return '%d,%s' % (self.action, self.label)

class UDTree:
    def __init__(self, nodes, root):
        self.nodes = nodes
        self.root = root

    def linearize(self):
        chars = list(itertools.chain.from_iterable([node.chars for node in self.nodes]))
        relations = [node.relation for node in self.nodes]
        pos = [node.pos for node in self.nodes]

        def dfs(node):
            for left in node.lefts:
                yield from dfs(left)

            yield Transition(Actions.SHIFT, node.pos)
            for c in node.chars[1:]:
                yield Transition(Actions.APPEND, None)

            for l in node.lefts[::-1]:
                yield Transition(Actions.ARC_LEFT, l.relation)

            for right in node.rights:
                yield from dfs(right)
                yield Transition(Actions.ARC_RIGHT, right.relation)

        return chars, (list(dfs(self.root)), relations, pos)

    def get_words(self):
        return [''.join(node.chars) for node in self.nodes]

    def to_line(self):
        return '\t'.join(['%s#%s#%d#%s' % (''.join(node.chars), node.pos, node.head_index, node.relation) for node in self.nodes])

    @staticmethod
    def create(chars, transitons):
        nodes = []
        stack = []
        char_index = 0
        word_index = 0
        for tran in transitons:
            if tran.action == Actions.SHIFT:
                new_node = Node(word_index, [chars[char_index]], tran.label)
                stack.append(new_node)
                nodes.append(new_node)
                char_index += 1
                word_index += 1
            elif tran.action == Actions.APPEND:
                stack[-1].chars.append(chars[char_index])
                char_index += 1
            elif tran.action == Actions.ARC_LEFT:
                head = stack.pop()
                modifier = stack.pop()
                modifier.head_index = head.index
                modifier.relation = tran.label
                head.lefts.append(modifier)
                stack.append(head)
            elif tran.action == Actions.ARC_RIGHT:
                modifier = stack.pop()
                head = stack[-1]
                head.rights.append(modifier)
                modifier.head_index = head.index
                modifier.relation = tran.label


        for node in nodes:
            node.lefts = list(reversed(node.lefts))

        stack[-1].relation = '核心成分'

        return UDTree(nodes, stack[-1])

    @staticmethod
    def parse_stanford_format(tokens):
        nodes = [Node(index, chars, pos, int(parent_id)-1, relation) for index, (chars, pos, parent_id, relation) in enumerate(tokens)]

        root_id = 0
        for id, (token, pos, parent_id, relation) in enumerate(tokens):
            parent_id = int(parent_id) - 1

            if parent_id == -1:
                root_id = id
            elif parent_id > id:
                nodes[parent_id].lefts.append(nodes[id])
            elif parent_id < id:
                nodes[parent_id].rights.append(nodes[id])

        def validate(tree):
            gold = tree.to_line()

            chars, (trans, _, _) = tree.linearize()

            new_tree = UDTree.create(chars, trans)
            new_line = new_tree.to_line()

            return gold == new_line

        tree = UDTree(nodes, nodes[root_id])

        if validate(tree) is False:
            print('invalid tree:')
            print(tree.to_line())
            return None

        return UDTree(nodes, nodes[root_id])

import argparse

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--src', type=str, required=True, help='raw data')
arg_parser.add_argument('--bad', type=str, required=True, help='conflict data')
arg_parser.add_argument('--tgt', type=str, required=True, help='formated data')

args = arg_parser.parse_args()


with open(args.src) as src, open(args.bad, 'w') as bad_file, open(args.tgt, 'w') as tgt:
    sentences = src.read().strip().split('\n\n')
    # uniq operation
    # sentences = set(sentences)
    for sentence in sentences:
        words = sentence.strip().split('\n')
        if len(words) > 0:
            words = [word.strip().split('\t') for word in words]
            words = [(word, pos, parent_id, relation)
                     for _, word, _, pos, _,  _, parent_id, relation in words]
            tree = UDTree.parse_stanford_format(words)

            if tree is None:
                bad_file.write(sentence + '\n\n')
            else:
                tgt.write(tree.to_line() + '\n')