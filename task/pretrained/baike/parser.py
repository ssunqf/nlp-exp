#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import re
import argparse
import json
from os import listdir, makedirs
from os.path import isfile, join, isdir, exists
from typing import Dict, List, Tuple, Union, Set
import torch
from torch import nn
from itertools import chain
from collections import Counter
from tqdm import tqdm
from torchtext.data import Dataset, Field, RawField

from .base import make_masks
from .attention import MultiHeadedAttention

UNK = '<UNK>'
START = '<S>'
END = '</S>'
PAD = '<PAD>'
DUMMY = '<DUMMY>'


class AbstractNode:
    def __init__(self, left, right, label):
        self.left = left
        self.right = right
        self.label = label


class Leaf(AbstractNode):
    def __init__(self, left, label, text):
        self.text = text
        super(Leaf, self).__init__(left, left+len(self.text), label)

    def iterate_leaves(self):
        yield self

    def to_char(self):
        tokens = list(self.text)
        labels = ['B-%s' % self.label] + ['I-%s' % self.label] * (len(tokens) - 1)
        return [Leaf(self.left+offset, label, token) for offset, (token, label) in enumerate(zip(tokens, labels))]

    def __str__(self):
        return '(%s %s)' % (self.label, self.text)


class Node(AbstractNode):
    def __init__(self, label: str, functions: Set[str], children: List[AbstractNode]):
        self.functions = functions
        self.children = children
        assert len(children) > 0
        left = children[0].left
        right = children[-1].right
        super(Node, self).__init__(left, right, label)

    def iterate_leaves(self):
        for child in self.children:
            yield from child.iterate_leaves()

    def iterate_no_leaves(self):
        yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.iterate_no_leaves()

    def iterate_nodes(self, only_leaf=False):
        if not only_leaf:
            yield self
        for child in self.children:
            if isinstance(child, Leaf):
                yield child
            else:
                yield from child.iterate_nodes()

    def compress(self):
        sublabels = [self.label]
        children = self.children
        while len(children) == 1 and isinstance(children[0], Node):
            if sublabels[-1] != children[0].label:
                sublabels.append(children[0].label)
            children = children[0].children

        if len(sublabels) > 1:
            print(self)
            self.label = sublabels
            self.children = children
            print(self)
        self.label = sublabels
        self.functions = []
        self.children = children

        for child in self.children:
            if isinstance(child, Node):
                child.compress()

        return self

    def to_char(self):
        children = []
        for child in self.children:
            if isinstance(child, Leaf):
                children.extend(child.to_char())
            else:
                children.append(child.to_char())

        return Node(self.label, self.functions, children)

    def get_text(self):
        for child in self.children:
            if isinstance(child, Leaf):
                yield child.text
            else:
                yield from child.get_text()

    def __str__(self):
        return '(%s%s %s)' % (self.label, '-'.join([''] + list(self.functions)), ' '.join(str(child) for child in self.children))


def make_chart(root: Node):
    tokens = [tok for span in root.get_text() for tok in span]

    phrases = {}
    for node in root.iterate_nodes():
        phrases[node.left, node.right] = node.label

    chart = []
    for slen in range(1, len(tokens) + 1):
        for begin in range(0, len(tokens)):
            chart.append(phrases.get((begin, begin+slen), 0))
    print(tokens)
    print(json.dumps(phrases))
    return chart


def read_sentence(files: List[str]):
    def _tokenize(file):
        with open(file) as input:
            for line in input:
                line = line.strip()
                if line.startswith('<'):
                    continue

                for token in line.replace('(', ' ( ').replace(')', ' ) ').split():
                    yield token

    def _make_node(tokens, left=0):
        # 只保留第一个类型
        label = next(tokens)

        if not (label == '-NONE-'):
            label, *functions = re.split('[-=]', label)

        if label in {'VCD', 'VCP', 'VNV', 'VPT', 'VRD', 'VSB'}:
            label = 'VP'

        second = next(tokens)

        if second == '(':
            children = []
            while second == '(':
                child = _make_node(tokens, left)
                if child:
                    left = child.right
                    children.append(child)
                second = next(tokens)
            assert second == ')'

            if len(children) == 0:
                return None
            else:
                return Node(label, functions=set(functions), children=children)
        else:
            third = next(tokens)
            assert third == ')'

            def _is_filterd(label, word):
                empty = re.compile('^(\*OP\*|\*pro\*|\*PRO\*|\*PNR\*|\*RNR\*|\*T\*|\*？\*|\*-[0-9]+)')
                return label in {'-NONE-'} or empty.search(word)

            return Leaf(left, label, text=second) if not _is_filterd(label, second) else None

    def _make_sentence(tokens):

        left = next(tokens)
        while left == '(':
            left2 = next(tokens)
            assert left2 == '('
            root = _make_node(tokens, 0)
            root.tag = 'S'
            yield root

            right = next(tokens)
            assert right == ')'

            left = next(tokens)

    for file in files:
        yield from _make_sentence(_tokenize(file))

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


class SpanClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(SpanClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_labels = num_labels

        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.num_labels, self.num_labels)
        )

    def forward(self,
                hiddens: torch.Tensor,
                spans: List[Tuple[int, int]]) -> Tuple[List[Dict[Tuple[int, int], int]], List[torch.Tensor]]:

        features = []
        for left, right in spans:
            features.append(torch.cat(
                (hiddens[right, :self.hidden_size//2] - hiddens[left, :self.hidden_size//2],
                 hiddens[left+1, self.hidden_size//2] - hiddens[right+1, self.hidden_size//2:]),
                dim=-1
            ))

        features = torch.stack(features, dim=0)
        scores = self.scorer(features)
        return scores


class ParseLeaf:
    def __init__(self, left, right, label):
        self.left = left
        self.right = right
        self.label = label


class ParseNode:
    def __init__(self, label, children):
        self.label = label
        self.children = children

        self.left = children[0].left
        self.right = children[-1].right


class ChartParser(nn.Module):
    def __init__(self,
                 embedding: nn.Embedding,
                 encoder: nn.Module,
                 attention: MultiHeadedAttention,
                 scorer: SpanClassifier):
        super(ChartParser, self).__init__()

        self.embedding = embedding
        self.encoder = encoder
        self.attention = attention
        self.scorer = SpanClassifier
        self.loss = nn.MultiLabelMarginLoss(reduction='sum')

    def forward(self, sentences, lens, trees=None):

        assert self.training and trees

        embed = self.embedding(sentences)
        hidden = self.encoder(embed)
        hidden, _ = self.attention(hidden, hidden, hidden, mask=make_masks(hidden, lens))

        batch_best_trees, batch_best_scores = [], []
        for bid, seq_len in enumerate(lens):
            label_scores = self.label_score(hidden[:, bid], seq_len)
            if self.training:
                oracle_score = self.oracle_parse(label_scores, trees[bid])
                best_tree, best_score = self.best_parse(label_scores)
                best_score = best_score - oracle_score
            else:
                best_tree, best_score = self.best_parse(label_scores)

            batch_best_trees.append(best_tree)
            batch_best_scores.append(best_score)

        return batch_best_trees, batch_best_scores

    def label_score(self, hidden, seq_len):

        spans = [[(left, left+span_len) for left in range(0, seq_len-span_len)] for span_len in range(1, seq_len)]
        scores = self.scorer(hidden, list(chain.from_iterable(spans)))
        scores = scores.split([len(_spans) for _spans in spans])

        return scores

    def oracle_parse(self, label_scores, tree) -> torch.Tensor:
        oracle_score = 0
        for node in tree.iterate_nodes():
            oracle_score += label_scores[node.right-node.left][node.left, node.label]
        return oracle_score

    def best_parse(self, label_scores, tree=None) -> Tuple[ParseNode, torch.Tensor]:

        seq_len = len(label_scores)

        if self.training:
            assert tree is not None

            label_scores = [_scores + 1 for _scores in label_scores]
            for node in tree.iterate_nodes():
                label_scores[node.right-node.left][node.left, node.label] -= 1

        chart = {}

        argmax_labels = [len_scores.max(-1) for len_scores in label_scores]
        for span_len, argmax_label_scores, argmax_label_indexes in enumerate(argmax_labels, start=1):
            if span_len == 1:
                for left in range(0, len(seq_len-span_len)):
                    chart[left, left+span_len] = [ParseLeaf(left, left+span_len, argmax_label_indexes[left])], argmax_label_scores[left]
            else:
                splits = []
                for split_offset in range(1, span_len):
                    splits.append((argmax_label_scores[split_offset][:seq_len-span_len-split_offset] + argmax_label_scores[span_len-split_offset][split_offset:]))

                argmax_split_scores, argmax_split_offsets = torch.cat(splits, dim=-1).argmax(-1)

                for left in range(0, len(seq_len-span_len)):
                    left_trees, left_score = chart[left, left+argmax_split_offsets[left]]
                    right_trees, right_score = chart[left+argmax_split_offsets[left], left+span_len]
                    children = left_trees + right_trees
                    if argmax_label_indexes[left] > 0:  # is phrase
                        chart[left, left+span_len] = ([ParseNode(argmax_label_indexes[left], children)],
                                                      argmax_label_scores[left] + left_score + right_score)
                    else:
                        chart[left, left+span_len] = (children, argmax_label_scores[left] + left_score + right_score)

        return chart[0, seq_len]

'''
class Vocab:
    def __init__(self, counter: Counter, specials, min_count=1):

        self.itos = specials + [w for w, c in counter.most_common() if c >= min_count]
        self.stoi = dict((s, i) for i, s in enumerate(self.itos))
        self.unk_id = self.stoi[UNK]

    def __len__(self):
        return len(self.itos)

    def get_word(self, index: Union[int, List[int]]):
        if isinstance(index, int):
            assert 0 <= index < len(self.itos)
            return self.itos[index]
        else:
            return [self.itos[id] for id in index]

    def get_index(self, word: Union[str, List[str]]):
        if isinstance(word, str):
            return self.stoi.get(word, self.unk_id)
        else:
            return [self.stoi.get(w, self.unk_id) for w in word]

class LabelField(RawField):
    def __init__(self,):

    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.

        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)


class CTBParseDataset(Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, files, fields, **kwargs):
        trees = [sentence.to_char().compress() for sentence in tqdm(files, desc='loading train data')]

        trees = [(list(tree.get_text()), tree) for tree in trees]
        super(CTBParseDataset, self).__init__(trees, fields, **kwargs)

    @classmethod
    def splits(cls, path=None, root='.data', train=None, validation=None,
               test=None, **kwargs):

        def split_files(ctb_root):
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

        train, valid, test = split_files(path)

        train = cls([join(path, file) for file in train], **kwargs)
        valid = cls([join(path, file) for file in valid], **kwargs)
        test = cls([join(path, file) for file in test], **kwargs)

        return tuple(d for d in (train, valid, test) if d is not None)


def build_parser(ctb_root,
                 embed_dim, hidden_dim, num_heads):

    text_field = Field(init_token=START, eos_token=END, unk_token=UNK, pad_token=PAD)
    label_field = Field(unk_token=DUMMY)

    train, valid, test = CTBParseDataset.splits(path=ctb_root,
                                                fields=[('text', text_field), ('label', label_field)])

    text_field.build_vocab(train)
    label_field.build_vocab(train)



    embedding = nn.Embedding(len(text_voc), embed_dim)
    encoder = nn.LSTM(embed_dim, hidden_dim//2)
    attention = MultiHeadedAttention(num_heads, hidden_dim)
    scorer = SpanScorer(hidden_dim, len(label_voc))

    parser = ChartParser(embedding, encoder, attention, scorer)

    return text_voc, label_voc, train_data, valid_data, test_data, parser
'''

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

    training = [make_chart(sentence) for sentence in tqdm(
        read_sentence([join(args.ctb, file) for file in training]),
        desc='loading train data')]
    development = [sentence.to_char() for sentence in tqdm(
        read_sentence([join(args.ctb, file) for file in development]),
        desc='loading dev data')]
    test = [sentence.to_char() for sentence in tqdm(
        read_sentence([join(args.ctb, file) for file in test]),
        desc='loading test data')]


    with open(join(root_path, 'train.txt'), 'w') as writer:
        writer.writelines(str(sentence) + '\n' for sentence in training)

    with open(join(root_path, 'valid.txt'), 'w') as writer:
        writer.writelines(str(sentence) + '\n' for sentence in development)

    with open(join(root_path, 'test.txt'), 'w') as writer:
        writer.writelines(str(sentence) + '\n' for sentence in test)

    text_counter = Counter(c for sentence in training for leaf in sentence.iterate_leaves() for c in leaf.text)
    label_counter = Counter('-'.join(node.label) for sentence in training for node in sentence.iterate_no_leaves())



