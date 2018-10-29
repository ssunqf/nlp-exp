#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
from .base import hash, hash_next
from typing import List, Set
from task.util import utils
from tqdm import tqdm
import gzip

def make_hash(emb_path):
    vocab = set()
    max_tok_len = 0
    max_word = None
    with gzip.open(emb_path, mode='rt', compresslevel=6) as file:
        word_size, dim = [int(i) for i in file.readline().rstrip().split()]
        for line in tqdm(file):
            head, *_ = line.split()
            if ' ' in head.strip():
                print(head)
                continue
            hash_id, tlen = hash(head)
            vocab.add(hash_id)
            if tlen > max_tok_len:
                max_tok_len = tlen
                max_word = head

    print('word vector size = ', len(vocab))
    print('{}, token_len={}'.format([w for t, w in utils.replace_entity(max_word)], max_tok_len))
    return vocab, max_tok_len


def filter(dataset: List[str], vocab: Set[str], max_tok_len):
    small = set()
    for path in dataset:
        with open(path) as file:
            for line in file:
                items = line.strip().split()
                if 0 < len(items) < 150:
                    items = [t.rsplit('#', maxsplit=1) for t in items]
                    tokens = [t[0] for t in items]
                    tags = [t[1] for t in items]

                    for s in range(len(tokens)):
                        hash_id = None
                        for l in range(min(max_tok_len, len(tokens) - s)):
                            hash_id = hash_next(hash_id, tokens[s+l])
                            if hash_id in vocab:
                                small.add(hash_id)
    return small


def extract(big_emb:str, small_emb: str, small: Set[int]):
    with gzip.open(big_emb, mode='rt', compresslevel=6) as reader,\
            gzip.open(small_emb, mode='wt', compresslevel=6) as writer:
        word_size, dim = [int(i) for i in reader.readline().rstrip().split()]
        writer.write('{} {}\n'.format(len(small), dim))
        for line in tqdm(reader):
            word, weights = line.rstrip().split(maxsplit=1)
            hash_id, tlen = hash(word)
            if hash_id in small:
                writer.write('{} {}\n'.format(word, weights))


if __name__ == '__main__':

    big_path = 'wordvec/Tencent_AILab_ChineseEmbedding.txt.gz'
    small_path = 'wordvec/Tencent_AILab_ChineseEmbedding.small.txt.gz'
    big_vocab, max_len = make_hash(big_path)
    small_vocab = filter(['./tagger-data/ctb.std/train', './tagger-data/ctb.std/valid', './tagger-data/ctb.std/test'], big_vocab, max_len)

    extract(big_path, small_path, small_vocab)
