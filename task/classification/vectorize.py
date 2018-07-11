#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import numpy as np
from pyhanlp import HanLP

def normalize(matrix):
    norm = np.sqrt(np.sum(matrix * matrix, axis=1))
    matrix = matrix / norm[:, np.newaxis]
    return matrix

class BOWVectorizer:
    def __init__(self, path, mode='average'):
        self.matrix, self.id2word, self.word2id, self.word_dim = self.read_vectors(path, 0)

        self.mode = mode
        assert mode in ['average', 'max', 'concat']
        if mode == 'average':
            self.text_dim = self.word_dim
        elif mode == 'max':
            self.text_dim = self.word_dim
        elif mode == 'concat':
            self.text_dim = self.word_dim*2

    def text_feature(self, text: str):
        words = [self.matrix[self.word2id[term.word]]
                 for term in HanLP.segment(text) if term.word in self.word2id]
        if self.mode == 'average':
            return np.mean(words, axis=0) if len(words) > 0 else np.zeros(shape=self.word_dim, dtype=np.float32)
        elif self.mode == 'max':
            return np.max(words, axis=0) if len(words) > 0 else np.zeros(shape=self.word_dim, dtype=np.float32)
        elif self.mode == 'concat':
            return np.concatenate((np.mean(words, axis=0), np.max(words, axis=0))) if len(words) > 0 \
                else np.zeros(shape=self.word_dim * 2, dtype=np.float32)

    def feature(self, data):
        if isinstance(data, str):
            return self.text_feature(data)
        elif isinstance(data, list):
            return np.mean([self.feature(i) for i in data], axis=0) if len(data) > 0 \
                else np.zeros(shape=self.text_dim, dtype=np.float32)
        elif isinstance(data, dict):
            return np.concatenate((self.feature(list(data.keys())), self.feature(list(data.values())))) if len(data) > 0 \
                else np.zeros(shape=self.text_dim * 2, dtype=np.float32)
        else:
            return np.zeros(shape=self.text_dim, dtype=np.float32)


    @staticmethod
    def read_vectors(path, topn):  # read top n word vectors, i.e. top is 10000
        lines_num, dim = 0, 0
        vectors = {}
        iw = []
        wi = {}
        with open(path, encoding='utf-8', errors='ignore') as f:
            first_line = True
            for line in f:
                if first_line:
                    first_line = False
                    dim = int(line.rstrip().split()[1])
                    continue
                lines_num += 1
                tokens = line.rstrip().split(' ')
                vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
                iw.append(tokens[0])
                if topn != 0 and lines_num >= topn:
                    break
        for i, w in enumerate(iw):
            wi[w] = i

        # Turn vectors into numpy format and normalize them
        matrix = np.zeros(shape=(len(iw), dim), dtype=np.float32)
        for i, word in enumerate(iw):
            matrix[i, :] = vectors[word]
        matrix = normalize(matrix)

        return matrix, iw, wi, dim