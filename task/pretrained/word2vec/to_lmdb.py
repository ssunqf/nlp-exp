#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import sys
from gensim.models.keyedvectors import KeyedVectors
from lmdb_embeddings.writer import LmdbEmbeddingsWriter


if __name__ == '__main__':
    print('Loading gensim model...')
    gensim_model = KeyedVectors.load_word2vec_format(sys.argv[1], binary=False)


    def iter_embeddings():
        for word in gensim_model.vocab.keys():
            yield word, gensim_model[word]

    print('Writing vectors to a LMDB database...')

    writer = LmdbEmbeddingsWriter(
        iter_embeddings()
    ).write(sys.argv[2])