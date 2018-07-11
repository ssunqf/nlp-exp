#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-


class Config:
    def __init__(self):
        self.dir = 'concat/'

        self.train = self.dir + 'train.gz'
        self.valid = self.dir + 'valid.gz'
        self.test = self.dir + 'test.gz'
        self.label_dict = self.dir + 'labels'

        self.word_embed_path = self.dir + 'sgns.baidubaike.bigram-char'
        self.word_embed_dim = 300
        self.mode = 'concat'

        self.model_prefix = self.dir + 'model'

        self.summary_dir = self.dir + 'summary'

config = Config()