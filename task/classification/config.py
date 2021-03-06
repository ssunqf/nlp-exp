#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import torch


class Config:
    def __init__(self):
        self.dir = 'max/'

        self.train = self.dir + 'train.gz'
        self.valid = self.dir + 'valid.gz'
        self.test = self.dir + 'test.gz'
        self.label_dict = self.dir + 'labels'

        self.word_embed_path = self.dir + 'sgns.baidubaike.bigram-char.gz'
        self.word_embed_dim = 300
        self.mode = 'concat'

        self.use_cuda = torch.cuda.is_available()

        self.model_prefix = self.dir + 'model'

        self.summary_dir = self.dir + 'summary'

config = Config()
