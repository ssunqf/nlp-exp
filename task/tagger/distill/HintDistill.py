#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

# knowledge distill
# https://github.com/dkozlov/awesome-knowledge-distillation

from typing import List

import torch
import torch.nn as nn
from torch.autograd import Variable
from ..train.model import Tagger
from model.nlp.task.task import Task, Loader, TaskConfig
from torchtext import data

class HintDistill(nn.Module):
    def __init__(self, teacher: Tagger, student: Tagger, hard_target_scale_factor=0.2):
        super(HintDistill, self).__init__()

        self.teacher = teacher
        self.student = student
        self.hint_loss = nn.MSELoss()
        self.hard_target_scale_factor = hard_target_scale_factor

    def _mask(self, max_length: int, batch_size: int, lengths: List[int]) -> Variable:
        mask = torch.FloatTensor(max_length, batch_size).fill_(0)
        for i, l in enumerate(lengths):
            mask[i, 0:l] = 1

        return Variable(mask)

    def forward(self, batch: data.Batch):

        teacher_feature, confidence = self.teacher.feature(batch, predict=True)

        student_feature = self.student.feature(batch, predict=False)

        _, text_len = batch.text
        max_len, batch_size, _ = teacher_feature.size()
        feature_mask = self._mask(max_len, batch_size, text_len).unsqueeze(-1)

        student_nll = self.student.crf.neg_log_likelihood(student_feature, batch.tags)
        hint_loss = self.hint_loss(teacher_feature * feature_mask, student_feature * feature_mask)

        return student_nll * self.hard_target_scale_factor + hint_loss


class DistillConfig:


    corpus_prefix = './ctb.pos'

    train = '.train'
    valid = '.valid'
    test = '.test'

    batch_sizes = [16, 16, 16]

    # vocabulary
    text_min_freq = 10
    text_min_size = 50000
    tag_min_freq = 10
    tag_min_size = 50000

    common_size = 1000

    # model
    embedding_dim = 512
    hidden_mode = 'GRU'
    hidden_dim = 512
    num_layers = 4
    dropout = 0.2

    use_cuda = torch.cuda.is_available()

    checkpoint_path = './ctb.pos.model'

    valid_step = 500
