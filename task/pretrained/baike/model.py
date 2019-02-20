#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import random
from collections import defaultdict
from typing import List, Dict, Tuple

import torch
from torch import nn
from torchtext import data
from torchtext.vocab import Vocab

from .base import MASK_TOKEN
from .classifier import LMClassifier
from .encoder import ElmoEncoder


class Model(nn.Module):
    def __init__(self,
                 text_voc: Vocab,
                 embedding: nn.Embedding,
                 encoder: ElmoEncoder,
                 lm_classifier: LMClassifier,
                 label_classifiers: nn.ModuleList):
        super(Model, self).__init__()

        self.text_voc = text_voc
        self.mask_token_id = self.text_voc.stoi[MASK_TOKEN]
        self.embedding = embedding
        self.encoder = encoder

        self.lm_classifier = lm_classifier
        self.label_classifiers = label_classifiers

    def forward(self, data: data.Batch) -> Tuple[Dict[str, torch.Tensor], int]:
        text, lens = data.text

        forward_h, backward_h = self.encoder(self.embedding(text), lens)

        losses = {}
        for classifier in self.label_classifiers:
            labels = getattr(data, classifier.name)
            losses[classifier.name] = classifier(forward_h, backward_h, labels)['loss']

        if self.lm_classifier is not None:
            losses[self.lm_classifier.name] = self.lm_classifier(forward_h, backward_h, text, lens)['loss']

        return losses, lens.size(0)

    def named_embeddings(self):
        if isinstance(self.embedding, nn.Embedding):
            yield 'voc', self.embedding.weight, self.text_voc.itos
        for classifier in self.label_classifiers:
            yield classifier.named_embedding()

    def predict(self, data: data.Batch) -> Tuple[Dict[str, List], int]:
        text, lens = data.text

        forward_h, backward_h = self.encoder(self.embedding(text), lens)

        results = defaultdict(list)
        for classifier in self.label_classifiers:
            labels = getattr(data, classifier.name)
            results.update(classifier.predict(forward_h, backward_h, labels))

        return results, lens.size(0)

