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
from .classifier import LMClassifier, PhraseClassifier
from .encoder import StackRNN, ElmoEncoder


class Model(nn.Module):
    def __init__(self,
                 text_voc: Vocab,
                 embedding: nn.Embedding,
                 encoder: StackRNN,
                 lm_classifier: LMClassifier,
                 label_classifiers: nn.ModuleList,
                 phrase_classifier: PhraseClassifier):
        super(Model, self).__init__()

        self.text_voc = text_voc
        self.mask_token_id = self.text_voc.stoi[MASK_TOKEN]
        self.embedding = embedding
        self.encoder = encoder

        self.lm_classifier = lm_classifier
        self.label_classifiers = label_classifiers
        self.phrase_classifier = phrase_classifier

    def forward(self, data: data.Batch) -> Tuple[Dict[str, torch.Tensor], int]:
        text, lens = data.text

        hiddens, _ = self.encoder(self.embedding(text), lens)

        losses = {}

        for classifier in self.label_classifiers:
            labels = getattr(data, classifier.name)
            losses[classifier.name] = classifier(hiddens[-1], labels)['loss']

        if self.lm_classifier is not None:
            losses[self.lm_classifier.name] = self.lm_classifier(hiddens[0], text, lens)['loss']

        if self.phrase_classifier is not None:
            results = self.phrase_classifier(
                    hiddens[-1], lens,
                    getattr(data, self.phrase_classifier.name))
            losses[self.phrase_classifier.name] = results['loss']

        return losses, lens.size(0)

    def named_embeddings(self):
        if isinstance(self.embedding, nn.Embedding):
            yield 'voc', self.embedding.weight, self.text_voc.itos
        for classifier in self.label_classifiers:
            yield classifier.named_embedding()

    def predict(self, data: data.Batch) -> Tuple[Dict[str, List], torch.Tensor, list]:
        text, lens = data.text

        hiddens, _ = self.encoder(self.embedding(text), lens)

        label_results = defaultdict(list)
        for classifier in self.label_classifiers:
            labels = getattr(data, classifier.name)
            label_results.update(classifier.predict(hiddens[-1], labels))
        lm_result = self.lm_classifier.predict(hiddens[0], lens)
        phrase_result = self.phrase_classifier.predict(
            hiddens[-1], lens,
            getattr(data, self.phrase_classifier.name))

        return label_results, lm_result, phrase_result

    def find_phrase(self, data: data.Batch):
        text, lens = data.text
        hiddens, _ = self.encoder(self.embedding(text), lens)
        return self.phrase_classifier.find_phrase(hiddens[-1], lens)

