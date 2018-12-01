#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import gzip
from collections import Counter, defaultdict, OrderedDict
from typing import List, Dict, Tuple, Set

import torch
from torch import nn
from torchtext import data
from torchtext.data import Dataset
from tqdm import tqdm

from .base import Label
from .encoder import LSTMEncoder
from .classifier import LabelClassifier, PhraseClassifier
from .preprocess import PhraseLabel


class Model(nn.Module):
    def __init__(self,
                 embedding: nn.Embedding,
                 encoder: LSTMEncoder,
                 label_classifiers: nn.ModuleList,
                 phrase_classifier: PhraseClassifier):
        super(Model, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.label_classifiers = label_classifiers
        self.phrase_classifier = phrase_classifier


    def forward(self,
                data: data.Batch) -> Tuple[Dict[str, torch.Tensor], int]:
        text, lens = data.text
        masks = self._make_masks(text, lens)
        hidden = self.encoder(self.embedding(text), masks)

        losses = {}
        for classifier in self.label_classifiers:
            labels = getattr(data, classifier.name)
            loss = classifier(hidden, masks, labels)
            losses[classifier.name] = loss

        phrases = self._collect_phrase(data)
        losses['phrase'] = self.phrase_classifier(hidden, masks, phrases)
        return losses, lens.size(0)

    def predict(self,
                data: data.Batch) -> Tuple[Dict[str, List], List[List[Tuple[int, int, float, float]]], int]:
        text, lens = data.text
        masks = self._make_masks(text, lens)
        hidden = self.encoder(self.embedding(text), masks)

        results = defaultdict(list)
        for classifier in self.label_classifiers:
            labels = getattr(data, classifier.name)
            res = classifier.predict(hidden, masks, labels)
            results[classifier.name].extend(res)

        phrases = self._collect_phrase(data)

        return results, self.phrase_classifier.predict(hidden, masks, phrases), lens.size(0)

    def list_phrase(self, data: data.Batch):
        text, lens = data.text
        masks = self._make_masks(text, lens)
        hidden = self.encoder(self.embedding(text), masks)
        phrases = self.phrase_classifier.find_phrase(hidden, masks)

        return phrases

    def _collect_phrase(self,
                        data: data.Batch):
        text, lens = data.text
        phrases = [set() for _ in range(lens.size(0))]
        for classifier in self.label_classifiers:
            labels = getattr(data, classifier.name)
            for bid, slabels in enumerate(labels):
                for label in slabels:
                    phrases[bid].add((label.begin, label.end))
        phrases = [list(s) for s in phrases]
        return phrases

    def _make_masks(self,
                    sens: torch.Tensor,
                    lens: torch.Tensor) -> torch.Tensor:
        masks = torch.ones(sens.size(), dtype=torch.uint8, device=sens.device)
        for i, l in enumerate(lens):
            masks[l:, i] = 0
        return masks

if __name__ == '__main__':

    train = BaikeDataset.iters(path='./baike', train='entity.pre.gz')

    for it in train:
        print(it)
        break
