#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import gzip
from collections import Counter, defaultdict, OrderedDict
from typing import List, Dict

import torch
from torch import nn
from torchtext import data
from torchtext.data import Dataset
from tqdm import tqdm

from .base import Label
from .encoder import LSTMEncoder
from .classifier import LabelClassifier
from .preprocess import PhraseLabel


class Model(nn.Module):
    def __init__(self,
                 embedding: nn.Embedding,
                 encoder: LSTMEncoder,
                 classifiers: nn.ModuleList):
        super(Model, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.classifiers = classifiers

    def forward(self,
                data: data.Batch) -> Dict[str, torch.Tensor]:
        text, lens = data.text
        masks = self._make_masks(text, lens)
        hidden = self.encoder(self.embedding(text), masks)

        losses = {}
        for classifier in self.classifiers:
            labels = getattr(data, classifier.name)
            loss = classifier(hidden, masks, labels)
            losses[classifier.name] = loss
        return losses

    def predict(self,
                data: data.Batch) -> Dict[str, List]:
        text, lens = data.text
        masks = self._make_masks(text, lens)
        hidden = self.encoder(self.embedding(text), masks)

        results = defaultdict(list)
        for classifier in self.classifiers:
            labels = getattr(data, classifier.name)
            res = classifier.predict(hidden, masks, labels)
            results[classifier.name].extend(res)
        return results

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
