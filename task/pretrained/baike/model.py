#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import random
from collections import defaultdict
from typing import List, Dict, Tuple

import torch
from torch import nn
from torchtext import data
from torchtext.vocab import Vocab

from task.pretrained.transformer.attention import TransformerLayer
from .base import MASK_TOKEN
from .classifier import PhraseClassifier
from .encoder import LSTMEncoder


class Model(nn.Module):
    def __init__(self,
                 text_voc: Vocab,
                 embedding: nn.Embedding,
                 encoder: LSTMEncoder,
                 transformer: TransformerLayer,
                 label_classifiers: nn.ModuleList,
                 phrase_classifier: PhraseClassifier,
                 phrase_mask_prob: float):
        super(Model, self).__init__()

        self.text_voc = text_voc
        self.mask_token_id = self.text_voc.stoi[MASK_TOKEN]
        self.embedding = embedding
        self.encoder = encoder
        self.transformer = transformer
        self.label_classifiers = label_classifiers
        self.phrase_classifier = phrase_classifier

        self.phrase_mask_prob = phrase_mask_prob

    def forward(self, data: data.Batch) -> Tuple[Dict[str, torch.Tensor], int]:
        text, lens = data.text
        phrases = self._collect_phrase(data)
        text, phrases = self._mask_phrase(text, lens, phrases)
        masks = self._make_masks(text, lens)
        hidden = self.encoder(self.embedding(text), masks)
        if self.transformer:
            hidden = self.transformer(hidden, masks, batch_first=False)

        losses = {}
        for classifier in self.label_classifiers:
            labels = getattr(data, classifier.name)
            loss = classifier(hidden, masks, labels)
            losses[classifier.name] = loss

        losses['phrase'] = self.phrase_classifier(hidden, masks, phrases)

        return losses, lens.size(0)

    def named_embeddings(self):
        yield 'voc', self.embedding.weight, self.text_voc.itos
        for classifier in self.label_classifiers:
            yield from classifier.named_embeddings()

    def predict(self, data: data.Batch) -> Tuple[Dict[str, List], List[List[Tuple[int, int, float, float]]], int]:
        text, lens = data.text
        masks = self._make_masks(text, lens)
        phrases = self._collect_phrase(data)
        text, phrases = self._mask_phrase(text, lens, phrases)

        hidden = self.encoder(self.embedding(text), masks)
        if self.transformer:
            hidden = self.transformer(hidden, masks, batch_first=False)

        results = defaultdict(list)
        for classifier in self.label_classifiers:
            labels = getattr(data, classifier.name)
            res = classifier.predict(hidden, masks, labels)
            results[classifier.name].extend(res)

        return results, self.phrase_classifier.predict(hidden, masks, phrases), lens.size(0)

    def list_phrase(self, data: data.Batch):
        text, lens = data.text
        masks = self._make_masks(text, lens)
        hidden = self.encoder(self.embedding(text), masks)
        if self.transformer:
            hidden = self.transformer(hidden, masks, batch_first=False)
        phrases = self.phrase_classifier.find_phrase(hidden, masks)

        return phrases

    def _mask_phrase(self, text: torch.Tensor, lens: torch.Tensor,
                     phrases: List[List[Tuple[int, int]]]) -> Tuple[torch.Tensor, List[List[Tuple[int, int]]]]:
        no_mask_phrases = []
        for bid, sphrases in enumerate(phrases):
            sen_phrases = []
            seq_len = lens[bid].item()
            for begin, end in sphrases:
                if random.random() < self.phrase_mask_prob and seq_len > 7 and (end - begin) < seq_len - 2:
                    text[begin:end, bid] = self.mask_token_id
                else:
                    sen_phrases.append((begin, end))

            no_mask_phrases.append(sen_phrases)

        return text, no_mask_phrases

    def _collect_phrase(self, data: data.Batch):
        text, lens = data.text
        phrases = [set() for _ in range(lens.size(0))]
        for classifier in self.label_classifiers:
            labels = getattr(data, classifier.name)
            for bid, slabels in enumerate(labels):
                for label in slabels:
                    phrases[bid].add((label.begin, label.end))
        phrases = [list(s) for s in phrases]
        return phrases

    def _make_masks(self, sens: torch.Tensor,
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
