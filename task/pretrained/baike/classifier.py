#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
from typing import Tuple, List, Set, Dict
import math
import random

import torch
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import Vocab

from task.pretrained.transformer.attention import MultiHeadedAttention

from .base import Label


class SoftmaxLoss(nn.Module):
    def __init__(self):
        super(SoftmaxLoss, self).__init__()

        # https://arxiv.org/pdf/1705.07115.pdf
        self.inv_temperature = nn.Parameter(torch.tensor([1.0], dtype=torch.float))

    def forward(self, logit: torch.Tensor, targets: List[torch.Tensor]) -> torch.Tensor:
        log_probs = F.log_softmax(logit * self.inv_temperature, dim=-1)

        loss = sum(-log_probs[bid].gather(-1, target).sum() for bid, target in enumerate(targets))

        count = sum(t.size(0) for t in targets)

        return loss / (count + 1e-5)

    def predict(self, logit: torch.Tensor, topk=5) -> Tuple[torch.Tensor, torch.Tensor]:
        return F.softmax(logit * self.inv_temperature, dim=-1).topk(topk, dim=-1)


class ContextClassifier(nn.Module):
    def __init__(self,
                 name: str,
                 voc: Vocab,
                 hidden_dim, label_dim,
                 dropout=0.3):
        super(ContextClassifier, self).__init__()
        self.name = name
        self.voc = voc
        self.hidden_dim = hidden_dim
        self.label_dim = label_dim

        self.hidden2label = nn.Linear(label_dim, len(voc), bias=True)

        self.context2ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, self.label_dim),
            nn.Tanh(),
        )

        self.phrase2ffn = nn.Sequential(
            nn.Linear(hidden_dim * 4, self.label_dim),
            nn.Tanh(),
        )

        self.loss = SoftmaxLoss()

    def forward(self,
                forwards: torch.Tensor,
                backwards: torch.Tensor,
                labels: List[List[Label]]) -> Dict[str, torch.Tensor]:

        device = forwards.device
        context_features, context_tags = [], []
        phrase_features, phrase_tags = [], []

        for bid, sen_labels in enumerate(labels):
            for label in sen_labels:
                if label.tags.size(0) > 0:
                    context = self._context_feature(forwards, backwards, bid, label.begin, label.end)
                    context_features.append(context)
                    context_tags.append(label.tags)

                    phrase = self._phrase_feature(forwards, backwards, bid, label.begin, label.end)
                    phrase_features.append(phrase)
                    phrase_tags.append(label.tags)

        if len(context_tags) > 0:
            context_features = self.hidden2label(self.context2ffn(torch.stack(context_features, dim=0)))
            phrase_features = self.hidden2label(self.phrase2ffn(torch.stack(phrase_features, dim=0)))
            return {
                'loss': (self.loss(context_features, context_tags)
                         + self.loss(phrase_features, phrase_tags)) / 2
            }
        else:
            return {
                'loss': torch.tensor([0.0], device=forwards.device)
            }

    def predict(self,
                forwards: torch.Tensor, backwards: torch.Tensor,
                labels: List[List[Label]]) -> Dict[str, Dict[str, List[List[Tuple[Label, torch.Tensor]]]]]:
        device = forwards.device
        context_features, context_tags = [], []
        phrase_features, phrase_tags = [], []
        for bid, sen_labels in enumerate(labels):
            for label in sen_labels:
                if label.tags.size(0) > 0:
                    context = self._context_feature(forwards, backwards, bid, label.begin, label.end)
                    context_features.append(context)
                    context_tags.append((bid, label))

                    phrase = self._phrase_feature(forwards, backwards, bid, label.begin, label.end)
                    phrase_features.append(phrase)
                    phrase_tags.append((bid, label))

        context_results = [[] for _ in range(len(labels))]
        phrase_results = [[] for _ in range(len(labels))]
        if len(context_features) > 0:
            context_logits = self.hidden2label(self.context2ffn(torch.stack(context_features, dim=0)))
            scores, indexes = self.loss.predict(context_logits, 5)
            for (bid, label), topk in zip(context_tags, indexes):
                context_results[bid].append((label, topk.tolist()))

            phrase_logits = self.hidden2label(self.phrase2ffn(torch.stack(phrase_features, dim=0)))
            scores, indexes = self.loss.predict(phrase_logits, 5)
            for (bid, label), topk in zip(phrase_tags, indexes):
                phrase_results[bid].append((label, topk.tolist()))

        return {
            self.name: {
                'context': context_results,
                'phrase': phrase_results
            }
        }

    def named_embedding(self):
        return self.name, self.hidden2label.weight, self.voc.itos

    def _context_feature(self,
                         forwards: torch.Tensor,
                         backwards: torch.Tensor,
                         bid: int, begin: int, end: int):
        return torch.cat((forwards[begin-1, bid], backwards[end, bid]), dim=-1)

    def _phrase_feature(self,
                        forwards: torch.Tensor,
                        backwards: torch.Tensor,
                        bid: int, begin: int, end: int):
        return torch.cat((forwards[begin-1, bid], forwards[end-1, bid],
                          backwards[end, bid], backwards[begin, bid]), dim=-1)


class LMClassifier(nn.Module):
    def __init__(self,
                 voc_size: int,
                 voc_dim: int,
                 hidden_dim: int,
                 padding_idx=-1,
                 dropout=0.3):
        super(LMClassifier, self).__init__()
        self.name = 'lm'
        self.voc_size = voc_size
        self.voc_dim = voc_dim
        self.hidden_dim = hidden_dim
        self.padding_idx = padding_idx
        self.context_ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, voc_dim),
            nn.Tanh()
        )

        self.context2token = nn.Linear(voc_dim, voc_size)

        self.inv_temperature = nn.Parameter(torch.tensor([1.0], dtype=torch.float))

    def forward(self,
                forwards: torch.Tensor,
                backwards: torch.Tensor,
                tokens: torch.Tensor,
                lens: torch.Tensor) -> Dict[str, torch.Tensor]:
        seq_len, batch_size, dim = forwards.size()
        assert dim == self.hidden_dim
        middle = torch.cat((forwards[:-2], backwards[2:]), dim=-1)

        logit = self.context2token(self.context_ffn(middle)) * self.inv_temperature
        return {
            'loss': F.cross_entropy(logit.view(-1, self.voc_size),
                                    tokens[1:-1].view(-1),
                                    ignore_index=self.padding_idx),
        }

    def predict(self,
                forwards: torch.Tensor,
                backwards: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, dim = forwards.size()
        assert dim == self.hidden_dim
        middle = torch.cat((forwards[:-2], backwards[2:]), dim=-1)

        logit = self.context2token(self.context_ffn(middle)) * self.inv_temperature
        return logit.max(dim=-1)[1]
