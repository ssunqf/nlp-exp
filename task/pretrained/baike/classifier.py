#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
from typing import Tuple, List, Set, Dict
import math
import random
import itertools

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
            nn.Linear(hidden_dim * 2, self.label_dim),
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
        return torch.cat((forwards[end-1, bid] - forwards[begin-1, bid],
                          backwards[begin, bid] - backwards[end, bid], ), dim=-1)


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

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

class PhraseClassifier(nn.Module):
    def __init__(self, hidden_dim, max_length=15, dropout=0.3):
        super(PhraseClassifier, self).__init__()

        self.name = 'phrase'
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, forwards: torch.Tensor, backwards: torch.Tensor,
                lens: torch.Tensor, phrases: List[List[Tuple[int, int]]]) -> Dict[str, torch.Tensor]:

        samples, features, targets, weigths = self._make_sample(forwards, backwards, lens, phrases)
        if len(samples) == 0:
            loss = torch.tensor([0.0], device=forwards.device)
        else:
            loss = F.binary_cross_entropy(self.ffn(features), targets.unsqueeze(-1), weight=weigths.unsqueeze(-1))

        return {'loss': loss}

    def predict(self, forwards: torch.Tensor, backwards: torch.Tensor,
                lens: torch.Tensor, phrases: List[List[Tuple[int, int]]]) -> \
            Tuple[List[List[Tuple[int, int, float, float, float]]], List[List[Tuple[int, int, float]]]]:

        samples, features, targets, weights = self._make_sample(forwards, backwards, lens, phrases)
        if len(samples) == 0:
            return [[] for _ in range(len(phrases))], self.find_phrase(forwards, backwards, lens, 0.5)

        preds = self.ffn(features)

        targets = targets.tolist()
        weights = weights.tolist()
        preds = preds.squeeze().tolist()
        results = [[] for _ in range(len(phrases))]
        for id, (bid, begin, end) in enumerate(samples):
            results[bid].append((begin, end, targets[id], preds[id], weights[id]))
        return results, self.find_phrase(forwards, backwards, lens, 0.5)

    def find_phrase(self, forwards: torch.Tensor, backwards: torch.Tensor, lens: torch.Tensor,
                    threshold=0.8) -> List[List[Tuple[int, int, float]]]:
        phrases = []
        for bid in range(lens.size(0)):
            samples = []
            features = []
            for begin in range(1, lens[bid] - 1):
                for step in range(2, min(self.max_length + 1, lens[bid] - begin)):
                    samples.append((begin, begin + step))
                    features.append(self._span_embed(forwards[:, bid], backwards[:, bid], begin, begin + step))
            if len(features) > 0:
                features = torch.stack(features, dim=0)
                probs = self.ffn(features).squeeze(-1).tolist()
            else:
                probs = []

            phrases.append([(begin, end, prob) for (begin, end), prob in zip(samples, probs) if prob > threshold])

        return phrases

    def _make_sample(self,
                     forwards: torch.Tensor, backwards: torch.Tensor, lens: torch.Tensor,
                     phrases: List[List[Tuple[int, int]]]) -> Tuple[List[Tuple[int, int, int]],
                                                                    torch.Tensor, torch.Tensor, torch.Tensor]:
        device = forwards.device
        positive_samples, negative_samples, boundary_samples, internal_samples = [], [], [], []
        for bid, sen_phrases in enumerate(phrases):
            for (f_b, f_e), (s_b, s_e) in pairwise(sen_phrases):
                for mid in range(f_b+1, f_e):
                    negative_samples.append((bid, mid, s_e))

                for mid in range(s_b+1, s_e-1):
                    negative_samples.append((bid, f_b, mid))

            for begin, end in sen_phrases:
                if end - begin > 1:
                    positive_samples.append((bid, begin, end))
                    for mid in range(begin, end):
                        for n_begin, n_end in [(random.randint(max(0, begin - self.max_length), begin - 1), mid),
                                               (mid, random.randint(end + 1, min(lens[bid], end + self.max_length)))]:
                            if (0 < n_begin < begin < n_end < end) or (begin < n_begin < end < n_end < lens[bid]):
                                negative_samples.append((bid, n_begin, n_end))

                        '''
                            # noise
                            if 0 < n_begin < begin:
                                boundary_samples.append((bid, n_begin, end))
                            if end < n_end < lens[bid]:
                                boundary_samples.append((bid, begin, n_end))

                        if begin < mid < end:
                            internal_samples.append((bid, begin, mid))
                            internal_samples.append((bid, mid, end))
                        '''

        samples = positive_samples + negative_samples #+ boundary_samples + internal_samples
        if len(samples) > 0:
            targets = [1] * len(positive_samples) + [0] * (len(samples) - len(positive_samples))
            features = torch.stack(
                [self._span_embed(forwards[:, bid], backwards[:, bid], begin, end)
                 for bid, begin, end in samples], dim=0)
            targets = torch.tensor(targets, dtype=torch.float, device=device)

            positive_weights = torch.tensor([1] * len(positive_samples), dtype=torch.float, device=device)
            negative_weights = torch.tensor([1] * len(negative_samples), dtype=torch.float, device=device)
            # boundary_weights = torch.tensor([1] * len(boundary_samples), dtype=torch.float, device=device)
            # internal_weights = torch.tensor([1] * len(internal_samples), dtype=torch.float, device=device)

            negative_weights = negative_weights * (positive_weights.sum() / negative_weights.sum())

            positive_total = positive_weights.sum()
            weights = torch.cat((
                positive_weights,
                negative_weights * (2 * positive_total / negative_weights.sum()),
                # boundary_weights * (positive_total * 0.4 / boundary_weights.sum()),
                # internal_weights * (positive_total * 0.6 / internal_weights.sum())
            ), dim=-1)
        else:
            features = torch.tensor([], device=device)
            targets = torch.tensor([], device=device)
            weights = torch.tensor([], device=device)

        return samples, features, targets, weights

    def _span_embed(self, forwards: torch.Tensor, backwards: torch.Tensor, begin: int, end: int):
        assert 0 < begin < end < forwards.size(0)
        return torch.cat((forwards[end - 1] - forwards[begin - 1],
                          backwards[begin] - backwards[end]), dim=-1)
