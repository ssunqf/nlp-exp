#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
from typing import Tuple, List, Set, Dict, Optional
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
                hidden: torch.Tensor,
                labels: List[List[Label]]) -> Dict[str, torch.Tensor]:

        device = hidden.device
        # context_features, context_tags = [], []
        phrase_features, phrase_tags = [], []

        for bid, sen_labels in enumerate(labels):
            for label in sen_labels:
                if label.tags.size(0) > 0:
                    # context = self._context_feature(hidden, bid, label.begin, label.end)
                    # context_features.append(context)
                    # context_tags.append(label.tags)

                    phrase = self._phrase_feature(hidden, bid, label.begin, label.end)
                    phrase_features.append(phrase)
                    phrase_tags.append(label.tags)

        if len(phrase_tags) > 0:
            # context_features = self.hidden2label(self.context2ffn(torch.stack(context_features, dim=0)))
            phrase_features = self.hidden2label(self.phrase2ffn(torch.stack(phrase_features, dim=0)))
            return {
                'loss': self.loss(phrase_features, phrase_tags)
                # 'loss': (self.loss(context_features, context_tags) + self.loss(phrase_features, phrase_tags)) / 2
            }
        else:
            return {
                'loss': torch.tensor([0.0], device=device)
            }

    def predict(self,
                hidden: torch.Tensor,
                labels: List[List[Label]]) -> Dict[str, Dict[str, List[List[Tuple[Label, torch.Tensor]]]]]:
        device = hidden.device
        # context_features, context_tags = [], []
        phrase_features, phrase_tags = [], []
        for bid, sen_labels in enumerate(labels):
            for label in sen_labels:
                if label.tags.size(0) > 0:
                    # context = self._context_feature(hidden, bid, label.begin, label.end)
                    # context_features.append(context)
                    # context_tags.append((bid, label))

                    phrase = self._phrase_feature(hidden, bid, label.begin, label.end)
                    phrase_features.append(phrase)
                    phrase_tags.append((bid, label))

        context_results = [[] for _ in range(len(labels))]
        phrase_results = [[] for _ in range(len(labels))]
        if len(phrase_tags) > 0:
            # context_logits = self.hidden2label(self.context2ffn(torch.stack(context_features, dim=0)))
            # scores, indexes = self.loss.predict(context_logits, 5)
            # for (bid, label), topk in zip(context_tags, indexes):
            #    context_results[bid].append((label, topk.tolist()))

            phrase_logits = self.hidden2label(self.phrase2ffn(torch.stack(phrase_features, dim=0)))
            scores, indexes = self.loss.predict(phrase_logits, 5)
            for (bid, label), topk in zip(phrase_tags, indexes):
                phrase_results[bid].append((label, topk.tolist()))

        return {
            self.name: {
                # 'context': context_results,
                'phrase': phrase_results
            }
        }

    def named_embedding(self):
        return self.name, self.hidden2label.weight, self.voc.itos

    def _context_feature(self,
                         hidden: torch.Tensor,
                         bid: int, begin: int, end: int):
        return torch.cat((hidden[begin-1, bid, :self.hidden_dim], hidden[end, bid, self.hidden_dim:]), dim=-1)

    def _phrase_feature(self,
                        hidden: torch.Tensor,
                        bid: int, begin: int, end: int):
        return torch.cat((hidden[end-1, bid, :self.hidden_dim] - hidden[begin-1, bid, :self.hidden_dim],
                          hidden[begin, bid, self.hidden_dim:] - hidden[end, bid, self.hidden_dim:], ), dim=-1)


class LMClassifier(nn.Module):
    def __init__(self,
                 voc_size: int,
                 voc_dim: int,
                 hidden_dim: int,
                 shared_weight: Optional[torch.Tensor]=None,
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
        if shared_weight is not None:
            assert self.context2token.weight.size() == shared_weight.size()
            self.context2token.weight = shared_weight

        self.inv_temperature = nn.Parameter(torch.tensor([1.0], dtype=torch.float))

    def forward(self,
                hidden: torch.Tensor,
                tokens: torch.Tensor,
                lens: torch.Tensor) -> Dict[str, torch.Tensor]:
        seq_len, batch_size, dim = hidden.size()
        assert dim == self.hidden_dim * 2
        middle = torch.cat((hidden[:-2, :, :self.hidden_dim], hidden[2:, :, self.hidden_dim:]), dim=-1)

        logit = self.context2token(self.context_ffn(middle)) * self.inv_temperature
        return {
            'loss': F.cross_entropy(logit.view(-1, self.voc_size),
                                    tokens[1:-1].view(-1),
                                    ignore_index=self.padding_idx),
        }

    def predict(self, hidden: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, dim = hidden.size()
        assert dim == self.hidden_dim * 2
        middle = torch.cat((hidden[:-2, :, :self.hidden_dim], hidden[2:, :, self.hidden_dim:]), dim=-1)

        logit = self.context2token(self.context_ffn(middle)) * self.inv_temperature
        return logit.max(dim=-1)[1]


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class PhraseClassifier(nn.Module):
    def __init__(self, name, hidden_dim, max_length=15, dropout=0.3):
        super(PhraseClassifier, self).__init__()

        self.name = name
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden: torch.Tensor,
                lens: torch.Tensor, phrases: List[List[Tuple[int, int]]]) -> Dict[str, torch.Tensor]:

        samples, features, targets, weigths = self._featured(hidden, lens, phrases)
        if len(samples) == 0:
            loss = torch.tensor([0.0], device=hidden.device)
        else:
            loss = F.binary_cross_entropy(self.ffn(features), targets.unsqueeze(-1), weight=weigths.unsqueeze(-1))

        return {'loss': loss}

    def predict(self,
                hidden: torch.Tensor,
                lens: torch.Tensor,
                phrases: List[List[Tuple[int, int, int]]]) -> List[List[Tuple[int, int, float, float, float]]]:

        samples, features, targets, weights = self._featured(hidden, lens, phrases)
        if len(samples) == 0:
            return [[] for _ in range(len(phrases))]

        preds = self.ffn(features)

        targets = targets.tolist()
        weights = weights.tolist()
        preds = preds.squeeze(-1).tolist()
        results = [[] for _ in range(len(phrases))]
        for id, (bid, begin, end) in enumerate(samples):
            results[bid].append((begin, end, targets[id], preds[id], weights[id]))
        return results

    def find_phrase(self, hidden: torch.Tensor, lens: torch.Tensor,
                    threshold=0.8) -> List[List[Tuple[int, int, float]]]:
        phrases = []
        for bid in range(lens.size(0)):
            samples = []
            features = []
            for begin in range(1, lens[bid] - 1):
                for step in range(2, min(self.max_length + 1, lens[bid] - begin)):
                    samples.append((begin, begin + step))
                    features.append(self._span_embed(hidden[:, bid], begin, begin + step))
            if len(features) > 0:
                features = torch.stack(features, dim=0)
                probs = self.ffn(features).squeeze(-1).tolist()
            else:
                probs = []

            phrases.append([(begin, end, prob) for (begin, end), prob in zip(samples, probs) if prob > threshold])

        return phrases

    def _featured(self,
                  hidden: torch.Tensor, lens: torch.Tensor,
                  phrases: List[List[Tuple[int, int, int]]]) -> Tuple[List[Tuple[int, int, int]],
                                                                    torch.Tensor, torch.Tensor, torch.Tensor]:
        device = hidden.device
        positive_samples, negative_samples, noise_samples = [], [], []
        for bid, sen_phrases in enumerate(phrases):
            positive_phrases = []

            for pid, (begin, end, flag) in enumerate(sen_phrases):
                if flag == 1:
                    if pid == 0 or (sen_phrases[pid-1][1] < begin):
                        positive_phrases.append((begin, end))

            noise_samples.extend((bid, begin, end) for begin, end, flag in sen_phrases if flag == 0)

            positive_samples.extend((bid, begin, end) for begin, end in positive_phrases)

            for (f_b, f_e), (s_b, s_e) in pairwise(positive_phrases):
                if f_b == s_b:
                    assert f_e < s_e
                    for left in range(f_b + 1, f_e):
                        for right in range(f_e+1, s_e):
                            assert left < right
                            negative_samples.append((bid, left, right))
                elif f_e < s_b:
                    for left in range(f_b, f_e):
                        for right in range(s_b + 1, s_e + 1):
                            if left != f_b or right != s_e:
                                assert left < right
                                negative_samples.append((bid, left, right))

            negative_samples.extend(
                (bid, left, right)
                for (f_b, f_e), (s_b, s_e) in pairwise(positive_phrases)
                for left in range(f_b, f_e)
                for right in range(s_b+1, s_e+1)
                if (left != f_b or right != s_e))

            negative_samples.extend(
                (bid, n_begin, n_end)
                for begin, end in positive_phrases if end - begin > 1
                for mid in range(begin, end)
                for n_begin, n_end in [(random.randint(max(0, begin - self.max_length), begin - 1), mid),
                                       (mid, random.randint(end + 1, min(lens[bid], end + self.max_length)))]
                if (0 < n_begin < begin < n_end < end) or (begin < n_begin < end < n_end < lens[bid])
            )

            '''
            noise_samples.extend(
                (bid, n_begin, n_end)
                for begin, end in sen_phrases if end - begin > 2
                for n_begin, n_end in [
                    (begin, random.randint(begin+1, end-1)),
                    (random.randint(begin+1, end-1), end),
                    (random.randint(max(0, begin - self.max_length), begin - 1), end),
                    (begin, random.randint(end + 1, min(lens[bid], end + self.max_length)))]
                if 0 < n_begin < n_end < lens[bid]
            )
            '''
        positive_samples = list(filter(lambda x: x[1] < x[2] < lens[x[0]], positive_samples))
        negative_samples = list(filter(lambda x: x[1] < x[2] < lens[x[0]], negative_samples))
        noise_samples = list(filter(lambda x: x[1] < x[2] < lens[x[0]], noise_samples))
        samples = positive_samples + negative_samples + noise_samples
        if len(samples) > 0:
            targets = [1] * len(positive_samples) + [0] * (len(samples) - len(positive_samples))
            features = torch.stack(
                [self._span_embed(hidden[:, bid], begin, end)
                 for bid, begin, end in samples], dim=0)
            targets = torch.tensor(targets, dtype=torch.float, device=device)

            positive_weights = torch.tensor([1] * len(positive_samples), dtype=torch.float, device=device)
            negative_weights = torch.tensor([1] * len(negative_samples), dtype=torch.float, device=device)
            noise_weights = torch.tensor([1] * len(noise_samples), dtype=torch.float, device=device)

            negative_weights = negative_weights * (positive_weights.sum() / negative_weights.sum())

            positive_total = positive_weights.sum()
            weights = torch.cat((
                positive_weights,
                negative_weights * (2 * positive_total / negative_weights.sum()),
                noise_weights * (0.5 * positive_total / noise_weights.sum())
            ), dim=-1)

        else:
            features = torch.tensor([], device=device)
            targets = torch.tensor([], device=device)
            weights = torch.tensor([], device=device)

        return samples, features, targets, weights

    def _span_embed(self, hidden: torch.Tensor, begin: int, end: int):
        if not(0 < begin < end < hidden.size(0)):
            print(begin, end, hidden.size(0))
        assert 0 < begin < end < hidden.size(0)
        return torch.cat((hidden[end - 1, :self.hidden_dim] - hidden[begin - 1, :self.hidden_dim],
                          hidden[begin, self.hidden_dim:] - hidden[end, self.hidden_dim:]), dim=-1)


# positive pointwise mutual information
class PPMI(nn.Module):
    def __init__(self, name, hidden_dim, max_length=15, dropout=0.3):
        super(PPMI, self).__init__()

        self.name = name
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self,
                hidden: torch.Tensor,
                lens: torch.Tensor,
                phrases: List[List[Tuple[int, int, int, float]]]) -> Dict[str, torch.Tensor]:

        samples, features, targets, weigths = self._featured(hidden, lens, phrases)
        if len(samples) == 0:
            loss = torch.tensor([0.0], device=hidden.device)
        else:
            loss = F.binary_cross_entropy(self.ffn(features), targets.unsqueeze(-1), weight=weigths.unsqueeze(-1))

        return {
            'loss': loss,
        }

    def predict(self,
                hidden: torch.Tensor,
                lens: torch.Tensor,
                phrases: List[List[Tuple[int, int, int, float]]]) -> List[List[Tuple[int, int, float, float, float]]]:

        samples, features, targets, weights = self._featured(hidden, lens, phrases)
        if len(samples) == 0:
            return [[] for _ in range(len(phrases))]

        preds = self.ffn(features)

        targets = targets.tolist()
        weights = weights.tolist()
        preds = preds.squeeze(-1).tolist()
        results = [[] for _ in range(len(phrases))]
        for id, (bid, begin, end) in enumerate(samples):
            results[bid].append((begin, end, targets[id], preds[id], weights[id]))
        return results

    def find_phrase(self, hidden: torch.Tensor, lens: torch.Tensor,
                    threshold=0.8) -> List[List[Tuple[int, int, float]]]:
        phrases = []
        for bid in range(lens.size(0)):
            samples = []
            features = []
            for begin in range(1, lens[bid] - 1):
                for step in range(2, min(self.max_length + 1, lens[bid] - begin)):
                    samples.append((begin, begin + step))
                    features.append(self._span_embed(hidden[:, bid], begin, begin + step))
            if len(features) > 0:
                features = torch.stack(features, dim=0)
                probs = self.ffn(features).squeeze(-1).tolist()
            else:
                probs = []

            phrases.append([(begin, end, prob) for (begin, end), prob in zip(samples, probs) if prob > threshold])

        return phrases

    def _featured(self,
                  hidden: torch.Tensor, lens: torch.Tensor,
                  phrases: List[List[Tuple[int, int, int, float]]]) -> Tuple[List[Tuple[int, int, int]],
                                                                    torch.Tensor, torch.Tensor, torch.Tensor]:
        device = hidden.device

        positive_samples, positive_weights = [], []
        negative_samples, negative_weights = [], []
        for bid, sentence in enumerate(phrases):
            for begin, end, flag, weight in sentence:
                if flag == 1:
                    positive_samples.append((bid, begin, end))
                    positive_weights.append(weight)
                else:
                    negative_samples.append((bid, begin, end))
                    negative_weights.append(weight)

        samples = positive_samples + negative_samples
        if len(positive_samples) > 0:

            features = torch.stack(
                [self._span_embed(hidden[:, bid], begin, end) for bid, begin, end in samples],
                dim=0)
            targets = torch.tensor([1] * len(positive_samples) + [0] * len(negative_samples),
                                   dtype=torch.float,
                                   device=device)

            positive_weights = torch.tensor(positive_weights, dtype=torch.float, device=device)
            negative_weights = torch.tensor(negative_weights, dtype=torch.float, device=device)

            scaled_ratio = 2 * len(positive_samples) / len(samples)
            negative_weights = negative_weights * scaled_ratio

            weights = torch.cat((positive_weights, negative_weights), dim=0)

        else:
            features = torch.tensor([], device=device)
            targets = torch.tensor([], device=device)
            weights = torch.tensor([], device=device)

        return samples, features, targets, weights

    def _span_embed(self, hidden: torch.Tensor, begin: int, end: int):
        '''
        return torch.cat((hidden[end - 1, :self.hidden_dim] - hidden[begin - 1, :self.hidden_dim],
                          hidden[begin, self.hidden_dim:] - hidden[end, self.hidden_dim:]), dim=-1)
        '''
        mean = hidden[begin:end].mean(dim=0)
        return torch.cat((hidden[begin], mean, hidden[end]), dim=-1)
