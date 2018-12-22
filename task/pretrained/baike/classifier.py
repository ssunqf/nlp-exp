#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
from typing import Tuple, List, Set
import math
import random

import torch
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import Vocab

from task.pretrained.transformer.attention import MultiHeadedAttention

from .base import Label


class NegativeSampleLoss(nn.Module):
    def __init__(self, voc: Vocab, label_size: int, sample_ratio=2, dropout=0.3):
        super(NegativeSampleLoss, self).__init__()
        self.voc = voc
        self.label_size = label_size
        self.sample_ratio = sample_ratio

        self.dropout = nn.Dropout(dropout)
        self.h2o = nn.Linear(label_size, len(voc), bias=False)

        self.register_buffer('negative_sample_probs', self._negative_sample_probs())

    def _negative_sample_probs(self):
        freqs = torch.tensor([self.voc.freqs.get(s, 1e-5) for s in self.voc.itos]).pow(0.75)
        return freqs / freqs.sum()

        # freqs = 1 - (10e-5/torch.FloatTensor([self.voc.freqs.get(s, 1e-5) for s in self.voc.itos])).sqrt()
        # return freqs

    def forward(self, features: torch.Tensor, targets: List[torch.Tensor]) -> torch.Tensor:
        return sum(self._forward(feature, target) for feature, target in zip(features, targets))

    def _forward(self,
                feature: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        probs = self.negative_sample_probs.index_fill_(0, targets, 0)
        noises = torch.multinomial(probs, self.sample_ratio * targets.size(0))

        feature = self.dropout(feature)

        targets_out = self._output(feature, targets)

        noise_out = self._output(feature, noises)

        out = torch.cat([targets_out, -noise_out], dim=-1)

        loss = -F.logsigmoid(out).mean()
        return loss

    def predict(self, feature: torch.Tensor,
                topk=5) -> Tuple[torch.Tensor, torch.Tensor]:
        scores, indexes = self.h2o(self.dropout(feature)).topk(topk)
        return scores, indexes

    def _output(self, input, indices: torch.LongTensor):
        weight = self.h2o.weight.index_select(0, indices)
        return F.linear(input, weight)


# Mixture of Softmaxes
class MOSLoss(nn.Module):
    def __init__(self, voc: Vocab, feature_size: int, num_experts: int, label_size: int, dropout=0.3):
        super(MOSLoss, self).__init__()
        self.voc = voc
        self.feature_size = feature_size
        self.label_size = label_size
        self.num_epxerts = num_experts

        self.voc_size = len(voc)

        self.dropout = nn.Dropout(dropout)
        self.hidden2expert = nn.Linear(feature_size, num_experts * self.label_size)
        self.hidden2prior = nn.Linear(feature_size, num_experts)

        self.expert2label = nn.Linear(label_size, len(voc), bias=True)

    # http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
    def _discard_probs(self):
        counts = torch.tensor([self.voc.freqs.get(s, 1e-5) for s in self.voc.itos])
        freqs = counts / counts.sum()
        discard_probs = 1 - (10e-5/freqs).sqrt()
        discard_probs = discard_probs.masked_fill(discard_probs < 0., 0.)
        return discard_probs

    def forward(self, features: torch.Tensor, targets: List[torch.Tensor]):

        expert_prob = self.expert2label(self.hidden2expert(self.dropout(features)).view(-1, self.label_size)) \
            .view(-1, self.num_epxerts, self.voc_size).softmax(-1)
        prior = self.hidden2prior(features).softmax(-1)

        log_prob = torch.bmm(prior.unsqueeze(-2), expert_prob).squeeze(-2).log()

        loss = torch.tensor([0.0], device=features.device)
        for log_softmax, target in zip(log_prob, targets):
            neg_log_softmax = -log_softmax.gather(-1, target)
            scaled_ratio = 1 - self.discard_probs.gather(-1, target)
            loss += (neg_log_softmax * scaled_ratio).sum() / scaled_ratio.sum()
        return loss / (len(targets) + 1e-5)

    def predict(self, features: torch.Tensor, topk=5) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.hidden2label(self.dropout(features)).softmax(-1).topk(topk, dim=-1)


class SoftmaxLoss(nn.Module):
    def __init__(self, voc: Vocab, label_size: int, dropout=0.3):
        super(SoftmaxLoss, self).__init__()
        self.voc = voc
        self.label_size = label_size

        self.dropout = nn.Dropout(dropout)
        self.hidden2label = nn.Linear(label_size, len(voc), bias=True)

        self.discard_probs = nn.Parameter(self._discard_probs(), requires_grad=False)

    # http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
    def _discard_probs(self):
        counts = torch.tensor([self.voc.freqs.get(s, 1e-5) for s in self.voc.itos])
        freqs = counts / counts.sum()
        discard_probs = 1 - (10e-5/freqs).sqrt()
        discard_probs = discard_probs.masked_fill(discard_probs < 0., 0.)
        return discard_probs

    def forward(self, features: torch.Tensor, targets: List[torch.Tensor]) -> torch.Tensor:
        log_softmaxes = self.hidden2label(self.dropout(features)).log_softmax(-1)
        loss = torch.tensor([0.0], device=features.device)

        for log_softmax, target in zip(log_softmaxes, targets):
            neg_log_softmax = -log_softmax.gather(-1, target)
            scaled_ratio = 1 - self.discard_probs.gather(-1, target)
            loss += (neg_log_softmax * scaled_ratio).sum() / scaled_ratio.sum()

        return loss / (len(targets) + 1e-5)

    def predict(self, features: torch.Tensor, topk=5) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.hidden2label(self.dropout(features)).softmax(-1).topk(topk, dim=-1)


class AdaptiveLoss(nn.Module):
    def __init__(self, vocab: Vocab, label_size: int, dropout=0.3):
        super(AdaptiveLoss, self).__init__()
        self.vocab = vocab
        self.label_size = label_size

        self.dropout = nn.Dropout(dropout)
        self.loss = nn.AdaptiveLogSoftmaxWithLoss(
            label_size,
            len(self.vocab),
            cutoffs=self._cutoffs(),
            div_value=2)

        self.discard_probs = nn.Parameter(self._discard_probs(), requires_grad=False)

    # http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
    def _discard_probs(self):
        counts = torch.tensor([self.vocab.freqs.get(s, 1e-5) for s in self.vocab.itos])
        freqs = counts / counts.sum()
        discard_probs = 1 - (10e-5/freqs).sqrt()
        discard_probs = discard_probs.masked_fill(discard_probs < 0, 0)
        return discard_probs

    def forward(self, features: torch.Tensor,
                targets: List[torch.Tensor]) -> torch.Tensor:
        logits = self.loss.log_prob(self.dropout(features))
        loss = torch.tensor([0.0], device=features.device)
        for logit, target in zip(logits, targets):
            neg_log_softmax = -logit.gather(0, target)
            scaled_ratio = 1 - self.discard_probs.gather(0, target)
            loss += (neg_log_softmax * scaled_ratio).sum() / scaled_ratio.sum()
        return loss / (len(targets) + 1e-5)

    def predict(self, feature: torch.Tensor, topk=5) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.loss.log_prob(self.dropout(feature)).topk(topk, dim=-1)

    def _cutoffs(self):
        voc_len = len(self.vocab)
        size = self.label_size
        cutoffs = [voc_len//10]
        while cutoffs[-1]*2 < voc_len and size//2 > 50:
            cutoffs.append(cutoffs[-1] * 2)
            size //= 2
        print('cutoffs = %s' % cutoffs)
        return cutoffs


class LabelClassifier(nn.Module):
    loss_dict = {
        'softmax': SoftmaxLoss,
        'negativesample': NegativeSampleLoss,
        'adaptivesoftmax': AdaptiveLoss
    }

    def __init__(self, name: str, loss_type: str, voc: Vocab,
                 hidden_size, label_size, dropout=0.3):
        super(LabelClassifier, self).__init__()
        self.name = name
        self.voc = voc
        self.hidden_size = hidden_size
        self.label_size = label_size

        self.hidden2feature = nn.Sequential(
            nn.Linear(hidden_size * 2, self.label_size),
            nn.Sigmoid(),
            nn.Linear(self.label_size, self.label_size)
        )

        assert loss_type.lower() in self.loss_dict

        self.loss = self.loss_dict[loss_type.lower()](self.voc, self.label_size)

    def forward(self, hidden: torch.Tensor,
                mask: torch.Tensor, labels: List[List[Label]]) -> torch.Tensor:
        features = []
        tags = []
        for bid, sen_labels in enumerate(labels):
            for label in sen_labels:
                if label.tags.size(0) > 0:
                    features.append(self._span_embed(hidden, bid, label.begin, label.end))
                    tags.append(label.tags)

        if len(tags) > 0:
            features = self.hidden2feature(torch.stack(features, dim=0))
            return self.loss(features, tags)
        else:
            return torch.tensor([0.0], device=hidden.device)

    def predict(self, hidden: torch.Tensor, mask: torch.Tensor,
                labels: List[List[Label]]) -> List[List[Tuple[Label, torch.Tensor]]]:
        features = []
        flat_labels = []
        for bid, sen_labels in enumerate(labels):
            for label in sen_labels:
                if label.tags.size(0) > 0:
                    features.append(self._span_embed(hidden, bid, label.begin, label.end))
                    flat_labels.append((bid, label))

        results = [[] for _ in range(len(labels))]
        if len(features) > 0:
            features = self.hidden2feature(torch.stack(features, dim=0))
            scores, indexes = self.loss.predict(features, 5)
            for (bid, label), topk in zip(flat_labels, indexes):
                results[bid].append((label, topk.tolist()))

        return results

    def named_embeddings(self):
        yield self.name, self.loss.hidden2label.weight, self.loss.voc.itos

    def _span_embed(self, hidden: torch.Tensor, bid: int, begin: int, end: int):
        return torch.cat([hidden[begin, bid], hidden[end-1, bid]], dim=-1)


class PhraseClassifier(nn.Module):
    def __init__(self, hidden_size, max_length=15, dropout=0.3):
        super(PhraseClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.max_length = max_length
        self.ffn = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor,
                phrases: List[Tuple[int, int, int]]) -> torch.Tensor:

        samples, features, targets = self._make_sample(hidden, mask, phrases)
        if len(samples) == 0:
            return torch.tensor([0.0], device=hidden.device)

        return F.binary_cross_entropy(self.ffn(features), targets.unsqueeze(-1))

    def predict(self, hidden: torch.Tensor, mask: torch.Tensor,
                phrases: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int, float, float]]]:

        samples, features, targets = self._make_sample(hidden, mask, phrases)
        if len(samples) == 0:
            return [[] for _ in range(len(phrases))]

        preds = self.ffn(features)

        targets = targets.tolist()
        preds = preds.squeeze().tolist()
        results = [[] for _ in range(len(phrases))]
        for id, (bid, begin, end) in enumerate(samples):
            results[bid].append((begin, end, targets[id], preds[id]))
        return results

    def find_phrase(self, hidden: torch.Tensor,
                    mask: torch.Tensor, threshold=0.8) -> List[List[Tuple[int, int, float]]]:

        lens = mask.sum(0)
        phrases = []
        for bid in range(lens.size(0)):
            samples = []
            features = []
            for begin in range(1, lens[bid] - 1):
                for step in range(2, min(self.max_length+1, lens[bid] - begin)):
                    samples.append((begin, begin+step))
                    features.append(self._span_embed(hidden, bid, begin, begin + step))
            if len(features) > 0:
                features = torch.stack(features, dim=0)
                probs = self.ffn(features).squeeze(-1).tolist()
            else:
                probs = []

            phrases.append([(begin, end, prob) for (begin, end), prob in zip(samples, probs) if prob > threshold])

        return phrases

    def _make_sample(self, hidden: torch.Tensor, mask: torch.Tensor,
                     phrases: List[List[Tuple[int, int]]]) -> Tuple[List[Tuple[int, int, int]], torch.Tensor, torch.Tensor]:
        lens = mask.sum(0)
        samples, targets = [], []
        for bid, sen_phrases in enumerate(phrases):
            for begin, end in sen_phrases:
                if end - begin > 1:
                    samples.append((bid, begin, end))
                    targets.append(1)
                    for mid in range(begin, end):
                        for n_begin, n_end in [(random.randint(max(0, begin-self.max_length), begin-1), mid),
                                               (mid, random.randint(end+1, min(lens[bid], end+self.max_length)))]:
                            if (0 <= n_begin < begin < n_end < end - 1) or (begin < n_begin < end < n_end < lens[bid]):
                                samples.append((bid, n_begin, n_end))
                                targets.append(0)

        if len(samples) > 0:
            features = torch.stack([self._span_embed(hidden, bid, begin, end) for bid, begin, end in samples], dim=0)
            targets = torch.tensor(targets, dtype=torch.float, device=hidden.device)
        else:
            features = torch.tensor([], device=hidden.device)
            targets = torch.tensor([], device=hidden.device)

        return samples, features, targets

    def _span_embed(self, hidden: torch.Tensor, bid: int, begin: int, end: int):
        return torch.cat([hidden[begin, bid], hidden[end-1, bid]], dim=-1)
