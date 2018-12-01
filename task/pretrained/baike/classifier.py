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

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.h2o.weight, gain=nn.init.calculate_gain('sigmoid'))

    def _negative_sample_probs(self):
        freqs = torch.FloatTensor([self.voc.freqs.get(s, 1e-5) for s in self.voc.itos]).pow(0.75)
        return freqs / freqs.sum()

        # freqs = 1 - (10e-5/torch.FloatTensor([self.voc.freqs.get(s, 1e-5) for s in self.voc.itos])).sqrt()
        # return freqs

    def forward(self,
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

    def predict(self,
                feature: torch.Tensor,
                topk=5) -> Tuple[torch.Tensor, torch.Tensor]:
        scores, indexes = self.h2o(self.dropout(feature)).topk(topk)
        return scores, indexes

    def _output(self, input, indices: torch.LongTensor):
        weight = self.h2o.weight.index_select(0, indices)
        return F.linear(input, weight)


class SoftmaxLoss(nn.Module):
    def __init__(self, voc: Vocab, label_size: int, dropout=0.3):
        super(SoftmaxLoss, self).__init__()
        self.voc = voc
        self.label_size = label_size

        self.dropout = nn.Dropout(dropout)
        self.h2o = nn.Linear(label_size, len(voc), bias=False)

        self.discard_probs = nn.Parameter(self._discard_probs(), requires_grad=False)

        self.reset_parameters()

    # http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
    def _discard_probs(self):
        counts = torch.FloatTensor([self.voc.freqs.get(s, 1e-5) for s in self.voc.itos])
        freqs = counts / counts.sum()
        discard_probs = 1 - (10e-5/freqs).sqrt()
        discard_probs = discard_probs.masked_fill(discard_probs < 0, 0)
        return discard_probs

    def reset_parameters(self):
        nn.init.xavier_normal_(self.h2o.weight, gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, feature: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        feature = self.dropout(feature)
        neg_log_softmax = -self.h2o(feature).log_softmax(0).gather(0, targets)
        scaled_ratio = 1 - self.discard_probs.gather(0, targets)
        return (neg_log_softmax * scaled_ratio).sum()

    def predict(self, feature: torch.Tensor, topk=5) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.h2o(feature).softmax(0).topk(topk)


class AdaptiveLoss(nn.Module):
    def __init__(self, vocab: Vocab, label_size: int, dropout=0.3):
        super(AdaptiveLoss, self).__init__()
        self.vocab = vocab
        self.label_size = label_size

        self.loss = nn.AdaptiveLogSoftmaxWithLoss(
            label_size,
            len(self.vocab),
            cutoffs=self._cutoffs(),
            div_value=2)

        self.discard_probs = nn.Parameter(self._discard_probs(), requires_grad=False)

        self.loss.reset_parameters()

    # http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
    def _discard_probs(self):
        counts = torch.FloatTensor([self.vocab.freqs.get(s, 1e-5) for s in self.vocab.itos])
        freqs = counts / counts.sum()
        discard_probs = 1 - (10e-5/freqs).sqrt()
        discard_probs = discard_probs.masked_fill(discard_probs < 0, 0)
        return discard_probs

    def forward(self,
                feature: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        feature = self.dropout(feature)
        neg_log_softmax = -self.loss.log_prob(feature.unsqueeze(0)).squeeze(0).gather(0, targets)
        scaled_ratio = 1 - self.discard_probs.gather(0, targets)
        return (neg_log_softmax * scaled_ratio).sum()

    def predict(self, feature: torch.Tensor, topk=5) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.loss.log_prob(feature.unsqueeze(0)).squeeze(0).topk(topk)

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
                 hidden_size, label_size, attention_num_heads=None, dropout=0.3):
        super(LabelClassifier, self).__init__()
        self.name = name
        self.voc = voc
        self.hidden_size = hidden_size
        self.label_size = label_size

        self.attention = None if attention_num_heads is None \
            else MultiHeadedAttention(attention_num_heads, hidden_size)
        self.hidden2feature = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, self.label_size),
            nn.Tanh()
        )

        assert loss_type.lower() in self.loss_dict

        self.loss = self.loss_dict[loss_type.lower()](self.voc, self.label_size)

    def forward(self,
                hidden: torch.Tensor,
                mask: torch.Tensor,
                labels: List[List[Label]]) -> torch.Tensor:
        if self.attention is not None:
            hidden = self.attention(hidden, hidden, hidden, mask)
        loss = torch.zeros(1, device=hidden.device)
        sum = 0
        for bid, sen_labels in enumerate(labels):
            for label in sen_labels:
                if label.tags.size(0) > 0:
                    sum += label.tags.size(0)
                    span_emb = self._span_embed(hidden, bid, label.begin, label.end)
                    feature = self.hidden2feature(span_emb)
                    loss += self.loss(feature, label.tags)
        return loss / (sum + 1e-5)

    def predict(self,
                hidden: torch.Tensor,
                mask: torch.Tensor,
                labels: List[List[Label]]) -> List[List[Tuple[Label, torch.Tensor]]]:
        if self.attention is not None:
            hidden = self.attention(hidden, hidden, hidden, mask)
        results = []
        for bid, sen_labels in enumerate(labels):
            sen_result = []
            for label in sen_labels:
                if label.tags.size(0) > 0:
                    span_emb = self._span_embed(hidden, bid, label.begin, label.end)
                    feature = self.hidden2feature(span_emb)
                    scores, indexes = self.loss.predict(feature, 5)
                    sen_result.append((label, indexes.tolist()))
            results.append(sen_result)

        return results

    def _span_embed(self, left: torch.Tensor, right: torch.Tensor, bid: int, begin: int, end: int):
        return left[begin, bid] + right[end - 1, bid]


class PhraseClassifier(nn.Module):
    def __init__(self, hidden_size, max_length=10, dropout=0.3):
        super(PhraseClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.max_length = max_length
        self.ffn = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )

    def forward(self,
                hidden: torch.Tensor,
                mask: torch.Tensor,
                phrases: List[Tuple[int, int, int]]) -> torch.Tensor:
        samples, features, targets = self._make_sample(hidden, mask, phrases)
        pos_weight = targets.size(0) / targets.sum() - 1
        weights = torch.full(targets.size(), 1, device=hidden.device).masked_fill_(targets > 0.5, pos_weight)
        loss = F.binary_cross_entropy(self.ffn(features), targets.unsqueeze(-1), reduction='none')
        return torch.mean(loss.squeeze() * weights)

    def predict(self,
                hidden: torch.Tensor,
                mask: torch.Tensor,
                phrases: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int, float, float]]]:

        samples, features, targets = self._make_sample(hidden, mask, phrases)
        preds = self.ffn(features)

        targets = targets.tolist()
        preds = preds.squeeze().tolist()
        results = [[] for _ in range(len(phrases))]
        for id, (bid, begin, end) in enumerate(samples):
            results[bid].append((begin, end, targets[id], preds[id]))
        return results

    def find_phrase(self, hidden: torch.Tensor, mask: torch.Tensor, threshold=0.8) -> List[List[Tuple[int, int, float]]]:
        lens = mask.sum(0)
        phrases = []
        for bid in range(lens.size(0)):
            samples = []
            features = []
            for begin in range(1, lens[bid] - 1):
                for step in range(2, min(self.max_length+1, lens[bid] - begin)):
                    samples.append((begin, begin+step))
                    features.append(self._span_embed(hidden, bid, begin, begin + step))

            features = torch.stack(features, dim=0)
            probs = self.ffn(features).squeeze(-1).tolist()

            phrases.append([(begin, end, prob) for (begin, end), prob in zip(samples, probs) if prob > threshold])

        return phrases

    def _make_sample(self,
                     hidden: torch.Tensor,
                     mask: torch.Tensor,
                     phrases: List[List[Tuple[int, int]]]) -> Tuple[List[Tuple[int, int, int]], torch.Tensor, torch.Tensor]:
        lens = mask.sum(0)
        samples, targets = [], []
        for bid, sen_phrases in enumerate(phrases):
            for begin, end in sen_phrases:
                if end - begin > 1:
                    samples.append((bid, begin, end))
                    targets.append(1)
                    for mid in range(begin, end):
                        if begin > 0 and mid < end - 1:
                            left = random.randint(max(0, begin-self.max_length), begin-1)
                            samples.append((bid, left, mid+1))
                            targets.append(0)
                        if end < lens[bid] and mid > begin:
                            right = random.randint(end+1, min(lens[bid], end+self.max_length))
                            samples.append((bid, mid, right))
                            targets.append(0)

        features = torch.stack([self._span_embed(hidden, bid, begin, end)
                               for bid, begin, end in samples], dim=0)
        targets = torch.FloatTensor(targets, device=hidden.device)
        return samples, features, targets

    def _span_embed(self, hidden: torch.Tensor, bid: int, begin: int, end: int):
        return torch.cat([hidden[begin-1:begin+1, bid], hidden[end-1:end+1, bid]], dim=0).view(-1)