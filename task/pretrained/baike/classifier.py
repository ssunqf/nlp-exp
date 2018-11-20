#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
from typing import Tuple, List

import torch
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import Vocab

from task.pretrained.transformer.attention import MultiHeadedAttention

from .base import Label


class NegativeSampleLoss(nn.Module):
    def __init__(self, voc: Vocab, dim: int, num_samples=20, dropout=0.2):
        super(NegativeSampleLoss, self).__init__()
        self.voc = voc
        self.dim = dim
        self.num_samples = num_samples

        self.dropout = nn.Dropout(dropout)
        self.h2o = nn.Linear(dim, len(voc))

        self.register_buffer('label_probs', self._sample_probs())

    def _sample_probs(self):
        freqs = torch.FloatTensor([self.voc.freqs.get(s, 1e-5) for s in self.voc.itos]).pow(0.75)
        return freqs / freqs.sum()

    def forward(self,
                feature: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        probs = self.label_probs.index_fill_(0, targets, 0)
        noises = torch.multinomial(probs, self.num_samples)

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
        bias = self.h2o.bias.index_select(0, indices) if self.h2o.bias is not None else None
        return F.linear(input, weight, bias)


class SoftmaxLoss(nn.Module):
    def __init__(self, voc: Vocab, dim: int, dropout=0.2):
        super(SoftmaxLoss, self).__init__()
        self.voc = voc
        self.dim = dim

        self.h2o = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, len(voc)))

    def forward(self, feature: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return -self.h2o(feature).log_softmax(-1).gather(-1, targets).mean()

    def predict(self, feature: torch.Tensor, topk=5) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.h2o(feature).softmax(-1).topk(topk)


class AdaptiveLoss(nn.Module):
    def __init__(self, vocab: Vocab, label_size: int, dropout=0.2):
        super(AdaptiveLoss, self).__init__()
        self.vocab = vocab
        self.label_size = label_size

        self.loss = nn.AdaptiveLogSoftmaxWithLoss(
            label_size,
            len(self.vocab),
            cutoffs=self._cutoffs(),
            div_value=2)

    def forward(self,
                feature: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        return -self.loss.log_prob(feature.unsqueeze(0)).squeeze(0).gather(0, targets).mean()

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
    def __init__(self, name, voc, hidden_size, label_size, attention_num_heads, dropout=0.2):
        super(LabelClassifier, self).__init__()
        self.name = name
        self.voc = voc
        self.hidden_size = hidden_size
        self.label_size = label_size

        self.attention = None if attention_num_heads is None \
            else MultiHeadedAttention(attention_num_heads, hidden_size)
        self.hidden2feature = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 3, self.label_size),
            nn.Sigmoid(),
            nn.Linear(label_size, self.label_size))

        self.loss = SoftmaxLoss(self.voc, label_size)
        # self.loss = NegativeSampleLoss(self.voc, label_size)
        # self.loss = AdaptiveLoss(self.voc, label_size)

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
                    sum += 1 # label.tags.size(0)
                    span_emb = self._span_embed(hidden, bid, label.begin, label.end)
                    feature = self.hidden2feature(span_emb)
                    loss += self.loss(feature, label.tags)
        return loss / (sum + 1e-5)

    def predict(self,
                hidden: torch.Tensor,
                mask: torch.Tensor,
                labels: List[List[Label]]) -> List[Tuple[int, Label, torch.Tensor]]:
        if self.attention is not None:
            hidden = self.attention(hidden, hidden, hidden, mask)
        results = []
        for bid, sen_labels in enumerate(labels):
            for label in sen_labels:
                if label.tags.size(0) > 0:
                    span_emb = self._span_embed(hidden, bid, label.begin, label.end)
                    feature = self.hidden2feature(span_emb)
                    scores, indexes = self.loss.predict(feature, 5)
                    results.append((bid, label, indexes.tolist()))

        return results

    def _span_embed(self, hidden: torch.Tensor, bid: int, begin: int, end: int):
        mean = hidden[begin:end, bid].mean(dim=0)
        return torch.cat([hidden[begin - 1, bid], mean, hidden[end, bid]], dim=-1)
