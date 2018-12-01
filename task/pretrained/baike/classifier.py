#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
from typing import Tuple, List
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import Vocab

from task.pretrained.transformer.attention import MultiHeadedAttention

from .base import Label


class NegativeSampleLoss(nn.Module):
    def __init__(self, voc: Vocab, label_size: int, sample_ratio=2, dropout=0.2):
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
        self.h2o = nn.Linear(label_size, len(voc), bias=True)

        self.register_buffer('discard_probs', self._discard_probs())

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
        neg_log_softmax = -self.h2o(feature).log_softmax(-1).gather(-1, targets)
        scaled_ratio = 1 - self.discard_probs.gather(0, targets)
        return (neg_log_softmax * scaled_ratio).sum()
        # return neg_log_softmax.sum()

    def predict(self, feature: torch.Tensor, topk=5) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.h2o(feature).softmax(-1).topk(topk)


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

        self.register_buffer('discard_probs', self._discard_probs())

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
        # return neg_log_softmax.sum()

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
                 hidden_size, label_size, attention_num_heads=None, dropout=0.2):
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
            nn.Sigmoid())
            # nn.Linear(self.label_size, self.label_size))

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

    def _span_embed(self, hidden: torch.Tensor, bid: int, begin: int, end: int):
        # mean = hidden[begin:end, bid].mean(0)
        # return torch.cat([hidden[begin-1, bid], mean, hidden[end, bid]], dim=-1)
        # return torch.cat([hidden[begin-1:begin+1, bid], hidden[end-1:end+1, bid]], dim=0).view(-1)
        return torch.cat([hidden[begin, bid], hidden[end-1, bid]], dim=-1)
