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


class SoftmaxLoss(nn.Module):
    def __init__(self, voc: Vocab, label_dim: int, dropout=0.3, gamma=1):
        super(SoftmaxLoss, self).__init__()
        self.voc = voc
        self.label_dim = label_dim
        self.gamma = gamma

        self.dropout = nn.Dropout(dropout)
        self.hidden2label = nn.Linear(label_dim, len(voc), bias=True)

        self.discard_probs = nn.Parameter(self._discard_probs(), requires_grad=False)

    # http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
    def _discard_probs(self):
        counts = torch.tensor([self.voc.freqs.get(s, 1e-5) for s in self.voc.itos])
        freqs = counts / counts.sum()
        discard_probs = 1 - (10e-5/freqs).sqrt()
        discard_probs = discard_probs.masked_fill(discard_probs < 0., 0.)
        return discard_probs

    def forward(self, features: torch.Tensor, targets: List[torch.Tensor]) -> torch.Tensor:
        probs = self.hidden2label(self.dropout(features)).softmax(-1)
        loss = torch.tensor([0.0], device=features.device)

        count = 0
        for probs, target in zip(probs, targets):
            positive_probs = probs.gather(-1, target)
            _focal_loss = -(1 - positive_probs).pow(self.gamma) * positive_probs.log()
            loss += _focal_loss.sum()
            count += target.size(0)

        return loss / (count + 1e-5)

    def predict(self, features: torch.Tensor, topk=5) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.hidden2label(self.dropout(features)).softmax(-1).topk(topk, dim=-1)


class LabelClassifier(nn.Module):
    def __init__(self, name: str, voc: Vocab,
                 hidden_dim, label_dim, dropout=0.3):
        super(LabelClassifier, self).__init__()
        self.name = name
        self.voc = voc
        self.hidden_dim = hidden_dim
        self.label_dim = label_dim

        self.hidden2feature = nn.Sequential(
            nn.Linear(hidden_dim * 2, self.label_dim),
            nn.Tanh()
        )

        self.loss = SoftmaxLoss(self.voc, self.label_dim)

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
        return torch.cat([hidden[begin-1, bid, :self.hidden_dim],
                          hidden[end, bid, self.hidden_dim:]], dim=-1)


class ContextClassifier(nn.Module):
    def __init__(self,
                 name: str,
                 voc: Vocab,
                 hidden_dim, label_dim,
                 phrase_length_embedding: nn.Embedding,
                 dropout=0.3,
                 focal_loss_gamma=2):
        super(ContextClassifier, self).__init__()
        self.name = name
        self.voc = voc
        self.hidden_dim = hidden_dim
        self.label_dim = label_dim

        # 短语长度embedding, 最长为5， 大于5按5计算
        self.phrase_length_max = phrase_length_embedding.num_embeddings
        self.phrase_length_dim = phrase_length_embedding.embedding_dim
        self.length_embedding = phrase_length_embedding
        self.context2ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2 + self.phrase_length_dim, self.label_dim),
            nn.Tanh()
        )

        self.phrase2ffn = nn.Sequential(
            nn.Linear(hidden_dim * 4 + self.phrase_length_dim, self.label_dim),
            nn.Tanh()
        )

        self.loss = SoftmaxLoss(self.voc, self.label_dim, gamma=focal_loss_gamma)

    def forward(self, forwards: torch.Tensor, backwards: torch.Tensor,
                labels: List[List[Label]]) -> torch.Tensor:

        device = forwards.device
        context_features = []
        context_lengthes = []
        context_tags = []
        phrase_features = []
        phrase_lengthes = []
        phrase_tags = []

        for bid, sen_labels in enumerate(labels):
            for label in sen_labels:
                if label.tags.size(0) > 0:
                    context = torch.cat([forwards[label.begin-1, bid], backwards[label.end, bid]], dim=-1)
                    context_features.append(context)
                    context_lengthes.append(min(label.end-label.begin, self.phrase_length_max) - 1)
                    context_tags.append(label.tags)

                    phrase = self._span_embed(forwards[:, bid], backwards[:, bid], label.begin, label.end)
                    phrase_features.append(phrase)
                    phrase_lengthes.append(min(label.end-label.begin, self.phrase_length_max) - 1)
                    phrase_tags.append(label.tags)

        if len(context_tags) > 0:
            context_length_emb = self.length_embedding(torch.LongTensor(context_lengthes, device=device))
            context_features = self.context2ffn(
                torch.cat([context_length_emb, torch.stack(context_features, dim=0)],
                          dim=-1))
            phrase_length_emb = self.length_embedding(torch.LongTensor(phrase_lengthes, device=device))
            phrase_features = self.phrase2ffn(
                torch.cat([phrase_length_emb, torch.stack(phrase_features, dim=0)], dim=-1))
            return self.loss(torch.cat([context_features, phrase_features], dim=0), context_tags + phrase_tags)
        else:
            return torch.tensor([0.0], device=forwards.device)

    def predict(self, forwards: torch.Tensor, backwards: torch.Tensor,
                labels: List[List[Label]]) -> List[List[Tuple[Label, torch.Tensor]]]:
        device = forwards.device
        context_features = []
        context_lengthes = []
        context_tags = []
        phrase_features = []
        phrase_lengthes = []
        phrase_tags = []
        for bid, sen_labels in enumerate(labels):
            for label in sen_labels:
                if label.tags.size(0) > 0:
                    # features.append(self._span_embed(hidden, bid, label.begin, label.end))
                    context = torch.cat([forwards[label.begin-1, bid], backwards[label.end, bid]], dim=-1)
                    context_features.append(context)
                    context_lengthes.append(min(label.end - label.begin, self.phrase_length_max) - 1)
                    context_tags.append((bid, label))

                    phrase = self._span_embed(forwards[:, bid], backwards[:, bid], label.begin, label.end)
                    phrase_features.append(phrase)
                    phrase_lengthes.append(min(label.end - label.begin, self.phrase_length_max) - 1)
                    phrase_tags.append((bid, label))

        results = [[] for _ in range(len(labels))]
        if len(context_features) > 0:
            context_length_emb = self.length_embedding(torch.LongTensor(context_lengthes, device=device))
            context_features = self.context2ffn(
                torch.cat([context_length_emb, torch.stack(context_features, dim=0)], dim=-1))
            scores, indexes = self.loss.predict(context_features, 5)
            for (bid, label), topk in zip(context_tags, indexes):
                results[bid].append((label, topk.tolist()))

            phrase_length_emb = self.length_embedding(torch.LongTensor(phrase_lengthes, device=device))
            phrase_features = self.phrase2ffn(
                torch.cat([phrase_length_emb, torch.stack(phrase_features, dim=0)], dim=-1))
            scores, indexes = self.loss.predict(phrase_features, 5)
            for (bid, label), topk in zip(phrase_tags, indexes):
                results[bid].append((label, topk.tolist()))

        return results

    def named_embeddings(self):
        yield self.name, self.loss.hidden2label.weight, self.loss.voc.itos

    def _span_embed(self,forwards: torch.Tensor, backwards: torch.Tensor, begin: int, end: int):
        return torch.cat([forwards[begin-1], forwards[end-1],
                          backwards[end], backwards[begin]], dim=-1)


class LMClassifier(nn.Module):
    def __init__(self, voc_size, voc_dim, hidden_dim, tied_weight: nn.Embedding=None, padding_idx=-1, dropout=0.3):
        super(LMClassifier, self).__init__()
        self.voc_size = voc_size
        self.voc_dim = voc_dim
        self.hidden_dim = hidden_dim
        self.padding_idx = padding_idx
        self.context_ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, voc_dim),
            nn.Tanh()
        )

        self.context2token = nn.Linear(voc_dim, voc_size)
        if tied_weight is not None:
            self.context2token.weight = tied_weight

    def forward(self, forwards: torch.Tensor, backwards: torch.Tensor, tokens: torch.Tensor):
        seq_len, batch_size, dim = forwards.size()
        assert dim == self.hidden_dim
        pad = forwards.new_zeros(1, batch_size, self.hidden_dim)
        features = torch.cat(
            [torch.cat([pad, backwards[1:2]], dim=-1),
             torch.cat([forwards[:-2], backwards[2:]], dim=-1),
             torch.cat([forwards[-2:-1], pad], dim=-1)],
            dim=0)
        features = self.context2token(self.context_ffn(features))
        return F.cross_entropy(features.view(-1, self.voc_size), tokens.view(-1), ignore_index=self.padding_idx)


class PhraseClassifier(nn.Module):
    def __init__(self, hidden_dim, max_length=15, dropout=0.3):
        super(PhraseClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, forwards: torch.Tensor, backwards: torch.Tensor,
                lens: torch.Tensor, phrases: List[List[Tuple[int, int]]]) -> torch.Tensor:

        samples, features, targets = self._make_sample(forwards, backwards, lens, phrases)
        if len(samples) == 0:
            return torch.tensor([0.0], device=forwards.device)

        return F.binary_cross_entropy(self.ffn(features), targets.unsqueeze(-1))

    def predict(self, forwards: torch.Tensor, backwards: torch.Tensor,
                lens: torch.Tensor, phrases: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int, float, float]]]:

        samples, features, targets = self._make_sample(forwards, backwards, lens, phrases)
        if len(samples) == 0:
            return [[] for _ in range(len(phrases))]

        preds = self.ffn(features)

        targets = targets.tolist()
        preds = preds.squeeze().tolist()
        results = [[] for _ in range(len(phrases))]
        for id, (bid, begin, end) in enumerate(samples):
            results[bid].append((begin, end, targets[id], preds[id]))
        return results

    def find_phrase(self, forwards: torch.Tensor, backwards: torch.Tensor, lens: torch.Tensor,
                    threshold=0.8) -> List[List[Tuple[int, int, float]]]:
        phrases = []
        for bid in range(lens.size(0)):
            samples = []
            features = []
            for begin in range(1, lens[bid] - 1):
                for step in range(2, min(self.max_length+1, lens[bid] - begin)):
                    samples.append((begin, begin+step))
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
                     phrases: List[List[Tuple[int, int]]]) -> Tuple[List[Tuple[int, int, int]], torch.Tensor, torch.Tensor]:

        samples, targets = [], []
        for bid, sen_phrases in enumerate(phrases):
            for begin, end in sen_phrases:
                if end - begin > 1:
                    samples.append((bid, begin, end))
                    targets.append(1)
                    for mid in range(begin, end):
                        for n_begin, n_end in [(random.randint(max(0, begin-self.max_length), begin-1), mid),
                                               (mid, random.randint(end+1, min(lens[bid], end+self.max_length)))]:
                            if (0 <= n_begin < begin < n_end < end) or (begin < n_begin < end < n_end < lens[bid]):
                                samples.append((bid, n_begin, n_end))
                                targets.append(0)

        if len(samples) > 0:
            features = torch.stack(
                [self._span_embed(forwards[:, bid], backwards[:, bid], begin, end)
                 for bid, begin, end in samples], dim=0)
            targets = torch.tensor(targets, dtype=torch.float, device=forwards.device)
        else:
            features = torch.tensor([], device=forwards.device)
            targets = torch.tensor([], device=forwards.device)

        return samples, features, targets

    def _span_embed(self, forwards: torch.Tensor, backwards: torch.Tensor, begin: int, end: int):
        return torch.cat([forwards[begin-1], forwards[end-1],
                          backwards[end], backwards[begin]], dim=-1)