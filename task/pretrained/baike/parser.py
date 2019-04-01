#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple
import torch
from torch import nn

from .base import make_masks
from .attention import MultiHeadedAttention


class SpanScorer(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(SpanScorer, self).__init__()

        self.hidden_size = hidden_size
        self.num_labels = num_labels

        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.num_labels, self.num_labels)
        )

    def forward(self, hiddens: torch.Tensor, lens: torch.Tensor) -> Tuple[List[Dict[Tuple[int, int], int]], List[torch.Tensor]]:
        charts, spans = self._make_chart(hiddens, lens)

        scores = self.scorer(spans).split([len(chart) for chart in charts])
        charts = [{span: i for i, span in enumerate(chart)} for chart in charts]
        return charts, scores

    def _make_chart(self, hiddens, lens):
        spans = []
        charts = []
        for bid, seq_len in enumerate(lens):
            charts.append([])
            for length in range(1, seq_len):
                for left in range(0, seq_len + 1 - length):
                    right = left + length
                    charts[-1].append((left, right))
                    spans.append(torch.cat((hiddens[right, bid, :self.hidden_size//2] - hiddens[left, bid, :self.hidden_size//2],
                                            hiddens[left+1, bid, self.hidden_size//2:] - hiddens[right+1, bid, self.hidden_size//2:]), dim=-1))
        spans = torch.stack(spans, dim=0)

        return charts, spans


class Parser(nn.Module):
    def __init__(self,
                 embedding: nn.Embedding,
                 encoder: nn.LSTM,
                 attention: MultiHeadedAttention,
                 scorer: SpanScorer):
        super(Parser, self).__init__()

        self.embedding = embedding
        self.encoder = nn.LSTM
        self.attention = attention
        self.scorer = SpanScorer

    def forward(self, input, lens, trees):

        embed = self.embedding(input)
        hidden, _ = self.encoder(embed)
        hidden, _ = self.attention(hidden, hidden, hidden, mask=make_masks(input, lens))

        charts, scores  = self.scorers(hidden, lens)

        sum(self.loss(score, chart, root) for chart, _scores, root in zip(charts, scores, trees)


        loss = sum(self.loss(_spans, tree) for _spans, tree in zip(spans, trees))

        return loss

    def loss(self, seq_spans, chart, root):
        loss = 0
        for node in root.traverse():
            loss += seq_spans[]

        return loss
