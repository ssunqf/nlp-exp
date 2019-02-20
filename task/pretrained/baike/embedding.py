#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F

from .encoder import ElmoEncoder


class WindowEmbedding(nn.Module):
    def __init__(self, voc_size, embed_size, max_window_size: int=5, padding_idx=None, dropout=0.3):
        super(WindowEmbedding, self).__init__()
        self.token_embed = nn.Embedding(voc_size, embed_size, padding_idx=padding_idx)

        self.windows = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Conv1d(embed_size, embed_size, window_size, padding=window_size//2),
                nn.Tanh())
            for window_size in range(1, max_window_size+1, 2)])

        self.max_pool = nn.MaxPool1d((max_window_size+1)//2, stride=(max_window_size+1)//2)

    def forward(self, input: torch.Tensor, batch_first=False):

        embed = self.token_embed(input)
        embed = embed.permute(0, 2, 1) if batch_first else embed.permute(1, 2, 0)

        N, C, L = embed.size()

        output = torch.stack([window(embed) for window in self.windows], dim=-1).view(N, C, -1)

        output = self.max_pool(output)

        output = output.permute(0, 2, 1) if batch_first else output.permute(2, 0, 1)

        return output


class ElmoEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, elmo_encoder: ElmoEncoder, mode='top'):
        super(ElmoEmbedding, self).__init__()

        self.embedding = embedding
        self.elmo_encoder = elmo_encoder
        self.mode = mode

    def forward(self, tokens):
        embed = self.embedding(tokens)
        forwards, backwards = self.elmo_encoder(embed)

        if self.mode == 'top':
            return torch.cat([forwards[-1], backwards[-1]], dim=-1)


class CompressedEmbedding(nn.Module):
    def __init__(self, num_embeddings, pretrained_dim, pretrained_weight, compressed_dim, freeze=True, padding_idx=None):
        super(CompressedEmbedding, self).__init__()
        self.num_embeddings = num_embeddings

        self.pretrained_dim = pretrained_dim
        self.pretrained_weight = nn.Parameter(pretrained_weight, requires_grad=not freeze)

        assert list(pretrained_weight.shape) == [num_embeddings, pretrained_dim], \
            'Shape of weight does not match num_embeddings and pretrained_dim'

        # assert compressed_dim < pretrained_dim, \
        #    'compressed_dim should be smaller than pretrained_dim'

        self.compressed_dim = compressed_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx

        if self.compressed_dim == self.pretrained_dim:
            self.compress_layer = nn.Sequential()
        else:
            self.compress_layer = nn.Sequential(
                nn.Linear(pretrained_dim, compressed_dim),
                nn.Hardtanh()
            )

    def forward(self, input):
        if self.training:
            if hasattr(self, 'compressed_weight'):
                delattr(self, 'compressed_weight')
            pretrained = F.embedding(input, self.pretrained_weight, self.padding_idx)

            return self.compress_layer(pretrained)
        else:
            if not hasattr(self, 'compressed_weight'):
                self.compressed_weight = self.compress_layer(self.pretrained_weight)
            return F.embedding(input, self.compressed_weight, self.padding_idx)

    @classmethod
    def from_pretrained(cls, path, compressed_dim, freeze=True, sparse=False):

        with open(path) as input:
            rows, cols = [int(i) for i in input.readline().split()]
            vocab = []
            embeddings = []
            for line in input:
                word, *vec = line.rsplit(sep=' ', maxsplit=cols)
                assert len(vec) == cols, line

                vocab.append(word)
                embeddings.append([float(i) for i in vec])

            embeddings = torch.FloatTensor(embeddings)

            embedding = cls(
                num_embeddings=rows,
                pretrained_dim=cols,
                pretrained_weight=embeddings,
                compressed_dim=compressed_dim,
                freeze=freeze
            )
            return vocab, embedding

