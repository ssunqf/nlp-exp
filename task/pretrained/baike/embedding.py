#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import torch
from torch import nn

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
