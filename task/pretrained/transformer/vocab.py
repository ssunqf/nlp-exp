#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from torchtext.vocab import Vocab
from typing import List, Tuple
import torch
from .base import BOS, EOS, PAD


class TagVocab(Vocab):
    def __init__(self, counter, **kwargs):
        super(TagVocab, self).__init__(counter, **kwargs)
        self.begin_mask = torch.tensor([1 if s.startswith('B_') else 0 for i, s in enumerate(self.itos)], dtype=torch.int8)
        self.end_mask = torch.tensor([1 if s.startswith('E_') else 0 for i, s in enumerate(self.itos)], dtype=torch.int8)
        self.middle_mask = torch.tensor([1 if s.startswith('M_') else 0 for i, s in enumerate(self.itos)], dtype=torch.int8)
        self.single_mask = torch.tensor([1 if s.startswith('S_') else 0 for i, s in enumerate(self.itos)], dtype=torch.int8)
        self.outer_mask = torch.tensor([1 if s.endswith('_O') or s == 'O' else 0 for i, s in enumerate(self.itos)], dtype=torch.int8)

        self.transition_constraints = torch.ones(len(self), len(self), dtype=torch.uint8)
        for i, si in enumerate(self.itos):
            for j, sj in enumerate(self.itos):
                if si[:2] in ['S_', 'E_'] and sj[:2] in ['M_', 'E_']:
                    self.transition_constraints[j, i] = 0
                elif si[:2] in ['B_', 'M_'] and (sj[:2] in ['S_', 'B_'] or si[2:] != sj[2:]):
                    self.transition_constraints[j, i] = 0
                elif si == BOS and (sj[:2] in ['M_', 'E_', PAD]):
                    self.transition_constraints[j, i] = 0
                elif si[:2] in ['B_', 'M_'] and sj == EOS:
                    self.transition_constraints[j, i] = 0
        print('trainstion constraints')
        print(self.transition_constraints)

    def check_valid(self, mask):
        seq_len, num_label = mask.size()
        assert num_label == len(self)

    def dense_mask(self, seqs: List[List[str]], batch_first: bool, device=None):
        batch_size = len(seqs)
        assert batch_size > 0
        seq_size = len(seqs[0])

        masks = torch.ones(batch_size, seq_size, len(self), dtype=torch.int8, device=device)
        for bid, seq in enumerate(seqs):
            for sid, tok in enumerate(seq):
                if tok == 'B_*':
                    masks[bid, sid] = self.begin_mask
                elif tok == 'M_*':
                    masks[bid, sid] = self.middle_mask
                elif tok == 'E_*':
                    masks[bid, sid] = self.end_mask
                elif tok == 'S_*':
                    masks[bid, sid] = self.single_mask
                elif tok == 'O':
                    masks[bid, sid] = self.outer_mask
                elif tok != '*':
                    tid = self.stoi[tok]
                    masks[bid, sid] = 0
                    masks[bid, sid, tid] = 1
            self.check_valid(masks[bid])

        if batch_first is False:
            masks = masks.transpose(0, 1)
        return masks

    def is_split(self, tag: Tuple[str, id]) -> bool:
        if isinstance(tag, id):
            assert id < len(self.itos)
            tag = self.itos[id]
        return tag.startswith('E_') or tag.startswith('S_')

