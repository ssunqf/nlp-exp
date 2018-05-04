#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
from torchtext.vocab import Vocab
from typing import List, Tuple
import torch


class TagVocab(Vocab):
    def __init__(self, counter, **kwargs):
        super(TagVocab, self).__init__(counter, **kwargs)
        self.begin_mask = torch.ByteTensor([1 if s.startswith('B_') else 0 for i, s in enumerate(self.itos)])
        self.end_mask = torch.ByteTensor([1 if s.startswith('E_') else 0 for i, s in enumerate(self.itos)])
        self.middle_mask = torch.ByteTensor([1 if s.startswith('M_') else 0 for i, s in enumerate(self.itos)])
        self.single_mask = torch.ByteTensor([1 if s.startswith('S_') else 0 for i, s in enumerate(self.itos)])
        self.outer_mask = torch.ByteTensor([1 if s.endswith('_O') else 0 for i, s in enumerate(self.itos)])

        self.begin_constraints = (self.begin_mask | self.single_mask) == 0
        # self.begin_constraints.masked_fill_((self.begin_mask | self.single_mask) == 0, 1)

        self.end_constraints = (self.end_mask | self.single_mask) == 0
        # self.end_constraints.masked_fill_((self.end_mask | self.single_mask) == 0, 1)

        self.transition_constraints = torch.zeros(len(self), len(self), dtype=torch.uint8)
        for i, si in enumerate(self.itos):
            for j, sj in enumerate(self.itos):
                if si.startswith('S_') or si.startswith('E_'):
                    if sj.startswith('M_') or sj.startswith('E_'):
                        self.transition_constraints[j, i] = 1
                elif si.startswith('B_') or si.startswith('M_'):
                    if si[2:] != sj[2:] or sj.startswith('S_') or sj.startswith('B_'):
                        self.transition_constraints[j, i] = 1

    def check_valid(self, mask):
        seq_len, num_label = mask.size()
        assert num_label == len(self)
        for i in range(seq_len):
            if i == 0:
                assert ((mask[i] == 0) | (self.begin_constraints == 0)).sum() > 0
            elif i == seq_len - 1:
                assert ((mask[i] == 0) | (self.end_constraints == 0)).sum() > 0

    def dense_mask(self, seqs: List[List[str]], batch_first: bool) -> torch.ByteTensor:
        batch_size = len(seqs)
        assert batch_size > 0
        seq_size = len(seqs[0])

        masks = torch.ByteTensor(batch_size, seq_size, len(self)).zero_() \
            if batch_first else torch.ByteTensor(seq_size, batch_size, len(self)).zero_()
        for bid, seq in enumerate(seqs):
            curr_mask = masks[bid] if batch_first else masks[:, bid]
            for sid, tok in enumerate(seq):
                if tok == 'B_*':
                    curr_mask[sid].masked_fill_(self.begin_mask == 0, 1)
                elif tok == 'M_*':
                    curr_mask[sid].masked_fill_(self.middle_mask == 0, 1)
                elif tok == 'E_*':
                    curr_mask[sid].masked_fill_(self.end_mask == 0, 1)
                elif tok == 'S_*':
                    curr_mask[sid].masked_fill_(self.single_mask == 0, 1)
                elif tok == 'O':
                    curr_mask[sid].masked_fill_(self.outer_mask == 0, 1)
                elif tok != '*':
                    tid = self.stoi[tok]
                    curr_mask[sid, :] = 1
                    curr_mask[sid, tid] = 0
            self.check_valid(curr_mask)
        return masks

    def is_split(self, tag: Tuple[str, id]) -> bool:
        if isinstance(tag, id):
            assert id < len(self.itos)
            tag = self.itos[id]
        return tag.startswith('E_') or tag.startswith('S_')