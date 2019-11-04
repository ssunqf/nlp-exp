#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from tabulate import tabulate
from torchtext.vocab import Vocab
from typing import List, Tuple
import torch
from .base import BOS, EOS, PAD


class TagVocab(Vocab):
    def __init__(self, counter, **kwargs):
        super(TagVocab, self).__init__(counter, **kwargs)

        print(counter.most_common())

        self.begin_mask = torch.ByteTensor(
            [1 if s.startswith('B_') else 0 for i, s in enumerate(self.itos)])
        self.end_mask = torch.ByteTensor(
            [1 if s.startswith('E_') else 0 for i, s in enumerate(self.itos)])
        self.middle_mask = torch.ByteTensor(
            [1 if s.startswith('M_') else 0 for i, s in enumerate(self.itos)])
        self.single_mask = torch.ByteTensor(
            [1 if s.startswith('S_') else 0 for i, s in enumerate(self.itos)])
        self.outer_mask = torch.ByteTensor(
            [1 if s.endswith('_O') or s == 'O' else 0 for i, s in enumerate(self.itos)])

        self.transition_constraints = torch.ones(len(self), len(self), dtype=torch.uint8)
        for i, si in enumerate(self.itos):
            for j, sj in enumerate(self.itos):
                if si == EOS and sj != PAD:
                    self.transition_constraints[j, i] = 0
                elif si == BOS and (sj[:2] in ['M_', 'E_']):
                    self.transition_constraints[j, i] = 0
                elif sj == EOS and si[:2] in ['B_', 'M_']:
                    self.transition_constraints[j, i] = 0
                elif si[:2] in ['S_', 'E_'] and sj[:2] in ['M_', 'E_']:
                    self.transition_constraints[j, i] = 0
                elif si[:2] in ['B_', 'M_'] and (sj[:2] in ['S_', 'B_'] or (sj[:2] in ['M_', 'E_'] and si[2:] != sj[2:])):
                    self.transition_constraints[j, i] = 0
        # print('trainstion constraints')
        # print(tabulate([self.itos] + self.transition_constraints.tolist(), headers="firstrow", tablefmt='grid'))

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
                elif tok in self.stoi:
                    tid = self.stoi[tok]
                    masks[bid, sid] = 0
                    masks[bid, sid, tid] = 1
            self.check_valid(masks[bid])

        if batch_first is False:
            masks = masks.transpose(0, 1)
        return masks

    def is_split(self, tag: Tuple[str, int]) -> bool:
        if isinstance(tag, int):
            assert tag < len(self.itos)
            tag = self.itos[id]
        return tag.startswith('E_') or tag.startswith('S_')


class BracketVocab(Vocab):
    def __init__(self, counter, **kwargs):
        super(BracketVocab, self).__init__(counter, **kwargs)

        self.masks = {}
        for parent in ['B', 'M', 'E', 'S']:
            self.masks['*%s*' % parent] = torch.ByteTensor(
                [1 if parent in s else 0 for i, s in enumerate(self.itos)]
            )
            for child in ['B', 'M', 'E', 'S']:
                self.masks['*%s**%s*' % (parent, child)] = torch.ByteTensor(
                    [1 if parent in s and child in s and s.find(parent) < s.rfind(child) else 0
                     for i, s in enumerate(self.itos)]
                )

        self.transition_constraints = torch.ones(len(self), len(self), dtype=torch.uint8)
        for i, si in enumerate(self.itos):
            for j, sj in enumerate(self.itos):
                if si == EOS and sj != PAD:
                    self.transition_constraints[j, i] = 0
                elif si == BOS and len(set(sj).intersection({'M', 'E'})) > 0:
                    self.transition_constraints[j, i] = 0
                elif sj == EOS and len(set(si).intersection({'B', 'M'})) > 0:
                    self.transition_constraints[j, i] = 0
                elif len(si) < len(sj) and len(set(sj[len(si):]).intersection({'M', 'E'})) > 0:
                    self.transition_constraints[j, i] = 0
                elif len(si) > len(sj) and len(set(si[len(sj):]).intersection({'B', 'M'})) > 0:
                    self.transition_constraints[j, i] = 0
                else:
                    for height in range(min(len(si), len(sj))):
                        if si[height] in ['S', 'E'] and sj[height] in ['M', 'E']:
                            self.transition_constraints[j, i] = 0
                            continue
                        elif si[height] in ['B', 'M'] and sj[height] in ['S', 'B']:
                            self.transition_constraints[j, i] = 0
                            continue
                # elif si[0] in ['B', 'M'] and abs(len(si) - len(sj)) > 1:
                #     self.transition_constraints[j, i] = 0
        # print('trainstion constraints')
        # print(tabulate([self.itos] + self.transition_constraints.tolist(), headers="firstrow", tablefmt='grid'))

    def check_valid(self, mask):
        seq_len, num_label = mask.size()
        assert num_label == len(self)

    def dense_mask(self, seqs: List[List[str]], batch_first: bool, device=None):
        batch_size = len(seqs)
        assert batch_size > 0
        seq_size = len(seqs[0])

        mask_flag = False
        masks = torch.zeros(batch_size, seq_size, len(self), dtype=torch.int8, device=device)
        for bid, tags in enumerate(seqs):
            for sid, tag in enumerate(tags):
                if '*' in tag:
                    mask_flag = True
                if tag in self.masks:
                    masks[bid, sid] = self.masks.get(tag)
                elif tag in self.stoi:
                    tid = self.stoi[tag]
                    masks[bid, sid, tid] = 1
                elif tag == '*':
                    masks[bid, sid] = 1
                else:
                    raise RuntimeError('%s is not exist.' % tag)

            self.check_valid(masks[bid])

        if batch_first is False:
            masks = masks.transpose(0, 1)
        return masks, mask_flag

    def is_split(self, tag: Tuple[str, int]) -> bool:
        if isinstance(tag, int):
            assert tag < len(self.itos)
            tag = self.itos[id]
        return tag.startswith('E') or tag.startswith('S')



