#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import six
from torchtext import data
from torchtext.data import Dataset
import torch
from typing import List
from task.transformer.vocab import TagVocab


class PartialField(data.Field):
    def __init__(self, **kwargs):
        self.with_type = False
        super(PartialField, self).__init__(**kwargs)
        self.vocab_cls = TagVocab

    def build_vocab(self, *args, **kwargs):
        from collections import Counter, OrderedDict
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                x = [i for i in x if not i.endswith('*') and i != 'O']
                counter.update(x)
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None))

        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def pad(self, minibatch):
        """Pad a batch of examples using this field.

        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True` and `self.sequential` is `True`, else just
        returns the padded list. If `self.sequential` is `False`, no padding is applied.
        """
        minibatch = list(minibatch)
        if not self.sequential:
            return minibatch
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch:
            padded.append(
                ([] if self.init_token is None else [self.init_token]) +
                list(x[:max_len]) +
                ([] if self.eos_token is None else [self.eos_token]) +
                [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return (padded, lengths)
        return padded

    def numericalize(self, arr, device=None, train=True):
        """Turn a batch of examples that use this field into a Variable.

        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.

        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (-1 or None): Device to create the Variable's Tensor on.
                Use -1 for CPU and None for the currently active GPU device.
                Default: None.
            train (boolean): Whether the batch is for a training set.
                If False, the Variable will be created with volatile=True.
                Default: True.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.LongTensor(lengths)

        def _dense_mask(seqs: List[List[str]], batch_first: bool):
            batch_size = len(seqs)
            assert batch_size > 0
            seq_size = len(seqs[0])

            masks = torch.ByteTensor(batch_size, seq_size, len(self.vocab)).zero_() \
                if batch_first else torch.ByteTensor(seq_size, batch_size, len(self.vocab)).zero_()
            for i, seq in enumerate(seqs):
                for j, tok in enumerate(seq):
                    if tok != '*':
                        id = self.vocab.stoi[tok]
                        if batch_first:
                            masks[i, j] = 1
                            masks[i, j, id] = 0
                        else:
                            masks[j, i] = 1
                            masks[j, i, id] = 0
            return masks

        masks = self.vocab.dense_mask(arr, self.batch_first)
        if device == -1:
            masks = masks.contiguous()
        else:
            masks = masks.cuda(device)
        return masks, arr