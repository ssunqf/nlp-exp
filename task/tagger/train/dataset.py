#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import os
from typing import List
from torchtext import data


class TaggerDataset(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.text), len(ex.tags))

    def __init__(self, path: str, fields: List[data.Field],
                 **kwargs):
        if not isinstance(fields[0], (tuple, list)):
            fields = [('html2text', fields[0]), ('tags', fields[1])]

        examples = []

        with open(path) as file:
            for line in file:
                items = line.strip().split()
                if 0 < len(items) < 150:
                    items = [t.rsplit('#', maxsplit=1) for t in items]
                    tokens = [t[0] for t in items]
                    tags = [t[1][0:2] for t in items]
                    examples.append(data.Example.fromlist([tokens, tags], fields))

        super(TaggerDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, fields, prefix='./',
               train='.train', valid='.valid', test='.test', **kwargs):
        """Create dataset objects for splits of a TranslationDataset.

        Arguments:
            fields: A tuple containing the fields that will be used for data
                in each language.
            prefix: The prefix of data.
            partial_train: The  of the train data. Default: 'train'.
            valid: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """

        train_data = None if train is None else cls(
            prefix + train, fields, **kwargs)
        val_data = None if valid is None else cls(
            prefix+valid, fields, **kwargs)
        test_data = None if test is None else cls(
            prefix+test, fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


