#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import os
from typing import List
from torchtext import data


class TaggerDataset(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.text), len(ex.tags))

    def __init__(self, path: str, fields: List,
                 **kwargs):

        examples = []

        with open(path) as file:
            for line in file:
                items = line.strip().split()
                if 0 < len(items) < 150:
                    items = [t.rsplit('#', maxsplit=1) for t in items]
                    tokens = [t[0] for t in items]
                    tags = [t[1] for t in items]
                    examples.append(data.Example.fromlist([tokens, tags], fields))

        super(TaggerDataset, self).__init__(examples, fields, **kwargs)


