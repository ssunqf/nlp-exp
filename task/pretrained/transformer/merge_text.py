#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import os
import sys


def iter_dir(path):
    if os.path.isdir(path):
        for name in os.listdir(path):
            sub_path = os.path.join(path, name)
            yield from iter_dir(sub_path)
    else:
        yield path


for path in iter_dir(sys.argv[1]):
    with open(path) as file:
        for line in file:
            line = line.strip()
            if len(line) > 0:
                print(line)