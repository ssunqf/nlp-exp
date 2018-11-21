#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import json
import gzip
from collections import Counter, OrderedDict
import sys
from task.util.utils import fix_word, replace_entity


if __name__ == '__main__':
    with gzip.open(sys.argv[1], mode='rt', compresslevel=6) as file:
        for line in file:
            print(line)
            print('\t'.join('%s:%s' % (t, c) for t, c in replace_entity(line)))
