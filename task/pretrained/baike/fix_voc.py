#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import json
import gzip
from collections import Counter, OrderedDict
import sys
from task.util.utils import fix_word


if __name__ == '__main__':
    with gzip.open(sys.argv[1], mode='rt', compresslevel=6) as file:
        counter = Counter(dict(json.loads(file.read())))

        new_counter = Counter()
        for word, count in counter.most_common():
            if count < 200:
                word = fix_word(word)

            new_counter[word] += count

        print('new vocab size = %d' % len(new_counter))
        for word, count in new_counter.most_common():
            print('%s\t%d' % (word, count))



