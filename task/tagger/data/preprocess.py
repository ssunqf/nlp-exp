#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import sys
from task.util import utils

if __name__ == '__main__':
    with open(sys.argv[1]) as reader, open(sys.argv[2], 'w') as writer:
        for sentence in reader:
            chars = []
            tags = []
            for left in sentence.split('[[['):
                for field in left.split(']]]'):
                    field = field.split('|||')
                    t_chars = [word for t, word in utils.replace_entity(field[0].strip())]
                    if len(field) == 2 and len(field[0]) > 0:
                        t_tags = ['S_'] if len(t_chars) == 1 else \
                            ['B_'] + ['M_'] * (len(t_chars) - 2) + ['E_']
                    else:
                        t_tags = ['*'] * len(t_chars)

                    chars.extend(t_chars)
                    tags.extend(t_tags)

            writer.write(' '.join([char + '#' + tag for char, tag in zip(chars, tags)]) + '\n')


