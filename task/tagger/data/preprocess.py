#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import argparse
import sys
from task.util import utils
import os

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--input', type=str, help='input path')
arg_parser.add_argument('--output', type=str, help='output path')


def load_dict(path: str):
    assert os.path.exists(path)
    with open(path, 'r') as f:
        return set([line.strip() for line in f if len(line.strip()) > 0])


dicts = [('PLACE', load_dict('./dic/place.dic')),
         ('NAME',  load_dict('./dic/name.dic')),
         ('O',     load_dict('./dic/common.dic'))]


if __name__ == '__main__':

    args = arg_parser.parse_args()
    with open(args.input) as reader, open(args.output, 'w') as writer, open(args.dict) as dict_file:
        for sentence in reader:
            chars = []
            tags = []
            for left in sentence.split('[[['):
                for field in left.split(']]]'):
                    field = field.split('|||')
                    field[0] = field[0].strip()
                    t_chars = [word for t, word in utils.replace_entity(field[0])]
                    if len(field) == 2 and len(field[0]) > 1:
                        contain = False
                        for name, dict in dicts:
                            if field[0] in dict and field[1].endswith(field[0]):
                                t_tags = ['S_'] if len(t_chars) == 1 else \
                                    ['B_'] + ['M_'+name] * (len(t_chars)-2) + ['E_']
                                contain = True
                                break

                        if not contain:
                            t_tags = ['S_*'] if len(t_chars) == 1 else \
                                ['B_*'] + ['*'] * (len(t_chars) - 2) + ['E_*']
                    else:
                        t_tags = ['*'] * len(t_chars)

                    chars.extend(t_chars)
                    tags.extend(t_tags)

            writer.write(' '.join([char + '#' + tag for char, tag in zip(chars, tags)]) + '\n')


