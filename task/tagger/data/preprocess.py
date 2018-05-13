#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import argparse
import sys
from task.util import utils

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--input', type=str, help='input path')
arg_parser.add_argument('--output', type=str, help='output path')
arg_parser.add_argument('--dict', type=str, help='common dict')

if __name__ == '__main__':

    args = arg_parser.parse_args()
    with open(args.input) as reader, open(args.output, 'w') as writer, open(args.dict) as dict_file:
        dict = set([line.strip() for line in dict_file.readlines()])
        for sentence in reader:
            chars = []
            tags = []
            for left in sentence.split('[[['):
                for field in left.split(']]]'):
                    field = field.split('|||')
                    t_chars = [word for t, word in utils.replace_entity(field[0].strip())]
                    if len(field) == 2 and len(field[0]) > 1:
                        if field[0].strip() in dict:
                            t_tags = ['S_'] if len(t_chars) == 1 else \
                                ['B_'] + ['M_'] * (len(t_chars) - 2) + ['E_']
                        else:
                            t_tags = ['S_'] if len(t_chars) == 1 else \
                                ['B_'] + ['*'] * (len(t_chars) - 2) + ['E_']
                    else:
                        t_tags = ['*'] * len(t_chars)

                    chars.extend(t_chars)
                    tags.extend(t_tags)

            writer.write(' '.join([char + '#' + tag for char, tag in zip(chars, tags)]) + '\n')


