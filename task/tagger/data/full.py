#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from task.util import utils

import functools

import argparse

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--input', type=str, help='input file')
arg_parser.add_argument('--output', type=str, help='output file')

args = arg_parser.parse_args()

with_type = None
with open(args.input) as input, open(args.output, 'w') as output:
    for line in input:
        line = line.strip()
        if len(line) > 0:
            words = line.split()
            if with_type is None:
                with_type = functools.reduce(lambda x, y: x and y, [w.find('_') != -1 for w in words])

            if with_type:
                word2type = [w.rsplit('_', 1) for w in words]
                chars2type = [(list(utils.replace_entity(word)), tag) for word, tag in word2type]
                text = [(type, char) for chars, _ in chars2type for type, char in chars]
                text = [char if type == '@zh_char@' else type for type, char in text]
                tags = [tag for chars, type in chars2type for tag in utils.BMESTagger.tag(len(chars), type)]

            else:
                words = [[char if type == '@zh_char@' else type for type, char in utils.replace_entity(word)]
                         for word in words
                         ]
                label = [utils.BMESTagger.tag(len(chars)) for chars in words]

                text = [c for w in words for c in w]
                tags = [t for word in label for t in word]

            output.write(' '.join([w + '#' + t for w, t in zip(text, tags)]) + '\n')