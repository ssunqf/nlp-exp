#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import sys



def read_sentence(path):
    with open(sys.argv[1]) as input:
        sentence = []
        for line in input:
            line = line.strip()
            if len(line) == 0 and len(sentence) > 0:
                yield sentence
                sentence = []

            sentence.append(line.split('\t'))

        if len(sentence) > 0:
            yield sentence

def to_flat(sentence):




if __name__ == '__main__':

    with open(sys.argv[1]) as input:
        sentence = []
        for line in input:
            line = line.strip()
            if len(line) == 0 and len(sentence) > 0:

