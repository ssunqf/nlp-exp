# -*- coding:utf-8 -*-
# Filename: make_ctb.py
# Author：hankcs
import argparse
import subprocess
from typing import List
from nltk import Tree
from os import listdir, makedirs
from os.path import isfile, join, isdir, exists

from nltk.corpus import brown


def split(ctb_root):
    chtbs = [f for f in listdir(ctb_root) if isfile(join(ctb_root, f)) and f.startswith('chtb')]
    folder = {}
    for f in chtbs:
        tag = f[-6:-4]
        if tag not in folder:
            folder[tag] = []
        folder[tag].append(f)
    train, dev, test = [], [], []
    for tag, files in folder.items():
        t = int(len(files) * .8)
        d = int(len(files) * .9)
        train += files[:t]
        dev += files[t:d]
        test += files[d:]
    return train, dev, test


def to_nltk_tree(lines: List[str]):
    chunks = []
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            yield Tree('S', chunks)
            chunks = []
            continue
        _, _, _, ct, wt, word, *_ = line.split()

        leaf = Tree(wt, [word])
        if ct.startswith('B-'):
            chunks.append(Tree(ct[2:], [leaf]))
        elif ct.startswith('I-'):
            chunks[-1].append(leaf)
        else:
            chunks.append(Tree(ct, [leaf]))


def ctb_to_chunk(files, ctb_root, out_path):
    print('Generating ' + out_path)

    with open(out_path, 'w') as out:
        for id, file in enumerate(files):
            file = join(ctb_root, file)
            print('processing %s' % file)
            res = subprocess.run(
                ['perl', './task/pretrained/chunk/chunklinkctb.pl', '-ns', file],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if res.returncode != 0:
                print(res.stdout)
                print(res.stderr)
                exit(-1)

            lines = res.stdout.decode('utf-8', errors='ignore').split('\n')

            if id > 0:
                for tree in to_nltk_tree(lines[2:]):
                    out.write(tree.pformat(margin=1e10) + '\n')


def reformat(files, ctb_root, out_path):
    print('Generating ' + out_path)

    with open(out_path, 'w') as out:
        for id, file in enumerate(files):
            file = join(ctb_root, file)
            print('processing %s' % file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine Chinese Treebank 8 bracketed files into train/dev/test set')
    parser.add_argument("--ctb", required=True,
                        help='The root path to Chinese Treebank 8')
    parser.add_argument("--output", required=True,
                        help='The folder where to store the output train.txt/dev.txt/test.txt')

    args = parser.parse_args()

    training, development, test = split(args.ctb)

    root_path = args.output
    if not exists(root_path):
        makedirs(root_path)
    ctb_to_chunk(training, args.ctb, join(root_path, 'train.txt'))
    ctb_to_chunk(development, args.ctb, join(root_path, 'dev.txt'))
    ctb_to_chunk(test, args.ctb, join(root_path, 'test.txt'))

