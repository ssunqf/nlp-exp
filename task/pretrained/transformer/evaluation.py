#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from typing import List


def evaluation_one(preds: List[str], golds: List[str]):
    correct = 0
    true = 0
    begin = 0
    while begin < len(golds):
        end = begin
        while end < len(golds):
            if golds[end].startswith('S_') or golds[end].startswith('E_') or golds[end] in ['O', '*']:
                end += 1
                break
            else:
                end += 1

        if not golds[begin].endswith('*'):
            true += 1
            if preds[begin:end] == golds[begin:end]:
                correct += 1

        begin = end

    return correct, true


if __name__ == '__main__':

    pred = ['S_', 'S_', 'S_', 'B_', 'E_', 'S_']
    gold = ['S_', 'S_', 'B_', 'M_', 'E_', 'S_']

    print(evaluation_one(pred, gold))
