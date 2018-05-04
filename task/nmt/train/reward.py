#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import collections

from typing import List


class Reward:
    def score(self, refer: List[int], cands: List[List[int]]) -> List[float]:
        raise NotImplementedError

class GLEU(Reward):
    def __init__(self, num_grams=4):
        self.num_grams = num_grams

    def make_gram(self, sen: List[int]):
        for l in range(1, self.num_grams+1, 1):
            for b in range(len(sen) - l):
                yield ' '.join(map(str, sen[b:b+l]))

    def score(self, gold: List[int], preds: List[List[int]]) -> List[float]:
        gold_counter = collections.Counter(self.make_gram(gold))
        gold_total = sum(gold_counter.values())

        scores = []
        for pred in preds:
            pred_counter = collections.Counter(self.make_gram(pred))
            pred_total = sum(pred_counter.values())
            match_count = 0
            for k, v in pred_counter.items():
                match_count += min(gold_counter[k], v)

            gleu = min(match_count/gold_total, match_count/pred_total)
            scores.append(gleu)

        return scores



