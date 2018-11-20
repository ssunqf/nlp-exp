#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import torch
from task.pretrained.transformer import Encoder, LinearCRF
from module.encoder import Encoder

encoder = Encoder(20, 'CNN', 20, 5, 10)

crf = LinearCRF(20, 10)

tests = torch.randn(10, 20).unsqueeze(-2).expand(10, 2, 20)
lens = [10, 10]
tags = torch.randint(0, 10, (10, 1), dtype=torch.int).expand(10, 2)
encoder.eval()
print(encoder(tests[:, :1], torch.ones(10, 1, dtype=torch.int8)))
print(encoder(tests[:, 1:2], torch.ones(10, 1, dtype=torch.int8)))
hiddens = encoder(tests, torch.ones(10, 2, dtype=torch.int8))
print(hiddens[:, 0, :5])
print(hiddens[:, 1, :5])
pred, score = crf(encoder(tests, torch.ones(10, 2, dtype=torch.int8)), lens)

hiddens = encoder(tests, torch.ones(10, 2, dtype=torch.int8))

masks = torch.ones(10, 2, 10, dtype=torch.int8)
for b in range(tags.size(1)):
    for t in range(tags.size(0)):
        masks[t, b, tags[t, b]] = 0

bloss = crf.neg_log_likelihood(hiddens, lens, masks, tags)

sloss = crf.valid_neg_log_likelihood(hiddens, lens, masks, tags)
print(bloss, sloss)
print(bloss[0].backward())

print(crf(hiddens, lens))
print(crf.nbest(hiddens, lens, 5))