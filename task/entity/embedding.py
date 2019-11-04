#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import sys
import torch
import time

tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertTokenizer', 'bert-base-chinese', do_basic_tokenize=False)

# Tokenized input
text = "作者为了保住可读性，会想尽各种办法，用很多黑魔法去实现一些高难度的设计，让使用者永远都能感受到它优雅简单的魅力。"
tokenized_text = ['[CLS]'] + tokenizer.tokenize(text) + ['[SEP]']
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)


### Get the hidden states computed by `bertModel`
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0] * len(tokenized_text)

# Convert inputs to PyTorch tensors
segments_tensors = torch.tensor([segments_ids])
tokens_tensor = torch.tensor([indexed_tokens])

model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertModel', 'bert-base-chinese')
model.eval()

with open(sys.argv[1]) as input:
    sentences = input.readlines()

with torch.no_grad():
    start = time.time()
    total = 0
    for sentence in sentences:
        tokenized_text = ['[CLS]'] + tokenizer.tokenize(sentence) + ['[SEP]']
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * len(tokenized_text)
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
        total += len(tokenized_text) - 2
    end = time.time()
    print('speed: %f char/s' % (total / (end - start)))
