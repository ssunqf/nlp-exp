#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, file_utils

model = BertModel.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

