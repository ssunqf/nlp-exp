#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from typing import List

import torch
import torch.nn as nn
from torchtext import data
from torchtext.vocab import Vocab
from task.tagger.train.vocab import TagVocab

from module import Encoder, PartialCRF, TimeSignal, MultiHead

import re

eng_num = re.compile('[a-zA-Z0-9]+')


class Tagger(nn.Module):
    def __init__(self, text_vocab: Vocab, tag_vocab: TagVocab,
                 embed_dim, hidden_mode, hidden_dim, hidden_layers, atten_heads=1, num_blocks=1, dropout=0.2):
        super(Tagger, self).__init__()

        self.text_vocab = text_vocab
        self.tag_vocab = tag_vocab

        self.embed = nn.Embedding(len(text_vocab), embed_dim, padding_idx=text_vocab.stoi['<pad>'])
        self.time_signal = TimeSignal(embed_dim)
        self.encoder = Encoder(embed_dim,
                               hidden_mode, hidden_dim, hidden_layers,
                               atten_heads=atten_heads,
                               num_blocks=num_blocks,
                               dropout=dropout)

        self.crf = PartialCRF(hidden_dim,
                              len(tag_vocab), tag_vocab.begin_constraints,
                              tag_vocab.end_constraints, tag_vocab.transition_constraints,
                              dropout)

    def criterion(self, batch: data.Batch):
        _, text_lens = batch.text
        masks, tags = batch.tags
        feats = self.feature(batch)

        return self.crf.criterion(feats, masks, text_lens)

    def feature(self, batch: data.Batch):

        text, text_lens = batch.text
        golds = batch.tags
        embed = self.embed(text) + self.time_signal(text)

        hidden = self.encoder(embed, text_lens)

        return hidden

    def predict(self, batch: data.Batch):

        self.eval()
        _, text_len = batch.text
        feats = self.feature(batch)

        return self.crf(feats, text_len)

    def print(self, batch: data.Batch):
        text, text_len = batch.text
        gold_masks, gold_tags = batch.tags
        for i in range(len(text_len)):
            length = text_len[i]
            print(' '.join([self.text_vocab.itos[w] + '#' + t
                            for w, t in zip(text[0:length, i].data.tolist(), gold_tags[i])]))

    def sample(self, batch: data.Batch):

        self.eval()
        text, text_len = batch.text
        gold_masks, gold_tags = batch.tags
        pred_tags = self.predict(batch)

        results = []
        for i in range(len(pred_tags)):
            score, pred_tag = pred_tags[i]
            length = text_len[i]
            sen = [self.text_vocab.itos[w] for w in text[0:length, i].data.tolist()]

            def tostr(words: List[str], tags: List[int]):
                tags = [self.tag_vocab.itos[tag_id] for tag_id in tags]
                prev_alnum = False
                for word, tag in zip(words, tags):
                    if tag == 'E_O' or tag == 'S_O':
                        yield word + ' '
                    elif tag == 'B_O':
                        yield word
                    elif tag.startswith('B_'):
                        yield '{{%s:%s' % (tag[2:], word)
                    elif tag.startswith('E_'):
                        yield word + '}} '
                    elif tag.startswith('S_'):
                        yield '{{%s:%s}} ' % (tag[2:], word)
                    else:
                        yield word

                    prev_alnum = word.isalnum()

            pred_tag = ''.join(tostr(sen, pred_tag))
            #gold_tag = ''.join([tostr(w, id) for w, id in zip(sen, gold_tags[0:length, i].data)])

            # if pred_tag != gold_tag:
            # print('\ngold: %s\npred: %s\nscore: %f' % (gold_tag, pred_tag, score))
            print('\npred: %s\nscore: %f' % (pred_tag, score))

        return results

    @staticmethod
    def evaluation_one(self, pred: List[int], gold: torch.Tensor):
        correct = 0
        true = 0

        for curr in range(0, len(pred)):
            tag_list = [i for i in range(gold.size(-1)) if gold[curr, i].data == 1]
            if len(tag_list) == 1:
                true += 1
                gold_tag = tag_list[0]
                if pred[curr] == gold_tag:
                    correct += 1

        return correct, true

    def evaluation(self, data_it):
        self.eval()
        correct, true, pos = 0, 0, 0
        for batch in data_it:
            _, text_len = batch.text
            golds = batch.tags
            preds = self.predict(batch)

            #print(gold_tags)
            for i in range(len(text_len)):
                score, pred = preds[i]
                gold = golds[0:text_len[i], i]
                #c, t = self.evaluation_one(pred, gold)
                #correct += c
                #true += t

        recall = correct/float(true+1e-5)
        return {'recall':recall}

    def coarse_params(self):
        yield from self.embed.parameters()
        yield from self.time_signal.parameters()
        yield from self.encoder.parameters()
        yield from self.crf.parameters()

    def fine_params(self):
        yield from self.embed.parameters()
        yield from self.time_signal.parameters()
        yield from self.crf.parameters()
