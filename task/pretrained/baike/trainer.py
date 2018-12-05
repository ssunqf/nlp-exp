#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import collections
import os
import json
import time
import random
from typing import Dict, Tuple

import torch
from torch import nn, optim
from tqdm import tqdm
from torchtext.vocab import Vocab

from .model import Model
from .data import BaikeDataset, Field, LabelField, lazy_iter
from .encoder import LSTMEncoder, StackLSTM
from .classifier import LabelClassifier, PhraseClassifier

from tensorboardX import SummaryWriter


class Task:
    def __init__(self, text_vocab: Vocab, label_vocabs: Dict[str, Vocab], model: Model):
        self.model = model
        self.text_vocab = text_vocab
        self.label_vocabs = label_vocabs

    def verbose(self, data):


    def _pretty_print(self, batch, label_results, phrases, find_phrases=None):
        text, lens = batch.text
        for bid in range(lens.size(0)):
            text_str = [self.text_vocab.itos[w] for w in text[:lens[bid], bid]]
            print()
            print(text_str)
            for name, result in label_results.items():
                for label, pred in result[bid]:
                    gold = set(self.label_vocabs[name].itos[i] for i in label.tags.tolist())
                    pred = set(self.label_vocabs[name].itos[i] for i in pred)
                    print('(%d,%d,%s): (%s, %s, %s)' % (
                        label.begin, label.end,
                        ''.join(text_str[label.begin:label.end]), name, gold, pred))

            for begin, end, gold, pred in phrases[bid]:
                print('(%d,%d,%s): (%f, %f)' % (begin, end, ''.join(text_str[begin:end]), gold, pred))

            if find_phrases:
                print()
                for begin, end, prob in find_phrases[bid]:
                    print('(%d,%d,%s): prob=%f' % (begin, end, ''.join(text_str[begin:end]), prob))


class Trainer:
    def __init__(self, config, model: Model,
                 dataset_it, text_voc: Vocab, label_vocabs: Dict[str, Vocab],
                 valid_step, checkpoint_dir):
        self.config = config
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), 1e-3, weight_decay=1e-6)

        self.dataset_it = dataset_it

        self.text_voc = text_voc
        self.label_vocabs = label_vocabs

        self.valid_step = valid_step
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_files = collections.deque()

        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir)

    def state_dict(self, train=True, optimizer=True):

        states = collections.OrderedDict()
        states['model'] = self.model.state_dict()
        if optimizer:
            states['optimizer'] = self.optimizer.state_dict()
        # if train:
        #    states['train_it'] = self.train_it.state_dict()

        return states

    def load_state_dict(self, states, strict):

        self.model.load_state_dict(states['model'], strict=strict)
        if 'optimizer' in states:
            self.optimizer.load_state_dict(states['optimizer'])

            # if 'train_it' in states:
            #    self.train_it.load_state_dict(states['train_it'])

    def load_checkpoint(self, path, strict=True):
        states = torch.load(path)
        self.load_state_dict(states, strict=strict)

    def train_one(self, batch) -> Tuple[Dict[str, float], int]:
        self.model.train()
        self.model.zero_grad()

        losses, batch_size = self.model(batch)
        rloss = {n: l.item() for n, l in losses.items()}

        loss = sum(loss for name, loss in losses.items())
        if loss.requires_grad is False:
            return rloss, batch_size

        loss.backward()

        # Step 3. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()
        return rloss, batch_size

    def valid(self, valid_it) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            losses = collections.defaultdict(float)
            count = 0
            with tqdm(total=len(valid_it.dataset), desc='valid') as valid_tqdm:
                for _, valid_batch in enumerate(valid_it):
                    _losses, batch_size = self.model(valid_batch)
                    valid_tqdm.update(batch_size)

                    for n, l in _losses.items():
                        losses[n] += l.item()
                    count += 1
            return {n: l/count for n, l in losses.items()}

    def metrics(self, valid_it):
        self.model.eval()
        acc, pre, recall = [collections.defaultdict(float) for _ in range(3)]
        counter = collections.defaultdict(float)

        phrase_correct, phrase_total = 0, 0
        with torch.no_grad():
            with tqdm(total=len(valid_it.dataset), desc='metrics') as valid_tqdm:
                for _, valid_batch in enumerate(valid_it):
                    results, phrases, batch_size = self.model.predict(valid_batch)
                    valid_tqdm.update(batch_size)

                    for name, result in results.items():
                        for sen_res in result:
                            for label, pred in sen_res:
                                if label.tags.size(0) > 0:
                                    counter[name] += 1
                                    gold = set(label.tags.tolist())
                                    pred = set(pred)
                                    inter = gold.intersection(pred)
                                    union = gold.union(pred)
                                    acc[name] += len(inter) / len(union)
                                    pre[name] += len(inter) / len(pred)
                                    recall[name] += len(inter) / len(gold)

                    for sps in phrases:
                        for _, _, gold, pred in sps:
                            gold, pred = gold > 0.5, pred > 0.5
                            if gold == pred:
                                phrase_correct += 1
                            phrase_total += 1

                    def pretty_print(batch, label_results, phrases, find_phrases=None):
                        text, lens = batch.text
                        for bid in range(lens.size(0)):
                            text_str = [self.text_voc.itos[w] for w in text[:lens[bid], bid]]
                            print()
                            print(text_str)
                            for name, result in label_results.items():
                                for label, pred in result[bid]:
                                    gold = set(self.label_vocabs[name].itos[i] for i in label.tags.tolist())
                                    pred = set(self.label_vocabs[name].itos[i] for i in pred)
                                    print('(%d,%d,%s): (%s, %s, %s)' % (
                                        label.begin, label.end,
                                        ''.join(text_str[label.begin:label.end]), name, gold, pred))

                            for begin, end, gold, pred in phrases[bid]:
                                print('(%d,%d,%s): (%f, %f)' % (begin, end, ''.join(text_str[begin:end]), gold, pred))

                            if find_phrases:
                                print()
                                for begin, end, prob in find_phrases[bid]:
                                    print('(%d,%d,%s): prob=%f' % (begin, end, ''.join(text_str[begin:end]), prob))

                    if random.random() < 0.005:
                        find_phrases = self.model.list_phrase(valid_batch)
                        self.pretty_print(valid_batch, results, phrases, find_phrases)

            scores = {n: {'acc': acc[n]/c, 'pre': pre[n]/c, 'recall': recall[n]/c} for n, c in counter.items()}
            scores['phrase'] = {'pre': phrase_correct/phrase_total}
            print(scores)
            return scores

    def pretty_print(self, batch, label_results, phrases, find_phrases=None):
        text, lens = batch.text
        for bid in range(lens.size(0)):
            text_str = [self.text_voc.itos[w] for w in text[:lens[bid], bid]]
            print()
            print(text_str)
            for name, result in label_results.items():
                for label, pred in result[bid]:
                    gold = set(self.label_vocabs[name].itos[i] for i in label.tags.tolist())
                    pred = set(self.label_vocabs[name].itos[i] for i in pred)
                    print('(%d,%d,%s): (%s, %s, %s)' % (
                        label.begin, label.end,
                        ''.join(text_str[label.begin:label.end]), name, gold, pred))

            for begin, end, gold, pred in phrases[bid]:
                print('(%d,%d,%s): (%f, %f)' % (begin, end, ''.join(text_str[begin:end]), gold, pred))

            if find_phrases:
                print()
                for begin, end, prob in find_phrases[bid]:
                    print('(%d,%d,%s): prob=%f' % (begin, end, ''.join(text_str[begin:end]), prob))

    def train(self):
        total_batch, start = 1e-10, time.time()
        label_losses = collections.defaultdict(float)
        num_iterations = 0
        for train_it, valid_it in tqdm(self.dataset_it, desc='dataset'):
            with tqdm(total=len(train_it.dataset), desc='train') as train_tqdm:
                for batch in train_it:
                    num_iterations += 1
                    losses, batch_size = self.train_one(batch)
                    train_tqdm.update(batch_size)

                    for n, l in losses.items():
                        label_losses[n] += l

                    total_batch += 1

                    if num_iterations % self.valid_step == 0:

                        valid_losses = self.valid(valid_it)
                        total_valid_loss = sum(l for n, l in valid_losses.items()) / len(valid_losses)

                        self.checkpoint(total_valid_loss)

                        label_losses = {label: (loss/total_batch) for label, loss in label_losses.items()}

                        if len(label_losses) > 1:
                            self.summary_writer.add_scalars(
                                'loss', {'train_mean_loss': sum(l for _, l in label_losses.items())/len(label_losses)},
                                num_iterations)
                        self.summary_writer.add_scalars(
                            'loss', {('train_%s_loss' % n) : l for n, l in label_losses.items()},
                            num_iterations)

                        if len(label_losses) > 1:
                            self.summary_writer.add_scalars(
                                'loss', {'valid_mean_loss': total_valid_loss},
                                num_iterations)
                        self.summary_writer.add_scalars(
                            'loss', {('valid_%s_loss' % n) : l for n, l in valid_losses.items()},
                            num_iterations)
                        total_batch, start = 1e-10, time.time()
                        label_losses = collections.defaultdict(float)

                    if train_it.iterations % (self.valid_step) == 0:
                        for n, scores in self.metrics(valid_it).items():
                            self.summary_writer.add_scalars(
                                'eval', {('%s_%s' % (n, sn)) : s for sn, s in scores.items()},
                                num_iterations)

                # reset optimizer
                # if self.train_it.iterations > self.config.warmup_step:
                #    self.optimizer = optim.Adam(self.model.parameters(), 1e-3, weight_decay=1e-3)

    def checkpoint(self, valid_loss):
        torch.save(self.state_dict(), '%s/model-last' % (self.checkpoint_dir))

        if len(self.checkpoint_files) == 0 or self.checkpoint_files[-1][0] > valid_loss:
            checkpoint_file = '%s/model-%.4f' % (self.checkpoint_dir, valid_loss)
            torch.save(self.state_dict(), checkpoint_file)
            self.checkpoint_files.append((valid_loss, checkpoint_file))

        while len(self.checkpoint_files) > 5:
            loss, file = self.checkpoint_files.popleft()
            os.remove(file)
