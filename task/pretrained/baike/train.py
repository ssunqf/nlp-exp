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
from .classifier import LabelClassifier

from tensorboardX import SummaryWriter


class Trainer:
    def __init__(self, config, model: Model,
                 dataset_it, text_voc: Vocab, label_vocabs: Dict[str, Vocab],
                 valid_step, checkpoint_dir):
        self.config = config
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), 1e-3, weight_decay=1e-3)

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

        sum(loss for name, loss in losses.items()).backward()

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
        with torch.no_grad():
            with tqdm(total=len(valid_it.dataset), desc='metrics') as valid_tqdm:
                for _, valid_batch in enumerate(valid_it):
                    results, batch_size = self.model.predict(valid_batch)
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

                    def pretty_print(batch, results):
                        text, lens = valid_batch.text

                        for bid in range(lens.size(0)):
                            text_str = [self.text_voc.itos[w] for w in text[:lens[bid], bid]]
                            print()
                            print(text_str)
                            for name, result in results.items():
                                for label, pred in result[bid]:
                                    gold = set(self.label_vocabs[name].itos[i] for i in label.tags.tolist())
                                    pred = set(self.label_vocabs[name].itos[i] for i in pred)
                                    print('(%d,%d,%s): (%s, %s, %s)' % (
                                        label.begin, label.end, ''.join(text_str[label.begin:label.end]), name, gold, pred))

                    if random.random() < 0.005:
                        pretty_print(valid_batch, results)

            scores = {n: {'acc': acc[n]/c, 'pre': pre[n]/c, 'recall': recall[n]/c} for n, c in counter.items()}

            return scores

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

                        self.summary_writer.add_scalars(
                            'loss', {'train_mean_loss': sum(l for _, l in label_losses.items())/len(label_losses)},
                            num_iterations)
                        self.summary_writer.add_scalars(
                            'loss', {('train_%s_loss' % n) : l for n, l in label_losses.items()},
                            num_iterations)

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
        torch.save(self.state_dict(), '%s/pretrained-last' % (self.checkpoint_dir))

        if len(self.checkpoint_files) == 0 or self.checkpoint_files[-1][0] > valid_loss:
            checkpoint_file = '%s/pretrained-%.4f' % (self.checkpoint_dir, valid_loss)
            torch.save(self.state_dict(), checkpoint_file)
            self.checkpoint_files.append((valid_loss, checkpoint_file))

        while len(self.checkpoint_files) > 5:
            loss, file = self.checkpoint_files.popleft()
            os.remove(file)

    @classmethod
    def load_voc(cls, config):
        TEXT = Field(include_lengths=True, init_token='<s>', eos_token='</s>')
        KEY_LABEL = LabelField('keys')
        ATTR_LABEL = LabelField('attrs')
        SUB_LABEL = LabelField('subtitles')
        TEXT.build_vocab(os.path.join(config.root, 'text.voc.gz'), max_size=50000, min_freq=50)
        KEY_LABEL.build_vocab(os.path.join(config.root, 'key.voc.gz'), min_freq=50)
        ATTR_LABEL.build_vocab(os.path.join(config.root, 'attr.voc.gz'), min_freq=50)
        SUB_LABEL.build_vocab(os.path.join(config.root, 'subtitle.voc.gz'), min_freq=50)

        print('text vocab size = %d' % len(TEXT.vocab))
        print('key vocab size = %d' % len(KEY_LABEL.vocab))
        print('attr vocab size = %d' % len(ATTR_LABEL.vocab))
        print('subtitle vocab size = %d' % len(SUB_LABEL.vocab))

        return TEXT, KEY_LABEL, ATTR_LABEL, SUB_LABEL

    @classmethod
    def load_dataset(cls, config, fields):
        # train_it, valid_it = BaikeDataset.iters(
        #    fields, path=config.root, train=config.train_file, device=config.device)
        # train_it.repeat = True
        def batch_size_fn(new, count, sofar):
            return sofar + (len(new.text) + 99)//100

        dataset_it = lazy_iter(fields, config.train_file,
                               batch_size=config.batch_size,
                               path=config.root,
                               batch_size_fn=batch_size_fn,
                               device=config.device)
        return dataset_it

    @classmethod
    def load_model(cls, config, text_field, key_field, attr_field, sub_field):

        embedding = nn.Embedding(len(text_field.vocab), config.embedding_size)

        encoder = StackLSTM(config.embedding_size,
                            config.encoder_size, config.encoder_num_layers,
                            residual=False, dropout=0.2)

        classifiers = nn.ModuleList([
            LabelClassifier(name, config.classifier_loss, field.vocab, config.encoder_size, config.label_size, config.attention_num_heads)
            for name, field in [('keys', key_field), ('attrs', attr_field), ('subtitles', sub_field)]])

        model = Model(embedding, encoder, classifiers)

        model.to(config.device)

        return model

    @classmethod
    def create(cls, config, checkpoint=None):
        TEXT, KEY_LABEL, ATTR_LABEL, SUB_LABEL = cls.load_voc(config)

        fields = [('text', TEXT), (('keys', 'attrs', 'subtitles'), (KEY_LABEL, ATTR_LABEL, SUB_LABEL))]

        # train_it, valid_it = BaikeDataset.iters(
        #    fields, path=config.root, train=config.train_file, device=config.device)
        # train_it.repeat = True

        dataset_it = cls.load_dataset(config, fields)

        model = cls.load_model(config, TEXT, KEY_LABEL, ATTR_LABEL, SUB_LABEL)
        trainer = cls(config, model, dataset_it,
            TEXT.vocab, {'keys': KEY_LABEL.vocab, 'attrs': ATTR_LABEL.vocab, 'subtitles': SUB_LABEL.vocab},
            config.valid_step, config.checkpoint_dir)
        if checkpoint:
            trainer.load_checkpoint(checkpoint)
        return trainer


class Config:
    def __init__(self):
        self.root = './baike/preprocess3'
        self.train_file = 'sentence.url'

        self.embedding_size = 256
        self.encoder_size = 512
        self.encoder_num_layers = 2
        self.attention_num_heads = 8

        self.label_size = 512

        self.valid_step = 500
        self.warmup_step = 5000

        self.batch_size = 32

        self.dir_prefix = 'subsample-sum-res'
        self.checkpoint_dir = os.path.join(self.root, self.dir_prefix, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.summary_dir = os.path.join(self.root, self.dir_prefix, 'summary')
        os.makedirs(self.summary_dir, exist_ok=True)

        self.classifier_loss = 'adaptivesoftmax' # ['softmax', 'negativesample', 'adaptivesoftmax]

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # save config
        with open(os.path.join(self.root, self.dir_prefix, 'config.txt'), 'wt') as file:
            file.write(json.dumps({name: param for name, param in self.__dict__.items() if name not in {'device'}},
                                  ensure_ascii=False, indent=2))


if __name__ == '__main__':

    trainer = Trainer.create(Config())

    trainer.train()
