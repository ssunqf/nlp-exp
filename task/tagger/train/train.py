#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import time
import random
import os
import collections
from typing import Tuple
import torch
from torch import optim
from torchtext import data
from torchtext.data.iterator import BucketIterator
from task.tagger.train.model import Tagger
from tqdm import tqdm
from task.tagger.train.dataset import TaggerDataset
from task.tagger.train.field import PartialField
from module import utils


class Trainer:
    def __init__(self, model: Tagger, device: torch.device,
                 partial_train_it,
                 partial_valid_it,
                 partial_test_it,
                 valid_step, checkpoint_dir):

        self.model = model
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters())

        self.partial_train_it, self.partial_valid_it, self.partial_test_it = \
            partial_train_it, partial_valid_it, partial_test_it
        self.valid_step = valid_step
        self.checkpoint_dir = checkpoint_dir

    def state_dict(self, train=True, optimizer=True):

        states = collections.OrderedDict()

        states['model'] = self.model.state_dict()
        if optimizer:
            states['optimizer'] = self.optimizer.state_dict()
        if train:
            states['partial_train_it'] = self.partial_train_it.state_dict()
        return states

    def load_state_dict(self, states, strict):

        self.model.load_state_dict(states['model'], strict=strict)

        if 'optimizer' in states:
            self.optimizer.load_state_dict(states['optimizer'])

        if 'partial_train_it' in states:
            self.partial_train_it.load_state_dict(states['partial_train_it'])

    def load_checkpoint(self, path, strict=True):
        states = torch.load(path)
        self.load_state_dict(states, strict=strict)

    def train_one(self, batch, scale) -> Tuple[float, int]:
        self.model.train()
        self.model.zero_grad()

        loss, num_sen = self.model.criterion(batch)
        rloss = loss.item()
        (loss * scale).div(num_sen).backward()

        # Step 3. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 5)
        self.optimizer.step()
        return rloss, num_sen

    def valid(self, valid_it) -> float:
        with torch.no_grad():
            valid_loss = 0
            valid_tokens = 0

            sample = False
            for _, valid_batch in enumerate(valid_it):
                loss, num_token = self.model.criterion(valid_batch)

                valid_loss += loss.item()
                valid_tokens += num_token

                if not sample and random.random() < 0.02:
                    self.model.sample(valid_batch)

            return valid_loss / valid_tokens

    def train(self):
        partial_total_loss, partial_total_sen, start = 0, 1e-10, time.time()
        checkpoint_losses = collections.deque()
        for step, batch in tqdm(enumerate(self.partial_train_it, start=1), total=len(self.partial_train_it)):

            loss, num_sen = self.train_one(batch, scale=1.0)
            partial_total_loss += loss
            partial_total_sen += num_sen

            if step % self.valid_step == 0:
                partial_valid_loss = self.valid(self.partial_valid_it)
                self.checkpoint(checkpoint_losses, partial_valid_loss)

                print("partail: train loss=%.6f\t\tvalid loss=%.6f" % (partial_total_loss/partial_total_sen, partial_valid_loss))
                print("speed:   %.2f sentence/s\n\n" % (partial_total_sen/(time.time()-start)))

                partial_total_loss, partial_total_sen, start = 0, 1e-10, time.time()

            if self.partial_train_it.iterations % len(self.partial_train_it) == 0:
                with torch.no_grad():
                    print(self.model.evaluation(self.partial_test_it))

    def checkpoint(self, checkpoint_losses, valid_loss):
        if len(checkpoint_losses) == 0 or checkpoint_losses[-1] > valid_loss:

            if os.path.exists(self.checkpoint_dir) is False:
                os.mkdir(self.checkpoint_dir)

            checkpoint_losses.append(valid_loss)

            torch.save(self.state_dict(),
                       '%s/ctb-pos-%0.4f' % (self.checkpoint_dir, valid_loss))

            if len(checkpoint_losses) > 5:
                removed = checkpoint_losses.popleft()
                os.remove('%s/ctb-pos-%0.4f' % (self.checkpoint_dir, removed))

    @classmethod
    def create(cls, config):

        text_field = data.Field(include_lengths=True)
        tag_field = PartialField()

        partial_train, partial_valid, partial_test = TaggerDataset.splits([text_field, tag_field],
                                                                          prefix=config.partial_prefix,
                                                                          train=config.train,
                                                                          valid=config.valid,
                                                                          test=config.test)

        text_field.build_vocab(partial_train, min_freq=config.text_min_freq)
        tag_field.build_vocab(partial_train, min_freq=config.tag_min_freq)

        partial_train_it, partial_valid_it, partial_test_it = \
            BucketIterator.splits([partial_train, partial_valid, partial_test],
                                  batch_sizes=config.batch_sizes,
                                  sort_key=TaggerDataset.sort_key,
                                  device=0 if config.use_cuda else -1)

        device = torch.device("cuda" if config.use_cuda else "cpu")
        model = Tagger(
                       text_field.vocab, tag_field.vocab,
                       config.embedding_dim, config.pos_dim, config.hidden_mode, config.hidden_dim, config.num_layers,
                       True, config.num_heads, config.dropout)
        model = model.to(device)
        return Trainer(model, device, partial_train_it, partial_valid_it, partial_test_it,
                       config.valid_step, config.checkpoint_path)


class Config:
    partial_prefix = './partial_ne/entity'

    train = '.train'
    valid = '.valid'
    test = '.test'

    batch_sizes = [16, 16, 16]

    # vocabulary
    text_min_freq = 10
    # text_min_size = 50000

    tag_min_freq = 10
    # tag_min_size = 50000

    common_size = 1000

    # model
    embedding_dim = 100
    pos_dim = 50
    hidden_mode = 'GRU'
    hidden_dim = 300
    num_layers = 2
    dropout = 0.2
    num_heads = 10
    assert hidden_dim % num_heads == 0

    use_cuda = torch.cuda.is_available()

    checkpoint_path = './partial_ne/summary.seg.model'

    valid_step = 500


import argparse

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')

args = arg_parser.parse_args()

config = Config()

trainer = Trainer.create(config)

if args.checkpoint is not None:
    trainer.load_checkpoint(args.checkpoint)

trainer.train()
