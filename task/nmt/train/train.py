#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import collections

import random
import time
import os

from torchtext.vocab import Vocab
from torchtext.data import Batch, Field, BucketIterator
from torchtext.datasets import TranslationDataset

from module.utils import *
import torch
from torch import nn, optim

from .model import Seq2Seq


def embedding(embed: nn.Embedding, indices: torch.Tensor, padding_idx, use_cuda):
    _weight = embed.weight.index_select(0, indices)
    _backend = embed._backend
    if use_cuda:
        weight = _weight.cuda()

    def _forward(input: torch.Tensor):
        return _backend.Embedding.apply(
            input, _weight,
            padding_idx, embed.max_norm, embed.norm_type,
            embed.scale_grad_by_freq, embed.sparse
        )

    return _forward


def linear(linear: nn.Linear, indices: torch.Tensor, use_cuda):
    weight = linear.weight.index_select(0, indices)
    if linear.bias is None:
        bias = None
    else:
        bias = linear.bias.index_select(0, indices)

    if use_cuda:
        weight = weight.cuda()
        bias = None if bias is None else bias.cuda()

    def _forward(input: torch.Tensor):
        return F.linear(input, weight, bias)

    return _forward

class PoolState:
    def __init__(self, source: Seq2Seq, src_counter: collections.Counter, trg_counter: collections.Counter, use_cuda):

        self.source = source
        src_counter = collections.Counter({source.src_voc.itos[i]: c for i, c in src_counter.items()})
        self.src_voc = Vocab(src_counter, specials=source.src_voc.itos[0:source.config.common_size])

        trg_counter = collections.Counter({source.trg_voc.itos[i]: c for i, c in trg_counter.items()})
        self.trg_voc = Vocab(trg_counter, specials=source.trg_voc.itos[0:source.config.common_size])

        src_s2b = [source.src_voc.stoi[w] for w in self.src_voc.itos]
        self.src_b2s = {b: s for s, b in enumerate(src_s2b)}

        trg_s2b = [source.trg_voc.stoi[w] for w in self.trg_voc.itos]
        self.trg_b2s = {b: s for s, b in enumerate(trg_s2b)}

        self.src_s2b = torch.LongTensor(src_s2b)
        self.trg_s2b = torch.LongTensor(trg_s2b)

        self.pad_index = self.trg_voc.stoi[PAD]
        self.sos_index = self.trg_voc.stoi[SOS]
        self.eos_index = self.trg_voc.stoi[EOS]

        self.use_cuda = use_cuda

    def forward(self, batch):
        src_embed = embedding(self.source.src_embed, self.src_s2b, self.src_voc.stoi[PAD], self.use_cuda)
        trg_embed = embedding(self.source.trg_embed, self.trg_s2b, self.trg_voc.stoi[PAD], self.use_cuda)

        output_embed = linear(self.source.output_embed, self.trg_s2b, self.use_cuda)

        if self.use_cuda:
            src_embed = src_embed.cuda()
            trg_embed = trg_embed.cuda()
            output_embed = output_embed.cuda()

        return self._transform(batch), src_embed, trg_embed, output_embed

    def _transform(self, batch):

        src, src_lens = batch.src
        src_size = src.size()
        src = torch.LongTensor([self.src_b2s[i] for i in src.data.view(-1).tolist()]).view(src_size)

        trg, trg_lens = batch.trg
        trg_size = trg.size()
        trg = torch.LongTensor([self.trg_b2s[i] for i in trg.data.view(-1).tolist()]).view(trg_size)

        if self.use_cuda:
            src = src.cuda()
            trg = trg.cuda()

        return Batch.fromvars(batch.dataset, batch.batch_size, batch.train, src=(src, src_lens), trg=(trg, trg_lens))


def pool(data: Generator[Batch, Any, None], model: Seq2Seq, pool_size, use_cuda):

    def pool_fun(it):
        _pool = []
        src_counter = collections.Counter()
        trg_counter = collections.Counter()
        for batch in data:
            src_counter.update(batch.src[0].view(-1).data)
            trg_counter.update(batch.trg[0].view(-1).data)
            _pool.append(batch)
            if len(_pool) >= pool_size:
                yield _pool, src_counter, trg_counter

                _pool = []
                src_counter = collections.Counter()
                trg_counter = collections.Counter()

        if len(_pool) > 0:
            yield _pool, src_counter, trg_counter

    for pool_data, src_counter, trg_counter in pool_fun(data):
        pool_state = PoolState(model, src_counter, trg_counter, use_cuda)

        for batch in pool_data:
            yield batch, pool_state



class Trainer:
    def __init__(self, model: Seq2Seq, train_it, valid_it, test_it, valid_step, checkpoint_dir, pool_size):

        self.model = model

        self.model = model
        self.optimizer = optim.Adam(self.model.parameters())

        self.train_it, self.valid_it, self.test_it = train_it, valid_it, test_it
        self.valid_step = valid_step
        self.checkpoint_dir = checkpoint_dir

        self.pool_size = pool_size

    def state_dict(self, train=True, optimizer=True):

        states = collections.OrderedDict()

        states['model'] = self.model.state_dict()
        if optimizer:
            states['optimizer'] = self.optimizer.state_dict()
        if train:
            states['train_it'] = self.train_it.state_dict()

        return states

    def load_state_dict(self, states):

        self.model.load_state_dict(states['model'])

        if 'optimizer' in states:
            self.optimizer.load_state_dict(states['optimizer'])

        if 'train_it' in states:
            self.train_it.load_state_dict(states['train_it'])

    def load(self, path):
        states = torch.load(path)
        self.load_state_dict(states)

    def train(self, use_cuda=False):
        total_loss, total_token, start = 0, 0, time.time()
        checkpoint_losses = collections.deque()

        for step, (batch, pool_state) in enumerate(pool(self.train_it, self.model, self.pool_size, use_cuda), start=1):
            self.model.train()
            self.model.zero_grad()

            src, src_lens = batch.src
            trg, trg_lens = batch.trg
            if src_lens[0] > 100:
                continue

            loss, num_token = self.model.rl_loss(batch, pool_state)

            total_loss += loss.data[0]
            total_token += num_token

            loss.div(num_token).backward()

            # Step 3. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 5)
            self.optimizer.step()

            if step % self.valid_step == 0:
                self.model.eval()
                valid_loss = 0
                valid_tokens = 0

                sample = False
                for _, (valid_batch, valid_pool) in enumerate(pool(self.valid_it, self.model, self.pool_size, use_cuda)):
                    loss, num_token = self.model.rl_loss(valid_batch, valid_pool)

                    valid_loss += loss.data[0]
                    valid_tokens += num_token

                    if not sample and random.random() < 0.02:
                        self.model.sample(valid_batch, valid_pool)

                valid_loss = valid_loss/valid_tokens
                self.checkpoint(checkpoint_losses, valid_loss)

                print('train loss: %.6f\t\tvalid loss: %.6f\t\tspeed: %.2f tokens/s'
                      %(total_loss/total_token, valid_loss, total_token/(time.time()-start)))

                total_loss, total_token, start = 0, 0, time.time()

    def checkpoint(self, checkpoint_losses, valid_loss):
        if len(checkpoint_losses) == 0 or checkpoint_losses[-1] > valid_loss:

            if os.path.exists(self.checkpoint_dir) is False:
                os.mkdir(self.checkpoint_dir)

            checkpoint_losses.append(valid_loss)

            torch.save(self.state_dict(),
                       '%s/nmt-model-loss-%0.4f' % (self.checkpoint_dir, valid_loss))

            if len(checkpoint_losses) > 5:
                removed = checkpoint_losses.popleft()
                os.remove('%s/nmt-model-loss-%0.4f' % (self.checkpoint_dir, removed))


    @classmethod
    def create(cls, config):

        src_field = Field(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', include_lengths=True)

        trg_field = Field(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', lower=True,
                               include_lengths=True)

        train = TranslationDataset(path=config.train_prefix,
                                   exts=config.exts,
                                   fields=(src_field, trg_field))
        valid = TranslationDataset(path=config.valid_prefix,
                                   exts=config.exts,
                                   fields=(src_field, trg_field))

        test = TranslationDataset(path=config.test_prefix,
                                  exts=config.exts,
                                  fields=(src_field, trg_field))

        train_it, valid_it, test_it = BucketIterator.splits([train, valid, test],
                                                                 batch_sizes=config.batch_sizes,
                                                                 sort_key=TranslationDataset.sort_key,
                                                                 device=-1)

        src_field.build_vocab(train, min_freq=10)
        trg_field.build_vocab(train, min_freq=10)

        src_voc = src_field.vocab
        trg_voc = trg_field.vocab

        model = Seq2Seq.create(src_voc, trg_voc, config)

        if config.use_cuda:
            model = model.cuda()

        return Trainer(model, train_it, valid_it, test_it, config.valid_step, config.checkpoint_path, config.pool_size)


import argparse

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')


args = arg_parser.parse_args()

class Config:

    # data
    corpus_prefix =  './mt-data/ted2013'

    train_prefix = corpus_prefix + '.train'
    valid_prefix = corpus_prefix + '.valid'
    test_prefix = corpus_prefix + '.test'

    # src, trg
    exts = ('.zh', '.en')
    batch_sizes = [8, 8, 8]

    # vocabulary
    src_min_freq = 10
    src_max_size = 50000
    trg_min_freq = 10
    trg_max_size = 50000

    common_size = 1000

    # model
    embedding_dim = 100
    hidden_model = 'GRU'
    hidden_dim = 100
    num_layers = 2
    dropout = 0.2

    use_cuda = torch.cuda.is_available()

    checkpoint_path = './mt-data/ted2013'

    valid_step = 1000

    pool_size = 200

config = Config()

trainer = Trainer.create(config)

if args.checkpoint is not None:
    trainer.load_checkpoint(args.checkpoint)

trainer.train(config.use_cuda)




