#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import time
import random
import os
import pickle
import collections
from typing import Tuple
import torch
from torch import optim
from torchtext import data
from torchtext.data.iterator import BucketIterator
from task.pretrained.transformer import Tagger
from tqdm import tqdm
from task.pretrained.transformer.dataset import TaggerDataset
from task.pretrained.transformer.field import PartialField
from .base import BOS, EOS, PAD

COARSE_STAGE = 'coarse'
FINE_STAGE = 'fine'


class Trainer:
    def __init__(self, model: Tagger, device: torch.device,
                 stage: str,
                 partial_train_it, partial_valid_it, partial_test_it,
                 valid_step, checkpoint_dir):

        self.model = model
        self.device = device
        self.stage = stage
        self.optimizer = optim.Adam(self.model.coarse_params() if stage == COARSE_STAGE else self.model.fine_params(),
                                    lr=1e-3)

        self.train_it, self.valid_it, self.test_it = \
            partial_train_it, partial_valid_it, partial_test_it

        self.valid_step = valid_step
        self.checkpoint_dir = checkpoint_dir

    def state_dict(self, train=True, optimizer=True):

        states = collections.OrderedDict()
        states['stage'] = self.stage
        states['model'] = self.model.state_dict()
        if optimizer:
            states['optimizer'] = self.optimizer.state_dict()
        if train:
            states['train_it'] = self.train_it.state_dict()
        return states

    def load_state_dict(self, states, strict):

        self.model.load_state_dict(states['model'], strict=strict)
        if states['stage'] == self.stage:
            if 'optimizer' in states:
                self.optimizer.load_state_dict(states['optimizer'])

            if 'train_it' in states:
                self.train_it.load_state_dict(states['train_it'])

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
        torch.nn.utils.clip_grad_norm_(self.model.coarse_params() if self.stage == COARSE_STAGE else self.model.fine_params(), 5)
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
        total_loss, total_sen, start = 0, 1e-10, time.time()
        checkpoint_losses = collections.deque()

        for step, batch in tqdm(enumerate(self.train_it, start=1), total=len(self.train_it)):

            loss, num_sen = self.train_one(batch, scale=1.0)

            total_loss += loss
            total_sen += num_sen

            if step % self.valid_step == 0:
                train_speed = total_sen / (time.time() - start)

                inference_start = time.time()
                valid_loss = self.valid(self.valid_it)

                self.checkpoint(checkpoint_losses, valid_loss)

                print("%s: train loss=%.6f\t\tvalid loss=%.6f" % (self.stage, total_loss/total_sen, valid_loss))
                print("speed:   train %.2f sentence/s  valid %.2f sentence/s\n\n" %
                      (train_speed, len(self.valid_it) / (time.time() - inference_start)))

                total_loss, total_sen, start = 0, 1e-10, time.time()

            if self.train_it.iterations % (self.valid_step * 5) == 0:
                with torch.no_grad():
                    # pass
                    print(self.model.evaluation(self.test_it))

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
    def create(cls, config, stage):
        text_field = data.Field(include_lengths=True, init_token=BOS, eos_token=EOS, pad_token=PAD)
        tag_field = PartialField(init_token=BOS, eos_token=EOS, pad_token=PAD)
        print('loading dataset.')
        if stage == COARSE_STAGE:
            partial_train, partial_valid, partial_test = TaggerDataset.splits(
                path=config.partial_prefix, train=config.train, validation=config.valid, test=config.test,
                fields = [('text', text_field), ('tags', tag_field)],
            )
        full_train, full_valid, full_test = TaggerDataset.splits(
            path=config.full_prefix,
            train=config.train,
            validation=config.valid,
            test=config.test,
            fields=[('text', text_field), ('tags', tag_field)],)

        print('build vocab.')
        if stage == FINE_STAGE:
            with open(config.cached_dataset_prefix + '.text.vocab', 'rb') as f:
                text_field.vocab = pickle.load(f, encoding='utf-8')
            with open(config.cached_dataset_prefix + '.tag.vocab', 'rb') as f:
                tag_field.vocab = pickle.load(f)
        else:
            text_field.build_vocab(partial_train, full_train, min_freq=config.text_min_freq)
            tag_field.build_vocab(partial_train, full_train, min_freq=config.tag_min_freq)

            with open(config.cached_dataset_prefix + '.text.vocab', 'wb') as f:
                pickle.dump(text_field.vocab, f)
            with open(config.cached_dataset_prefix + '.tag.vocab', 'wb') as f:
                pickle.dump(tag_field.vocab, f)

        device = torch.device("cuda" if config.use_cuda else "cpu")

        if stage == COARSE_STAGE:
            train_it, valid_it, test_it = \
                BucketIterator.splits([partial_train, partial_valid, partial_test],
                                      batch_sizes=config.batch_sizes,
                                      sort_within_batch=True,
                                      device=device)
        elif stage == FINE_STAGE:
            train_it, valid_it, test_it = \
                BucketIterator.splits([full_train, full_valid, full_test],
                                      batch_sizes=config.batch_sizes,
                                      sort_within_batch=True,
                                      device=device)

        train_it.repeat = True

        model = Tagger.create(
                       text_field.vocab, tag_field.vocab,
                       config.embedding_dim, config.encoder_dim, config.encoder_depth, config.attention_num_head)
        model = model.to(device)
        return Trainer(model, device, stage,
                       train_it, valid_it, test_it,
                       config.valid_step, config.checkpoint_path)


    @classmethod
    def createLattice(cls, config, stage):
        text_field = data.Field(include_lengths=True, init_token=BOS, eos_token=EOS, pad_token=PAD)
        tag_field = PartialField(init_token=BOS, eos_token=EOS, pad_token=PAD)
        print('loading dataset.')

        from .lattice import LatticeField
        lattice = LatticeField(config.pretrained_embed)
        if stage == COARSE_STAGE:
            partial_train, partial_valid, partial_test = TaggerDataset.splits(
                path=config.partial_prefix,
                train=config.train,
                validation=config.valid,
                test=config.test,
                fields=[(('text', 'lattices'), (text_field, lattice)), ('tags', tag_field)],)
        full_train, full_valid, full_test = TaggerDataset.splits(
            path=config.full_prefix,
            train=config.train,
            validation=config.valid,
            test=config.test,
            fields=[(('text', 'lattices'), (text_field, lattice)), ('tags', tag_field)],)

        print('build vocab.')
        if stage == FINE_STAGE:
            with open(config.cached_dataset_prefix + '.text.vocab', 'rb') as f:
                text_field.vocab = pickle.load(f, encoding='utf-8')
            with open(config.cached_dataset_prefix + '.tag.vocab', 'rb') as f:
                tag_field.vocab = pickle.load(f)
        else:
            text_field.build_vocab(partial_train, full_train, min_freq=config.text_min_freq)
            tag_field.build_vocab(partial_train, full_train, min_freq=config.tag_min_freq)

            print(tag_field.vocab.itos)
            with open(config.cached_dataset_prefix + '.text.vocab', 'wb') as f:
                pickle.dump(text_field.vocab, f)
            with open(config.cached_dataset_prefix + '.tag.vocab', 'wb') as f:
                pickle.dump(tag_field.vocab, f)

        device = torch.device("cuda" if config.use_cuda else "cpu")

        if stage == COARSE_STAGE:
            train_it, valid_it, test_it = \
                BucketIterator.splits([partial_train, partial_valid, partial_test],
                                      batch_sizes=config.batch_sizes,
                                      sort_within_batch=True,
                                      device=device)
        elif stage == FINE_STAGE:
            train_it, valid_it, test_it = \
                BucketIterator.splits([full_train, full_valid, full_test],
                                      batch_sizes=config.batch_sizes,
                                      sort_within_batch=True,
                                      device=device)

        train_it.repeat = True

        # model = Tagger.createLattice(
        #               text_field.vocab, tag_field.vocab,
        #               config.embedding_dim, config.encoder_dim, config.encoder_depth, config.attention_num_head,
        #               lattice.word_emb, lattice.max_sub_len)
        # model = Tagger.createLattice(
        model = Tagger.createLSTM(
            text_field.vocab, tag_field.vocab,
            config.embedding_dim, config.encoder_dim, config.encoder_depth, config.attention_num_head,
            lattice.word_emb, lattice.max_sub_len
        )
        model = model.to(device)
        return Trainer(model, device, stage,
                       train_it, valid_it, test_it,
                       config.valid_step, config.checkpoint_path)

class Config:
    partial_prefix = './pos/data'
    full_prefix = './pos/data'

    train = 'std.train'
    valid = 'std.valid'
    test = 'std.gold'

    pretrained_embed = './wordvec/Tencent_AILab_ChineseEmbedding.small.txt.gz'

    batch_sizes = [16, 32, 32]

    # vocabulary
    text_min_freq = 10
    # text_min_size = 50000

    tag_min_freq = 1
    # tag_min_size = 50000

    common_size = 1000

    # model
    vocab_size = 100
    embedding_dim = 512
    encoder_dim = 512
    attention_num_head = None
    encoder_depth = 2

    use_cuda = torch.cuda.is_available()

    cached_dataset_prefix = './pos/dataset'
    checkpoint_path = './pos/model/mode_{}_emb_{}_hidden_{}_layer_{}_head_{}'.format(
        'transformer', embedding_dim, encoder_dim, encoder_depth, attention_num_head)

    valid_step = 200


import argparse

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    arg_parser.add_argument('--stage', type=str, choices=[COARSE_STAGE, FINE_STAGE], default=COARSE_STAGE, help='coarse or fine')

    args = arg_parser.parse_args()

    config = Config()

    trainer = Trainer.createLattice(config, args.stage)

    if args.checkpoint is not None:
        trainer.load_checkpoint(args.checkpoint)

    if args.stage in ['coarse', 'fine']:
        trainer.train()