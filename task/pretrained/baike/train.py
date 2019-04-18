#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import argparse
import collections
import json
import os
import random
import time
from typing import Dict, Tuple
from collections import Counter

from tabulate import tabulate

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torchtext.vocab import Vocab
from tqdm import tqdm

from task.pretrained.transformer.attention import TransformerLayer
from .base import INIT_TOKEN, EOS_TOKEN, MASK_TOKEN, PAD_TOKEN, UNK_TOKEN
from .embedding import WindowEmbedding
from .classifier import ContextClassifier, LMClassifier, PhraseClassifier, PPMI
from .data import Field, LabelField, PhraseField, lazy_iter
from .encoder import StackRNN, ElmoEncoder
from .model import Model

from .transformer import Embeddings, Transformer


class Trainer:
    def __init__(self, config, model: Model,
                 dataset_it, text_voc: Vocab, label_vocabs: Dict[str, Vocab],
                 valid_step, checkpoint_dir):
        self.config = config
        self.model = model

        self.optimizer = optim.Adam(
            [
                # shared
                {'params': list(self.model.embedding.parameters()) + list(self.model.encoder.parameters()) + list(self.model.lm_classifier.parameters())},
                # phrase
                {'params': self.model.phrase_classifier.parameters()},
            ] + [{'params': task.parameters()} for task in self.model.label_classifiers],
            lr=1e-3,
            weight_decay=1e-6
        )

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
        # if optimizer:
        #     states['optimizer'] = self.optimizer.state_dict()
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

    def acc_train_one(self, pool: list):

        self.model.train()
        self.model.zero_grad()
        losses = Counter()
        for batch in pool:
            _losses, _batch_size = self.model(batch)
            losses.update(_losses)
            loss = sum(l for l in _losses.values()) / len(pool)
            if loss.requires_grad:
                loss.backward()

        # Step 3. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

        self.optimizer.step()

        self.model.zero_grad()
        return losses

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

                    del valid_batch

            return {n: l/count for n, l in losses.items()}

    def metrics(self, valid_it):
        self.model.eval()
        acc, pre, recall = [collections.defaultdict(float) for _ in range(3)]
        counter = collections.defaultdict(float)

        with torch.no_grad():
            with tqdm(total=len(valid_it.dataset), desc='metrics') as valid_tqdm:
                for _, valid_batch in enumerate(valid_it):
                    label_results, lm_result, phrase_result = self.model.predict(valid_batch)
                    valid_tqdm.update(len(valid_batch))

                    for label_name, label_result in label_results.items():
                        for sub_name, sub_result in label_result.items():
                            name = '%s_%s' % (label_name, sub_name)
                            for sen_res in sub_result:
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

                    if random.random() < 0.005:
                        self.pretty_print(valid_batch, label_results, lm_result, phrase_result, self.model.find_phrase(valid_batch))

                    del valid_batch

            scores = {n: {'acc': acc[n]/(c+1e-5), 'pre': pre[n]/(c+1e-5), 'recall': recall[n]/(c+1e-5)} for n, c in counter.items()}
            print(scores)
            return scores

    def pretty_print(self, batch, label_results, lm_result, phrase_result, find_phrases):
        text, lens = batch.text
        for bid in range(lens.size(0)):
            text_str = [self.text_voc.itos[w] for w in text[:lens[bid], bid]]
            print()
            print(text_str)
            predict_str = [INIT_TOKEN] + [self.text_voc.itos[w] for w in lm_result[:lens[bid]-2, bid]] + [EOS_TOKEN]
            print(predict_str)
            for label_name, label_result in label_results.items():
                for sub_name, sub_result in label_result.items():
                    name = '%s_%s' % (label_name, sub_name)
                    for label, pred in sub_result[bid]:
                        gold = set(self.label_vocabs[label_name].itos[i] for i in label.tags.tolist())
                        pred = set(self.label_vocabs[label_name].itos[i] for i in pred)
                        print('(%d,%d,%s): (%s, %s, %s)' % (
                            label.begin, label.end,
                            ''.join(text_str[label.begin:label.end]), name, gold, pred))

            for begin, end, gold, pred, weight in phrase_result[bid]:
                print('(%d,%d,%s): (%.5f, %.5f, %.5f)' % (begin, end, ''.join(text_str[begin:end]), gold, pred, weight))

            for begin, end, prob in find_phrases[bid]:
                print('(%d,%d,%s): %.5f' % (begin, end, ''.join(text_str[begin:end]), prob))

    def pool_dataset(self, dataset_it, pool_size=5):

        with tqdm(total=len(dataset_it.dataset), desc='train') as train_tqdm:
            pool = []
            for batch in dataset_it:
                train_tqdm.update(len(batch))
                if len(pool) >= pool_size:
                    yield pool
                    pool = []
                pool.append(batch)

            if len(pool) > 0:
                yield pool

    def train(self):
        total_batch, start = 1e-10, time.time()
        label_losses = Counter()
        num_iterations = 0
        for train_it, valid_it in tqdm(self.dataset_it, desc='dataset'):
            for pool in self.pool_dataset(train_it):
                num_iterations += 1
                losses = self.acc_train_one(pool)

                label_losses.update(losses)

                total_batch += len(pool)

                if num_iterations % self.valid_step == 0:

                    valid_losses = self.valid(valid_it)
                    total_valid_loss = sum(l for n, l in valid_losses.items()) / len(valid_losses)

                    self.checkpoint(num_iterations, total_valid_loss)

                    label_losses = {label: (loss/total_batch) for label, loss in label_losses.items()}

                    self.summary_writer.add_scalars(
                        'loss', {'train_mean_loss': sum(l for _, l in label_losses.items())/len(label_losses)},
                        num_iterations)
                    self.summary_writer.add_scalars(
                        'loss', {('train_%s' % n) : l for n, l in label_losses.items()},
                        num_iterations)

                    self.summary_writer.add_scalars(
                        'loss', {'valid_mean_loss': total_valid_loss},
                        num_iterations)
                    self.summary_writer.add_scalars(
                        'loss', {('valid_%s' % n) : l for n, l in valid_losses.items()},
                        num_iterations)

                    total_batch, start = 1e-10, time.time()
                    label_losses = Counter()

                if num_iterations % (self.valid_step * 1) == 0:
                    for tag, mat, metadata in self.model.named_embeddings():
                        '''
                        if len(metadata) > self.config.projector_max_size:
                            half_size = self.config.projector_max_size // 2
                            mat = torch.cat([mat[:half_size], mat[-half_size:]], dim=0)
                            metadata = metadata[:half_size] + metadata[-half_size:]
                        '''
                        mat = mat[:self.config.projector_max_size]
                        metadata = metadata[:self.config.projector_max_size]
                        metadata = ['<SPACE>' if len(tok.strip()) == 0 else tok for tok in metadata]
                        self.summary_writer.add_embedding(mat, metadata=metadata, tag=tag)

                    for n, scores in self.metrics(valid_it).items():
                        self.summary_writer.add_scalars(
                            'eval', {('%s_%s' % (n, sn)) : s for sn, s in scores.items()},
                            num_iterations)

            # reset optimizer
            # if self.train_it.iterations > self.config.warmup_step:
            #    self.optimizer = optim.Adam(self.model.parameters(), 1e-3, weight_decay=1e-3)

    def checkpoint(self, num_iterations, valid_loss):
        torch.save(self.state_dict(),
                   '%s/pretrained-last' % self.checkpoint_dir)

        if num_iterations % 1e5 == 0:
            torch.save(self.state_dict(),
                       '%s/pretrained-%d' % (self.checkpoint_dir, num_iterations))

        if len(self.checkpoint_files) == 0 or self.checkpoint_files[-1][0] > valid_loss:
            checkpoint_file = '%s/pretrained-%.4f' % (self.checkpoint_dir, valid_loss)
            torch.save(self.state_dict(), checkpoint_file)
            self.checkpoint_files.append((valid_loss, checkpoint_file))

        while len(self.checkpoint_files) > 5:
            loss, file = self.checkpoint_files.popleft()
            try:
                os.remove(file)
            except:
                pass

    @staticmethod
    def load_voc(config):
        TEXT = Field(mask_token=MASK_TOKEN, include_lengths=True,
                     init_token=INIT_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, unk_token=UNK_TOKEN)
        KEY_LABEL = LabelField('keys')
        ATTR_LABEL = LabelField('attrs')
        SUB_LABEL = LabelField('subtitles')
        PHRASE_LABEL = PhraseField('phrase')
        # ENTITY_LABEL = LabelField('entity')
        TEXT.build_vocab(os.path.join(config.root, 'text.voc.gz'),
                         max_size=config.voc_max_size,
                         min_freq=config.voc_min_freq)
        KEY_LABEL.build_vocab(os.path.join(config.root, 'key.voc.gz'),
                              max_size=config.key_max_size,
                              min_freq=config.key_min_freq)
        ATTR_LABEL.build_vocab(os.path.join(config.root, 'attr.voc.gz'),
                               max_size=config.attr_max_size,
                               min_freq=config.attr_min_freq)
        SUB_LABEL.build_vocab(os.path.join(config.root, 'subtitle.voc.gz'),
                              max_size=config.subtitle_max_size,
                              min_freq=config.subtitle_min_freq)
        # ENTITY_LABEL.build_vocab(os.path.join(config.root, 'entity.voc.gz'),
        #                         max_size=config.entity_max_size,
        #                         min_freq=config.entity_min_freq)

        print('text vocab size = %d' % len(TEXT.vocab))
        print('key vocab size = %d' % len(KEY_LABEL.vocab))
        print('attr vocab size = %d' % len(ATTR_LABEL.vocab))
        print('subtitle vocab size = %d' % len(SUB_LABEL.vocab))
        # print('entity vocab size = %d' % len(ENTITY_LABEL.vocab))

        return TEXT, [KEY_LABEL, ATTR_LABEL, SUB_LABEL], PHRASE_LABEL #, ENTITY_LABEL]

    @staticmethod
    def load_dataset(config, fields):
        # train_it, valid_it = BaikeDataset.iters(
        #    fields, path=config.root, train=config.train_file, device=config.device)
        # train_it.repeat = True
        def batch_size_fn(new, count, sofar):
            return sofar + (len(new.text) + 99)//100

        dataset_it = lazy_iter(fields,
                               path=config.root, data_prefix=config.train_prefix, valid_file=config.valid_file,
                               distant_dict=config.distant_dict,
                               batch_size=config.batch_size,
                               batch_size_fn=batch_size_fn,
                               device=config.device)

        return dataset_it

    @classmethod
    def load_model(cls, config, text_field, label_fields, phrase_field):

        embedding = nn.Embedding(len(text_field.vocab), config.embedding_dim,
                                    padding_idx=text_field.vocab.stoi[PAD_TOKEN])

        encoder = StackRNN(config.encoder_mode,
                           config.embedding_dim,
                           config.encoder_hidden_dim,
                           config.encoder_num_layers,
                           dropout=0.5,
                           bidirectional=True)

        lm_classifier = LMClassifier(len(text_field.vocab), config.embedding_dim, config.encoder_hidden_dim,
                                     shared_weight=embedding.weight,
                                     padding_idx=text_field.vocab.stoi[PAD_TOKEN])

        label_classifiers = nn.ModuleList([
            ContextClassifier(field.name, field.vocab, config.encoder_hidden_dim, config.label_dim) for field in label_fields])

        phrase_classifier = PPMI(phrase_field.name, config.encoder_hidden_dim)
        model = Model(text_field.vocab,
                      embedding,
                      encoder,
                      lm_classifier,
                      label_classifiers,
                      phrase_classifier)

        model.to(config.device)

        return model

    @classmethod
    def create(cls, config, checkpoint=None):
        TEXT, label_fields, PHRASE_FIELD = cls.load_voc(config)

        fields = [('text', TEXT), (tuple(field.name for field in (label_fields + [PHRASE_FIELD])), tuple(label_fields + [PHRASE_FIELD]))]

        # train_it, valid_it = BaikeDataset.iters(
        #    fields, path=config.root, train=config.train_file, device=config.device)
        # train_it.repeat = True

        dataset_it = cls.load_dataset(config, fields)

        model = cls.load_model(config, TEXT, label_fields, PHRASE_FIELD)

        trainer = cls(config, model, dataset_it,
            TEXT.vocab,
            dict((field.name, field.vocab) for field in label_fields),
            config.valid_step, config.checkpoint_dir)
        if checkpoint:
            trainer.load_checkpoint(checkpoint)
        return trainer


class Config:
    def __init__(self, output_dir=None):
        self.root = './baike/preprocess-char'
        self.train_prefix = 'sentence.url'
        self.valid_file = 'valid.gz'
        self.distant_dict = 'distant.dic'

        self.voc_max_size = 50000
        self.voc_min_freq = 50
        self.key_max_size = 150000
        self.key_min_freq = 50
        self.subtitle_max_size = 150000
        self.subtitle_min_freq = 50
        self.attr_max_size = 150000
        self.attr_min_freq = 50
        self.entity_max_size = 150000
        self.entity_min_freq = 50

        self.embedding_dim = 64
        self.encoder_mode = 'LSTM'  # ['RNN', 'LSTM', 'GRU']
        self.encoder_hidden_dim = 64
        self.encoder_num_layers = 2
        self.attention_num_heads = None

        self.label_dim = 64

        self.valid_step = 100

        self.batch_size = 16

        self.dir_prefix = output_dir if output_dir else 'elmo-focal'
        self.checkpoint_dir = os.path.join(self.root, self.dir_prefix, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # summary parameters
        self.summary_dir = os.path.join(self.root, self.dir_prefix, 'summary')
        os.makedirs(self.summary_dir, exist_ok=True)
        self.projector_max_size = 20000

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # save config
        with open(os.path.join(self.root, self.dir_prefix, 'config.txt'), 'wt') as file:
            file.write(json.dumps({name: param for name, param in self.__dict__.items() if name not in {'device'}},
                                  ensure_ascii=False, indent=2))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Preprocess baike corpus and save vocabulary')
    argparser.add_argument('--checkpoint', type=str, help='checkpoint path')
    argparser.add_argument('--output', type=str, default=None, help='output dir')

    args = argparser.parse_args()

    trainer = Trainer.create(Config(args.output), args.checkpoint)

    trainer.train()
