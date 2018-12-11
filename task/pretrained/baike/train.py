#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import collections
import os
import json
import time
import random
import argparse
from typing import Dict, Tuple

import torch
from torch import nn, optim
from tqdm import tqdm
from torchtext.vocab import Vocab
from torchtext import data
from torchtext.data.iterator import BucketIterator

from .base import INIT_TOKEN, EOS_TOKEN, MASK_TOKEN, PAD_TOKEN
from .model import Model
from .data import BaikeDataset, Field, LabelField, lazy_iter
from .encoder import LSTMEncoder, StackLSTM
from .classifier import LabelClassifier, PhraseClassifier

from task.pretrained.transformer.attention import MultiHeadedAttention
from ..transformer.transformer import Tagger, MaskedCRF, LinearCRF
from ..transformer.field import PartialField
from ..transformer import base
from ..transformer.dataset import TaggerDataset

from tensorboardX import SummaryWriter


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

                    if num_iterations % (self.valid_step * 5) == 0:
                        for tag, mat, metadata in self.model.named_embeddings():
                            if len(metadata) > self.config.projector_max_size:
                                half_size = self.config.projector_max_size // 2
                                mat = torch.cat([mat[:half_size], mat[-half_size:]], dim=0)
                                metadata = metadata[:half_size] + metadata[-half_size:]
                            self.summary_writer.add_embedding(mat, metadata=metadata, tag=tag)

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
            try:
                os.remove(file)
            except:
                pass

    @staticmethod
    def load_voc(config):
        TEXT = Field(mask_token=MASK_TOKEN, include_lengths=True, init_token=INIT_TOKEN, eos_token=EOS_TOKEN)
        KEY_LABEL = LabelField('keys')
        ATTR_LABEL = LabelField('attrs')
        SUB_LABEL = LabelField('subtitles')
        TEXT.build_vocab(os.path.join(config.root, 'text.voc.gz'), max_size=50000, min_freq=100)
        KEY_LABEL.build_vocab(os.path.join(config.root, 'key.voc.gz'), max_size=150000, min_freq=100)
        ATTR_LABEL.build_vocab(os.path.join(config.root, 'attr.voc.gz'), max_size=150000, min_freq=100)
        SUB_LABEL.build_vocab(os.path.join(config.root, 'subtitle.voc.gz'), max_size=150000, min_freq=100)

        print('text vocab size = %d' % len(TEXT.vocab))
        print('key vocab size = %d' % len(KEY_LABEL.vocab))
        print('attr vocab size = %d' % len(ATTR_LABEL.vocab))
        print('subtitle vocab size = %d' % len(SUB_LABEL.vocab))

        return TEXT, KEY_LABEL, ATTR_LABEL, SUB_LABEL

    @staticmethod
    def load_dataset(config, fields):
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

    @staticmethod
    def load_model(config, text_field, key_field, attr_field, sub_field):

        embedding = nn.Embedding(len(text_field.vocab), config.embedding_size)

        encoder = StackLSTM(config.embedding_size,
                            config.encoder_size, config.encoder_num_layers,
                            residual=False, dropout=0.2)

        attention = None if config.attention_num_heads is None \
            else MultiHeadedAttention(config.attention_num_heads, config.encoder_size)

        label_classifiers = nn.ModuleList([
            LabelClassifier(name, config.classifier_loss, field.vocab, config.encoder_size, config.label_size)
            for name, field in [('keys', key_field), ('attrs', attr_field), ('subtitles', sub_field)]])
        phrase_classifier = PhraseClassifier(config.encoder_size)
        model = Model(text_field.vocab, embedding,
                      encoder, attention, label_classifiers, phrase_classifier, config.phrase_mask_prob)

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


class FineTrainer:
    def __init__(self, config, model: Tagger,
                 partial_train_it, partial_valid_it, partial_test_it,
                 valid_step, checkpoint_dir):
        self.config = config
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        # self.shared_optimizer = optim.Adam(list(self.model.embedding.parameters()) + list(self.model.encoder.parameters()), lr=1e-5)

        self.train_it, self.valid_it, self.test_it = \
            partial_train_it, partial_valid_it, partial_test_it

        self.valid_step = valid_step
        self.checkpoint_dir = checkpoint_dir

        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir)

    def state_dict(self, train=True, optimizer=True):

        states = collections.OrderedDict()
        states['model'] = self.model.state_dict()
        if optimizer:
            states['optimizer'] = self.optimizer.state_dict()
        if train:
            states['train_it'] = self.train_it.state_dict()
        return states

    def load_state_dict(self, states, strict):

        self.model.load_state_dict(states['model'], strict=strict)
        if 'optimizer' in states:
            self.optimizer.load_state_dict(states['optimizer'])

        if 'train_it' in states:
            self.train_it.load_state_dict(states['train_it'])

    def load_checkpoint(self, path, strict=True):
        states = torch.load(path)
        self.load_state_dict(states, strict=strict)

    def _model_training(self, all=False):
        if all is False:
            self.model.embedding.eval()
            self.model.encoder.eval()
        self.model.crf.train()

    def train_one(self, batch, scale) -> Tuple[float, int]:
        self.model.train()
        self.model.zero_grad()

        loss, num_sen = self.model.criterion(batch)
        rloss = loss.item()
        (loss * scale).div(num_sen).backward()

        # Step 3. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()
        # self.shared_optimizer.step()
        return rloss, num_sen

    def valid(self, valid_it) -> float:
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            num_samples = 0

            sample = False
            for _, valid_batch in tqdm(enumerate(valid_it), desc='valid', total=len(self.valid_it)):
                loss, num_token = self.model.criterion(valid_batch)

                total_loss += loss.item()
                num_samples += num_token

                if not sample and random.random() < 0.02:
                    self.model.sample(valid_batch)

            return total_loss / num_samples, num_samples

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
                valid_loss, valid_num_samples = self.valid(self.valid_it)

                self.checkpoint(checkpoint_losses, valid_loss)

                print("train loss=%.6f\t\tvalid loss=%.6f" % (total_loss/total_sen, valid_loss))
                print("speed:   train %.2f sentence/s  valid %.2f sentence/s\n\n" %
                      (train_speed, valid_num_samples / (time.time() - inference_start)))

                self.summary_writer.add_scalars('loss', {'train': total_loss/total_sen, 'valid': valid_loss}, step)

                total_loss, total_sen, start = 0, 1e-10, time.time()

            if self.train_it.iterations % (self.valid_step) == 0:
                with torch.no_grad():
                    # pass
                    eval_res = self.model.evaluation(self.test_it)
                    print(eval_res)
                    self.summary_writer.add_scalars('eval', eval_res, step)

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
    def create(cls, coarse_config, checkpoint, fine_config):
        if checkpoint is not None:
            TEXT, KEY_LABEL, ATTR_LABEL, SUB_LABEL = Trainer.load_voc(coarse_config)
            model = Trainer.load_model(coarse_config, TEXT, KEY_LABEL, ATTR_LABEL, SUB_LABEL)
            states = torch.load(checkpoint)
            model.load_state_dict(states['model'])
            embedding, encoder = model.embedding, model.encoder
        else:
            TEXT = data.Field(include_lengths=True, init_token=INIT_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN)
            embedding, encoder = None, None

        tag_field = PartialField(init_token=INIT_TOKEN, eos_token=EOS_TOKEN)
        train, valid, test = TaggerDataset.splits(
            path=fine_config.full_prefix,
            train=fine_config.train,
            validation=fine_config.valid,
            test=fine_config.test,
            fields=[('text', TEXT), ('tags', tag_field)])

        if checkpoint is None:
            TEXT.build_vocab(train, min_freq=fine_config.text_min_freq)
        tag_field.build_vocab(train, min_freq=fine_config.tag_min_freq)

        train_it, valid_it, test_it = \
            BucketIterator.splits([train, valid, test],
                                  batch_sizes=fine_config.batch_sizes,
                                  shuffle=True,
                                  sort_within_batch=True,
                                  device=fine_config.device)

        train_it.repeat = True

        if checkpoint is None:
            embedding = nn.Embedding(len(TEXT.vocab), coarse_config.embedding_size)
            encoder = StackLSTM(coarse_config.embedding_size, coarse_config.encoder_size,
                            coarse_config.encoder_num_layers, residual=False, dropout=0.2)
        crf = MaskedCRF(coarse_config.encoder_size,
                        len(tag_field.vocab),
                        tag_field.vocab.transition_constraints,
                        fine_config.attention_num_heads)
        fine_model = Tagger(TEXT.vocab, tag_field.vocab, embedding, encoder, crf)

        fine_model.to(fine_config.device)

        return cls(fine_config,
                   fine_model,
                   train_it, valid_it, test_it,
                   fine_config.valid_step, fine_config.checkpoint_path)


class CoarseConfig:
    def __init__(self):
        self.root = './baike/preprocess4'
        self.train_file = 'sentence.url'

        self.embedding_size = 256
        self.encoder_size = 512
        self.encoder_num_layers = 2
        self.attention_num_heads = 8

        self.label_size = 256

        self.phrase_mask_prob = 0.2

        self.valid_step = 1000
        self.warmup_step = 5000

        self.batch_size = 16

        self.dir_prefix = 'subsample-sum-phrase'
        self.checkpoint_dir = os.path.join(self.root, self.dir_prefix, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # summary parameters
        self.summary_dir = os.path.join(self.root, self.dir_prefix, 'summary')
        os.makedirs(self.summary_dir, exist_ok=True)
        self.projector_max_size = 20000

        self.classifier_loss = 'softmax' # ['softmax', 'negativesample', 'adaptivesoftmax]

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # save config
        with open(os.path.join(self.root, self.dir_prefix, 'config.txt'), 'wt') as file:
            file.write(json.dumps({name: param for name, param in self.__dict__.items() if name not in {'device'}},
                                  ensure_ascii=False, indent=2))


class FineConfig:
    def __init__(self):
        self.partial_prefix = './pos/data'
        self.full_prefix = './pos/data'

        self.train = 'std.train'
        self.valid = 'std.valid'
        self.test = 'std.gold'

        self.batch_sizes = [16, 32, 32]

        # vocabulary
        self.text_min_freq = 10
        # text_min_size = 50000

        self.tag_min_freq = 10
        # tag_min_size = 50000

        self.common_size = 1000

        # model
        self.vocab_size = 100
        self.embedding_size = 512
        self.encoder_size = 512
        self.attention_num_heads = 8
        self.encoder_depth = 2

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.cached_dataset_prefix = './pos/dataset'
        self.checkpoint_path = './pos/model/mode_{}_emb_{}_hidden_{}_layer_{}_head_{}'.format(
            'lstm', self.embedding_size, self.encoder_size, self.encoder_depth, self.attention_num_heads)

        # summary parameters
        self.summary_dir = os.path.join('./pos', 'summary')
        os.makedirs(self.summary_dir, exist_ok=True)

        self.valid_step = 200


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Preprocess baike corpus and save vocabulary')
    argparser.add_argument('--checkpoint', type=str, help='checkpoint path')
    argparser.add_argument('--fine', type=bool, default=False, help='fine stage. default=False')

    args = argparser.parse_args()

    if args.fine is False:
        trainer = Trainer.create(CoarseConfig(), args.checkpoint)
    else:
        trainer = FineTrainer.create(CoarseConfig(), args.checkpoint, FineConfig())

    trainer.train()
