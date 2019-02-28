#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import argparse
import itertools
import math
import os
import re
import random
import time
from collections import defaultdict, deque
from typing import Dict

from tabulate import tabulate
from tensorboardX import SummaryWriter
from torch import optim
from torchtext import vocab
from torchtext.data.iterator import BucketIterator
from tqdm import tqdm

from .attention import MultiHeadedAttention
from .base import INIT_TOKEN, EOS_TOKEN, PAD_TOKEN, make_masks, bio_to_bmeso, listfile, strQ2B
from .crf import LinearCRF, MaskedCRF
from .encoder import ElmoEncoder, LSTM
from .tags import *
from .train import Trainer, Config
from ..transformer.field import PartialField
from ..transformer.vocab import TagVocab
from .embedding import CompressedEmbedding


class NamedEntityData(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path: str, fields: List, **kwargs):

        examples = []
        for chars, types in self.get_line(path):
            chars, types = bio_to_bmeso(chars, types)
            examples.append(data.Example.fromlist([chars, types], fields))
        super(NamedEntityData, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def get_line(path):
        with open(path) as file:
            chars, types = [], []
            for line in tqdm(file, desc='load data from %s ' % (path)):
                line = line.strip()
                if len(line) == 0:
                    yield chars, types
                    chars, types = [], []
                else:
                    char, type = line.rsplit(maxsplit=1)
                    chars.append(char)
                    types.append(type)

            if len(chars) > 0:
                yield chars, types


class POSData(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path: str, fields: List, **kwargs):
        examples = [data.Example.fromlist((chars, types), fields)
                    for chars, types in self.get_line(path)]
        super(POSData, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def get_line(path):
        with open(path) as file:
            for line in tqdm(file, desc='load data from %s ' % (path)):
                chars, types = [], []
                for tokens in line.split():
                    _word, _type = tokens.rsplit('#', maxsplit=1)
                    chars.extend(_word)
                    if len(_word) == 1:
                        types.extend(['S_%s' % _type])
                    elif len(_word) >= 2:
                        types.extend(['B_%s' % _type] + ['M_%s' % _type] * (len(_word) - 2) + ['E_%s' % _type])
                yield chars, types


class People2014(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, fields: List, **kwargs):
        raws = []
        char_counter = Counter()
        tag_counter = Counter()
        for line in tqdm(self.read_line(path), desc='load data from %s' % path):
            try:
                chars, tags = [], []
                for word, tag in self.fix_tokenize(line):
                    chars.extend(list(word))
                    tags.extend(to_bmes(len(word), tag))

                char_counter.update(chars)
                tag_counter.update(tags)
                raws.append((chars, tags))
            except Exception as e:
                print(e)
                print(line)
                print(self.fix_tokenize(line))

        print(tag_counter.most_common()[-100:])
        examples = [data.Example.fromlist((chars, types), fields)
                    for chars, types in raws]
        super(People2014, self).__init__(examples, fields, **kwargs)

    def fix_tokenize(self, raw_line):
        tokens = []
        line = re.sub(r'\[([^\]]+)\]/[a-z]+[0-9]?', r'\1', raw_line)
        for token in line.split():
            items = token.rsplit('/', maxsplit=1)
            if len(items) == 1:
                tokens.extend([(c, 'w') for c in items])
            else:
                word, tag = items
                for i in range(len(word)):
                    if word[i] in {'，', '：', '、', '。', '“', '”', '《', '》', '（', '）', '(', ')', '……', '‘', '’'}:
                        tokens.append((word[i], 'w'))
                    else:
                        if tag in ['ude1', 'ude2', 'ude3', 'w', 'wb'] and len(word[i:]) > 1:
                            print(raw_line)
                            print(line)
                            print(items)
                        tokens.append((word[i:], tag))
                        break
        return tokens

    def read_line(self, path: str):
        if os.path.isdir(path):
            for name in os.listdir(path):
                child = os.path.join(path, name)
                yield from self.read_line(child)
        elif path.endswith('.txt'):
            with open(path) as file:
                for line in file:
                    line = line.strip()
                    if len(line) > 0:
                        yield line
        else:
            print('%s is not loaded.' % path)

    @classmethod
    def splits(cls, path=None, root='.data', train=None, validation=None,
               test=None, **kwargs):

        train_data = None if train is None else cls(
            os.path.join(path, train), **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), **kwargs)

        if val_data is None and test_data is None:
            train_data, val_data, test_data = train_data.split(split_ratio=[0.85, 0.05, 0.1])

        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


class MSRA(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, fields: List, **kwargs):
        raws = []
        char_counter = Counter()
        tag_counter = Counter()
        for line in tqdm(self.read_line(path), desc='load data from %s' % path):
            try:
                chars, tags = [], []
                for word, tag in self.fix_tokenize(line):
                    chars.extend(list(word))
                    tags.extend(to_bmes(len(word), tag))

                char_counter.update(chars)
                tag_counter.update(tags)
                raws.append((chars, tags))
            except Exception as e:
                print(e)
                print(line)
                print(self.fix_tokenize(line))

        print(tag_counter.most_common()[-100:])
        examples = [data.Example.fromlist((chars, types), fields)
                    for chars, types in raws]
        super(MSRA, self).__init__(examples, fields, **kwargs)

    def fix_tokenize(self, raw_line):
        tokens = []
        line = re.sub(r'\[([^\]]+)\]/[a-z]+[0-9]?', r'\1', raw_line)
        for token in line.split():
            items = token.rsplit('/', maxsplit=1)
            if len(items) == 1:
                tokens.extend([(c, 'w') for c in items])
            else:
                word, tag = items
                for i in range(len(word)):
                    if word[i] in {'，', '：', '、', '。', '“', '”', '《', '》', '（', '）', '(', ')', '……', '‘', '’'}:
                        tokens.append((word[i], 'w'))
                    else:
                        if tag in ['ude1', 'ude2', 'ude3', 'w', 'wb'] and len(word[i:]) > 1:
                            print(raw_line)
                            print(line)
                            print(items)
                        tokens.append((word[i:], tag))
                        break
        return tokens

    def read_line(self, path: str):
        if os.path.isdir(path):
            for name in os.listdir(path):
                child = os.path.join(path, name)
                yield from self.read_line(child)
        elif path.endswith('.txt'):
            with open(path) as file:
                for line in file:
                    line = line.strip()
                    if len(line) > 0:
                        yield line
        else:
            print('%s is not loaded.' % path)

    @classmethod
    def splits(cls, path=None, root='.data', train=None, validation=None,
               test=None, **kwargs):

        train_data = None if train is None else cls(
            os.path.join(path, train), **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), **kwargs)

        if val_data is None and test_data is None:
            train_data, val_data, test_data = train_data.split(split_ratio=[0.85, 0.05, 0.1])

        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


class ChunkDataset(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, fields: List, **kwargs):
        examples = [data.Example.fromlist((chars, poses, types), fields)
                    for chars, poses, types in self.read_line(path)]
        super(ChunkDataset, self).__init__(examples, fields, **kwargs)

    def read_line(self, path: str):
        with open(path) as file:
            words, poses, tags = [], [], []
            for id, line in tqdm(enumerate(file)):
                line = strQ2B(line.strip())
                if len(line) == 0:
                    if len(words) > 0:
                        if set(tags) == {'O'}:
                            print(list(zip(words, tags)))
                        else:
                            try:
                                yield self.to_flat(words, poses, tags)
                            except Exception as e:
                                print(e)
                                print(list(zip(words, tags)))
                        words, poses, tags = [], [], []
                    continue

                tokens = line.split()
                words.append(tokens[5])
                poses.append(tokens[4].split('-')[0])
                tags.append(tokens[3] if tokens[4] != 'URL' else 'URL')

            if len(words) > 0:
                try:
                    yield self.to_flat(words, poses, tags)
                except Exception as e:
                    print(e)
                    print(zip(words, poses, tags))

    def isalnum(self, char):
        return 'a' < char < 'z' or 'A' < char < 'Z' or '0' < char < '9'

    def add_space(self, words, poses, tags):
        fixed_words, fixed_poses, fixed_tags = [], [], []
        for word, pos, tag in zip(words, poses, tags):
            if len(fixed_words) > 0 and self.isalnum(fixed_words[-1][-1]) and self.isalnum(word[0]):
                fixed_words.append(' ')
                fixed_poses.append('SPACE')
                if tag.startswith('I-'):
                    fixed_tags.append(tag)
                else:
                    fixed_tags.append('O')
            fixed_words.append(word)
            fixed_poses.append(pos)
            fixed_tags.append(tag)
        return fixed_words, fixed_poses, fixed_tags

    def to_flat(self, words, poses, tags):
        chunks, chunk_poses, chunk_types = [], [], []
        for word, pos, tag in zip(*self.add_space(words, poses, tags)):
            if tag.startswith('B-'):
                chunks.append([word])
                chunk_poses.append([pos])
                chunk_types.append(tag[2:])
            elif tag.startswith('I-'):
                chunks[-1].append(word)
                chunk_poses[-1].append(pos)
                assert chunk_types[-1] == tag[2:]
            else:
                chunks.append([word])
                chunk_poses.append([pos])
                chunk_types.append(tag)

        chars, char_poses, char_tags = [], [], []
        for chunk, poses, type in zip(chunks, chunk_poses, chunk_types):
            chunk_chars = []
            for word, pos in zip(chunk, poses):
                chunk_chars.extend(word)
                char_poses.extend(to_bmes(len(word), pos))

            chars.extend(chunk_chars)
            char_tags.extend(to_bmes(len(chunk_chars), type))

        return chars, char_poses, char_tags

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

def showAttention(input_sentence: List[str], output_words: List[str], attentions: torch.Tensor):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels(input_sentence, rotation=90)
    ax.set_yticklabels(output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


class Tagger(nn.Module):
    def __init__(self,
                 words: vocab.Vocab,
                 tags: Dict[str, TagVocab],
                 embedding: nn.Embedding,
                 encoder: nn.Module,
                 crfs: nn.ModuleDict):
        super(Tagger, self).__init__()
        self.words = words
        self.tags = tags

        self.embedding = embedding
        self.dropout = nn.Dropout(0.5)
        self.encoder = encoder
        self.crfs = crfs

    def _encode(self, batch: data.Batch):
        sens, lens = batch.text
        token_masks = make_masks(sens, lens)
        emb = self.embedding(sens)

        if isinstance(self.encoder, ElmoEncoder):
            forwards, backwards = self.encoder(emb, lens)
            hidden = torch.cat((forwards[-1], backwards[-1]), dim=-1)
        else:
            hidden, _ = self.encoder(emb)

        return hidden, token_masks

    def predict(self, batch: data.Batch):
        sens, lens = batch.text
        hiddens, token_masks = self._encode(batch)
        return {name: crf(hiddens, lens) for name, crf in self.crfs.items()}

    def predict_with_prob(self, batch: data.Batch) -> Dict[str, list]:
        sens, lens = batch.text
        hiddens, token_masks = self._encode(batch)
        return {name: crf.predict_with_prob(hiddens, lens) for name, crf in self.crfs.items()}

    def nbest(self, batch: data.Batch):
        sens, lens = batch.text
        hiddens, token_masks = self._encode(batch)
        return {name: crf.nbest(hiddens, lens, 5) for name, crf in self.crfs.items()}

    def criterion(self, batch: data.Batch) -> Dict[str, torch.Tensor]:
        sens, lens = batch.text
        hidden, token_masks = self._encode(batch)
        return {name: crf.neg_log_likelihood(hidden, lens, getattr(batch, name)[0]) for name, crf in self.crfs.items()}

    def print_transition(self):
        for name in self.tags.keys():
            print(tabulate([self.tags[name].itos] + self.crfs[name].transition.tolist(),
                           headers="firstrow", tablefmt='grid', floatfmt='3.3g'))

    def print(self, batch: data.Batch):
        text, text_len = batch.text
        for name in self.crfs.keys():
            gold_masks, gold_tags = batch[name]
            for i in range(len(text_len)):
                length = text_len[i]
                print(name + ': ' + ' '.join([self.words.itos[w] + '#' + t
                                for w, t in zip(text[0:length, i].data.tolist(), gold_tags[i])]))

    def sample_nbest(self, batch: data.Batch):
        self.eval()
        text, text_len = batch.text
        gold_masks, gold_tags = batch.tags
        results = self.nbest(batch)

        for name, nbests in results.items():
            for i in range(len(nbests)):
                length = text_len[i]
                sen = [self.words.itos[w] for w in text[0:length, i].data.tolist()]

                def tostr(words: List[str], tags: List[str]):
                    for word, tag in zip(words, tags):
                        if tag == 'E_O' or tag == 'S_O':
                            yield word + ' '
                        elif tag == 'B_O':
                            yield word + ' '
                        elif tag.startswith('B_'):
                            yield '[[%s' % (word)
                        elif tag.startswith('E_'):
                            yield '%s||%s]] ' % (word, tag[2:])
                        elif tag.startswith('S_'):
                            yield '[[%s||%s]] ' % (word, tag[2:])
                        elif tag.startswith('M_'):
                            yield word
                        else:
                            yield word + ' '

                gold_tag = ''.join(tostr(sen, gold_tags[i]))
                # gold_tag = ''.join([tostr(w, id) for w, id in zip(sen, gold_tags[0:length, i].data)])

                # if pred_tag != gold_tag:
                # print('\ngold: %s\npred: %s\nscore: %f' % (gold_tag, pred_tag, score))
                print('\ngold: %s' % (gold_tag))
                for pred_tag, score in nbests[i]:
                    pred_tag = ''.join(tostr(sen, [self.tags[name].itos[tag_id] for tag_id in pred_tag]))
                    print('pred: %s\nscore: %f' % (pred_tag, score))

    def sample(self, batch: data.Batch):
        self.eval()
        text, text_len = batch.text
        pred_results = self.predict_with_prob(batch)

        for name, pred_tags in pred_results.items():
            gold_masks, gold_tags = getattr(batch, name)

            for i in range(len(pred_tags)):
                pred_tag, score = pred_tags[i]
                length = text_len[i]
                sen = [self.words.itos[w] for w in text[0:length, i].data.tolist()]

                def tostr(words: List[str], tags: List[str]):
                    for word, tag in zip(words, tags):
                        if tag == 'E_O' or tag == 'S_O':
                            yield word + ' '
                        elif tag == 'B_O':
                            yield word
                        elif tag.startswith('B_'):
                            yield '[[%s' % (word)
                        elif tag.startswith('E_'):
                            yield '%s||%s]] ' % (word, tag[2:])
                        elif tag.startswith('S_'):
                            yield '[[%s||%s]] ' % (word, tag[2:])
                        elif tag.startswith('M_'):
                            yield word
                        else:
                            yield word + ' '

                pred_tag = [self.tags[name].itos[tag_id] for tag_id in pred_tag]
                gold_tag = gold_tags[i][:text_len[i]]
                if pred_tag != gold_tag:
                    pred_tag = ''.join(tostr(sen, pred_tag))
                    gold_tag = ''.join(tostr(sen, gold_tag))
                    print('\n%s\ngold: %s\npred: %s\nscore: %f' % (name, gold_tag, pred_tag, score))

    def evaluation_one(self,
                       preds: List[str],
                       golds: List[str]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        assert len(preds) == len(golds)
        correct_counts = defaultdict(float)
        gold_counts = defaultdict(float)
        pred_counts = defaultdict(float)

        begin = 0
        while begin < len(golds):
            end = begin
            while end < len(golds):
                if golds[end][0:2] in ['S_', 'E_'] or golds[end] in ['O', '*']:
                    end += 1
                    break
                else:
                    end += 1
            if not golds[begin].endswith('*') and not golds[begin].endswith('_O'):
                tag_type = golds[begin][2:]
                gold_counts[tag_type] += 1
                if preds[begin:end] == golds[begin:end]:
                    correct_counts[tag_type] += 1

            begin = end

        for t in preds:
            if t[0:2] in ['B_', 'S_'] and not t.endswith('_O'):
                pred_counts[t[2:]] += 1
            # elif t in ['O', '*']:
            #    pred_counts[t] += 1

        return correct_counts, gold_counts, pred_counts

    def evaluation(self, data_it) -> Dict[str, Dict[str, float]]:
        self.eval()
        counts = {name: {'correct': Counter(), 'gold': Counter(), 'pred': Counter()}
                                                    for name in self.crfs.keys()}
        for batch in tqdm(data_it, desc='eval', total=len(data_it)):
            _, text_len = batch.text
            results = self.predict(batch)

            for task_name, preds in results.items():
                masks, golds = getattr(batch, task_name)

                # print(gold_tags)
                for i in range(len(text_len)):
                    pred, score = preds[i]
                    _correct_counts, _gold_counts, _pred_counts = self.evaluation_one(
                        [self.tags[task_name].itos[p] for p in pred[1:-1]], golds[i][1:text_len[i] - 1])

                    counts[task_name]['correct'].update(_correct_counts)
                    counts[task_name]['gold'].update(_gold_counts)
                    counts[task_name]['pred'].update(_pred_counts)

        tasks = {}
        for task_name, detail in counts.items():
            correct_counts, gold_counts, pred_counts = detail['correct'], detail['gold'], detail['pred']
            total_correct = sum(correct_counts.values())
            total_gold = sum(gold_counts.values())
            total_pred = sum(pred_counts.values())

            results = {'total_f1': total_correct * 2 / (total_gold + total_pred + 1e-5),
                       'total_prec': total_correct / (total_gold + 1e-5),
                       'total_recall': total_correct / (total_pred + 1e-5)}
            for name in detail['gold'].keys():
                results['%s_f1' % name] = correct_counts[name] * 2 / (pred_counts[name] + gold_counts[name] + 1e-5)
                # results['%s_prec' % name] = correct_counts[name] / (pred_counts[name] + 1e-5)
                # results['%s_recall' % name] = correct_counts[name] / (gold_counts[name] + 1e-5)

            tasks[task_name] = results
        return tasks


class FineTrainer:
    def __init__(self,
                 config,
                 model: Tagger,
                 train_it, valid_it, test_it,
                 valid_step, checkpoint_dir):
        self.config = config
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=5e-5)

        self.train_it, self.valid_it, self.test_it = \
            train_it, valid_it, test_it

        self.valid_step = valid_step
        self.checkpoint_dir = checkpoint_dir

        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir)

    def state_dict(self, train=True, optimizer=True):

        states = OrderedDict()
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

    def train_one(self, batch) -> Dict[str, float]:
        self.model.train()
        self.model.zero_grad()

        losses = self.model.criterion(batch)

        (sum(losses.values()) / len(batch)).backward()

        # Step 3. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()

        return {name: loss.item() for name, loss in losses.items()}

    def valid(self, valid_it) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            total_losses = Counter()
            num_samples = 0

            sample = False
            for _, valid_batch in tqdm(enumerate(valid_it), desc='valid', total=len(self.valid_it)):
                losses = self.model.criterion(valid_batch)

                total_losses.update({name: l.item() for name, l in losses.items()})
                num_samples += len(valid_batch)

                if not sample and random.random() < 0.02:
                    self.model.sample(valid_batch)

                del valid_batch

            return {name: loss/num_samples for name, loss in total_losses.items()}

    @staticmethod
    def tabulate_format(scores: Dict) -> str:
        return tabulate([[key, [(n.rsplit('_', maxsplit=1)[1], '%0.5f' % s) for n, s in values]]
                         for key, values in
                         itertools.groupby(scores.items(), key=lambda item: item[0].rsplit('_', maxsplit=1)[0])])

    def train(self):
        total_losses, total_sen, start = Counter(), 1e-10, time.time()
        checkpoint_losses = deque()

        for step, batch in tqdm(enumerate(self.train_it, start=1), total=len(self.train_it)):

            losses = self.train_one(batch)

            total_losses.update(losses)
            total_sen += len(batch)

            del batch

            if step % self.valid_step == 0:
                train_speed = total_sen / (time.time() - start)

                inference_start = time.time()
                valid_losses = self.valid(self.valid_it)

                self.checkpoint(checkpoint_losses, sum(valid_losses.values()))

                for name, loss in total_losses.items():
                    print("%s train loss=%.6f\t\tvalid loss=%.6f" % (name, loss / total_sen, valid_losses[name]))
                    self.summary_writer.add_scalars(
                        'loss',
                        {'train_%s' % name: loss / total_sen, 'valid_%s' % name: valid_losses[name]},
                        step)

                print("speed:   train %.2f sentence/s  valid %.2f sentence/s\n\n" %
                      (train_speed, len(self.valid_it.dataset) / (time.time() - inference_start)))

                total_losses, total_sen, start = Counter(), 1e-10, time.time()

            if self.train_it.iterations % (self.valid_step * 5) == 0:
                with torch.no_grad():
                    '''
                    print([(embedding.name, embedding.scale_ratio.item()) for embedding in self.model.tag_embeddings])
                    self.summary_writer.add_scalars(
                        'tag_weights',
                        {embedding.name: embedding.scale_ratio.item() for embedding in self.model.tag_embeddings},
                        step)
                    '''
                    eval_start = time.time()
                    results = self.model.evaluation(self.valid_it)
                    print("speed: eval %.2f sentence/s" % (len(self.valid_it.dataset)/(time.time() - eval_start)))
                    for name, result in results.items():
                        print('------- %s -------' % name)
                        print(self.tabulate_format(result))
                        self.summary_writer.add_scalars('%s_eval_valid' % name, result, step)

                    results = self.model.evaluation(self.test_it)
                    for name, result in results.items():
                        print('------- %s -------' % name)
                        print(self.tabulate_format(result))
                        self.summary_writer.add_scalars('%s_eval_test' % name, result, step)

    def checkpoint(self, checkpoint_losses, valid_loss):
        if len(checkpoint_losses) == 0 or checkpoint_losses[-1] > valid_loss:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

            checkpoint_losses.append(valid_loss)

            torch.save(self.state_dict(),
                       '%s/model-%0.4f' % (self.checkpoint_dir, valid_loss))

            if len(checkpoint_losses) > 5:
                removed = checkpoint_losses.popleft()
                os.remove('%s/model-%0.4f' % (self.checkpoint_dir, removed))

    @classmethod
    def create(cls, coarse_config, checkpoint, fine_config):
        if checkpoint is not None:
            TEXT, label_fields = Trainer.load_voc(coarse_config)
            model = Trainer.load_model(coarse_config, TEXT, label_fields)
            states = torch.load(checkpoint, map_location=fine_config.device)
            model.load_state_dict(states['model'])
            embedding, encoder = model.embedding, model.encoder
        else:
            TEXT = data.Field(include_lengths=True, init_token=INIT_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN)
            embedding, encoder = None, None

        pos_field = PartialField(init_token=INIT_TOKEN, eos_token=EOS_TOKEN)
        chunk_field = PartialField(init_token=INIT_TOKEN, eos_token=EOS_TOKEN)

        train, valid, test = fine_config.dataset_class.splits(
            path=fine_config.data_dir,
            train=fine_config.train,
            validation=fine_config.valid,
            test=fine_config.test,
            fields=[('text', TEXT), ('pos', pos_field), ('chunk', chunk_field)])

        if checkpoint is None and not hasattr(TEXT, 'vocab'):
            TEXT.build_vocab(train, min_freq=fine_config.text_min_freq)

        pos_field.build_vocab(train, min_freq=fine_config.tag_min_freq)
        chunk_field.build_vocab(train, min_freq=fine_config.tag_min_freq)

        train_it, valid_it, test_it = \
            BucketIterator.splits([train, valid, test],
                                  batch_sizes=fine_config.batch_sizes,
                                  shuffle=True,
                                  device=fine_config.device,
                                  sort_within_batch=True)

        train_it.repeat = True

        if embedding is None:
            embedding = nn.Embedding(
                len(TEXT.vocab),
                fine_config.embedding_dim,
                padding_idx=TEXT.vocab.stoi[PAD_TOKEN],
                scale_grad_by_freq=True
            )

        if encoder is None:
            encoder = nn.LSTM(
                fine_config.embedding_dim,
                fine_config.encoder_hidden_dim // 2,
                fine_config.encoder_num_layers,
                bidirectional=True,
                # residual=fine_config.encoder_residual,
                dropout=0.5)

        pos_crf = LinearCRF(encoder.hidden_size * 2,
                            len(pos_field.vocab),
                            pos_field.vocab.transition_constraints,
                            attention_num_heads=fine_config.attention_num_heads,
                            dropout=0.3)
        chunk_crf = LinearCRF(encoder.hidden_size * 2,
                              len(chunk_field.vocab),
                              chunk_field.vocab.transition_constraints,
                              attention_num_heads=fine_config.attention_num_heads,
                              dropout=0.3)

        fine_model = Tagger(TEXT.vocab, {'pos': pos_field.vocab, 'chunk': chunk_field.vocab},
                            embedding,
                            encoder,
                            nn.ModuleDict({
                                'pos': pos_crf,
                                'chunk': chunk_crf}))

        fine_model.to(fine_config.device)

        return cls(fine_config,
                   fine_model,
                   train_it, valid_it, test_it,
                   fine_config.valid_step,
                   fine_config.checkpoint_path)


class NERConfig:
    def __init__(self, model_dir: str = None):
        self.data_dir = './ner/data'
        self.model_dir = model_dir if model_dir else './ner/model'

        os.makedirs(self.model_dir, exist_ok=True)

        self.train = 'example.train'
        self.valid = 'example.dev'
        self.test = 'example.test'

        self.dataset_class = NamedEntityData

        self.batch_sizes = [16, 32, 32]

        # vocabulary
        self.text_min_freq = 5
        # text_min_size = 50000

        self.tag_min_freq = 5
        # tag_min_size = 50000

        self.common_size = 1000

        self.char_tag_dim = 10

        self.taggers = []  # [(radical, 32)]

        self.ngram_taggers = [place_ngram, person_ngram, digit_ngram, quantifier_ngram, idioms_ngram, org_ngram]
        # (jieba_pos, 64), (place, 8), (person, 8), (idioms, 8), (organizations, 8), ]

        # model
        self.embedding_dim = 256
        self.encoder_hidden_dim = 256
        self.encoder_num_layers = 2
        self.encoder_residual = False
        self.attention_num_heads = 8

        self.loss = 'nll'  # ['nll', 'focal_loss']

        self.device = torch.device('cpu')  # torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.cached_dataset_prefix = os.path.join(self.data_dir, 'dataset')
        self.checkpoint_path = os.path.join(self.model_dir, 'checkpoint')
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # summary parameters
        self.summary_dir = os.path.join(self.model_dir, 'summary')
        os.makedirs(self.summary_dir, exist_ok=True)

        self.valid_step = 400


class CTBPOSConfig:
    def __init__(self, model_dir: str = None):
        self.data_dir = './pos/data'
        self.model_dir = model_dir if model_dir else './pos/model'
        os.makedirs(self.model_dir, exist_ok=True)

        self.train = 'train'
        self.valid = 'valid'
        self.test = 'gold'

        self.dataset_class = POSData

        self.batch_sizes = [16, 32, 32]

        # vocabulary
        self.text_min_freq = 5
        # text_min_size = 50000

        self.tag_min_freq = 1
        # tag_min_size = 50000

        self.common_size = 1000

        # model
        self.vocab_size = 100
        self.embedding_dim = 64
        self.encoder_hidden_dim = 64
        self.encoder_num_layers = 2
        self.encoder_residual = False
        self.attention_num_heads = None

        self.taggers = []  # [(radical, 32)]

        self.ngram_taggers = [] # [place_ngram, person_ngram, digit_ngram, quantifier_ngram, idioms_ngram, org_ngram]
        # (jieba_pos, 64), (place, 8), (person, 8), (idioms, 8), (organizations, 8), ]

        self.loss = 'nll'  # ['nll', 'focal_loss']

        self.device = torch.device('cpu')  # torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.cached_dataset_prefix = os.path.join(self.data_dir, 'dataset')
        self.checkpoint_path = os.path.join(self.model_dir, 'checkpoint')
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # summary parameters
        self.summary_dir = os.path.join(self.model_dir, 'summary')
        os.makedirs(self.summary_dir, exist_ok=True)

        self.valid_step = 200


class CNC_config:
    def __init__(self, model_dir: str = None):
        self.data_dir = './cnc/data'
        self.model_dir = model_dir if model_dir else os.path.join(self.data_dir, 'model')
        os.makedirs(self.model_dir, exist_ok=True)

        self.train = 'cnc_train.txt'
        self.valid = 'cnc_dev.txt'
        self.test = 'cnc_test.txt'

        self.dataset_class = POSData

        self.batch_sizes = [16, 64, 64]

        # vocabulary
        self.text_min_freq = 5
        # text_min_size = 50000

        self.tag_min_freq = 1
        # tag_min_size = 50000

        self.common_size = 1000

        # model
        self.vocab_size = 100
        self.embedding_dim = 64
        self.encoder_hidden_dim = 64
        self.encoder_num_layers = 1
        self.encoder_residual = False
        self.attention_num_heads = None

        self.taggers = []  # [(radical, 32)]

        self.ngram_taggers = [] # [place_ngram, person_ngram, digit_ngram, quantifier_ngram, idioms_ngram, org_ngram]
        # (jieba_pos, 64), (place, 8), (person, 8), (idioms, 8), (organizations, 8), ]

        self.loss = 'nll'  # ['nll', 'focal_loss']

        self.device = torch.device('cpu')  # torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.cached_dataset_prefix = os.path.join(self.data_dir, 'dataset')
        self.checkpoint_path = os.path.join(self.model_dir, 'checkpoint')
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # summary parameters
        self.summary_dir = os.path.join(self.model_dir, 'summary')
        os.makedirs(self.summary_dir, exist_ok=True)

        self.valid_step = 500


class PeopleConfig:
    def __init__(self, model_dir: str = None):
        self.data_dir = 'people'
        self.model_dir = model_dir if model_dir else './ner/model'
        os.makedirs(self.model_dir, exist_ok=True)

        self.train = '2014'
        self.valid = None
        self.test = None

        self.dataset_class = People2014

        self.batch_sizes = [16, 32, 32]

        # vocabulary
        self.text_min_freq = 5
        # text_min_size = 50000

        self.tag_min_freq = 1
        # tag_min_size = 50000

        self.common_size = 1000

        # model
        self.vocab_size = 100
        self.embedding_dim = 64
        self.encoder_hidden_dim = 64
        self.encoder_num_layers = 2
        self.encoder_residual = False
        self.attention_num_heads = None

        self.taggers = []  # [(radical, 32)]

        self.ngram_taggers = []  # [place_ngram, person_ngram, digit_ngram, quantifier_ngram, idioms_ngram, org_ngram]
        # (jieba_pos, 64), (place, 8), (person, 8), (idioms, 8), (organizations, 8), ]

        self.loss = 'nll'  # ['nll', 'focal_loss']

        self.device = torch.device('cpu')  # torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.cached_dataset_prefix = os.path.join(self.data_dir, 'dataset')
        self.checkpoint_path = os.path.join(self.model_dir, 'checkpoint')
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # summary parameters
        self.summary_dir = os.path.join(self.model_dir, 'summary')
        os.makedirs(self.summary_dir, exist_ok=True)

        self.valid_step = 200


class ChunkConfig:
    def __init__(self, model_dir: str = None):
        self.data_dir = 'chunk'
        self.model_dir = model_dir if model_dir else './chunk/model'
        os.makedirs(self.model_dir, exist_ok=True)

        self.train = 'data/train.txt'
        self.valid = 'data/dev.txt'
        self.test = 'data/test.txt'

        self.dataset_class = ChunkDataset

        self.batch_sizes = [16, 32, 32]

        # vocabulary
        self.text_min_freq = 1
        # text_min_size = 50000

        self.tag_min_freq = 1
        # tag_min_size = 50000

        self.common_size = 1000

        # model
        self.vocab_size = 100
        self.embedding_dim = 128
        self.encoder_hidden_dim = 128
        self.encoder_num_layers = 2
        self.encoder_residual = False
        self.attention_num_heads = 8

        self.taggers = []  # [(radical, 32)]

        self.ngram_taggers = []  # [place_ngram, person_ngram, digit_ngram, quantifier_ngram, idioms_ngram, org_ngram]
        # (jieba_pos, 64), (place, 8), (person, 8), (idioms, 8), (organizations, 8), ]

        self.loss = 'nll'  # ['nll', 'focal_loss']

        self.device = torch.device('cpu')  # torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.cached_dataset_prefix = os.path.join(self.data_dir, 'dataset')
        self.checkpoint_path = os.path.join(self.model_dir, 'checkpoint')
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # summary parameters
        self.summary_dir = os.path.join(self.model_dir, 'summary')
        os.makedirs(self.summary_dir, exist_ok=True)

        self.valid_step = 400


configs = {'ctb': CTBPOSConfig, 'ner': NERConfig, 'people': PeopleConfig, 'cnc': CNC_config, 'chunk': ChunkConfig}

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Preprocess baike corpus and save vocabulary')
    argparser.add_argument('--checkpoint', type=str, help='checkpoint path')
    argparser.add_argument('--task', type=str, help='task type, [ctb, ner, people, cnc, chunk]')

    args = argparser.parse_args()

    assert args.task in configs

    trainer = FineTrainer.create(Config(), args.checkpoint, configs[args.task]())

    trainer.train()
