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
from .base import INIT_TOKEN, EOS_TOKEN, PAD_TOKEN, make_masks, bio_to_bmeso, listfile
from .crf import LinearCRF, MaskedCRF
from .encoder import ElmoEncoder, LSTM
from .tags import *
from .train import Trainer, Config
from ..transformer.field import PartialField
from ..transformer.vocab import TagVocab


def add_boundary(tokens: List[str], name: str=None) -> List[str]:
    if name:
        return ['<%s>' % name] + tokens + ['</%s>' % name]
    else:
        return [INIT_TOKEN] + tokens + [EOS_TOKEN]


class POSDataset(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, name, path: str, fields: List, **kwargs):
        self.name = name
        examples = [data.Example.fromlist((chars, types), fields)
                    for chars, types in self.get_line(path)]
        super(POSDataset, self).__init__(examples, fields, **kwargs)

    def get_line(self, path):
        return NotImplementedError()

    @classmethod
    def splits(cls, name,
               path=None, root='.data',
               train=None, validation=None, test=None,
               **kwargs):
        train_data = None if train is None else cls(
            name, os.path.join(path, train), **kwargs)
        val_data = None if validation is None else cls(
            name, os.path.join(path, validation), **kwargs)
        test_data = None if test is None else cls(
            name, os.path.join(path, test), **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)


class CTB(POSDataset):

    def get_line(self, path):
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
                yield add_boundary(chars, self.name), add_boundary(types)


class People2014(POSDataset):
    def get_line(self, path):
        for line in tqdm(self.read_line(path), desc='load data from %s' % path):
            try:
                chars, tags = [], []
                for word, tag in self.fix_tokenize(line):
                    chars.extend(list(word))
                    tags.extend(to_bmes(len(word), tag))
                yield add_boundary(chars, self.name), add_boundary(tags)
            except Exception as e:
                print(e)
                print(line)
                print(self.fix_tokenize(line))

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

    def _read_line(self, path: str):
        if os.path.isdir(path):
            for name in os.listdir(path):
                child = os.path.join(path, name)
                yield from self._read_line(child)
        elif path.endswith('.txt'):
            with open(path) as file:
                for line in file:
                    line = line.strip()
                    if len(line) > 0:
                        yield line
        else:
            print('%s is not loaded.' % path)


class MSRA(POSDataset):
    def get_line(self, path):
        for line in tqdm(self.read_line(path), desc='load data from %s' % path):
            try:
                chars, tags = [], []
                for word, tag in self.fix_tokenize(line):
                    chars.extend(list(word))
                    tags.extend(to_bmes(len(word), tag))

                yield add_boundary(chars, self.name), add_boundary(tags)
            except Exception as e:
                print(e)
                print(line)
                print(self.fix_tokenize(line))

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


class Tagger(nn.Module):
    def __init__(self, words: vocab.Vocab, tags: TagVocab,
                 embedding: nn.Embedding,
                 ngram_fields: List[NgramField],
                 elmo_encoder: ElmoEncoder,
                 encoder: nn.LSTM,
                 attention: MultiHeadedAttention,
                 crf: LinearCRF):
        super(Tagger, self).__init__()
        self.words = words
        self.tags = tags

        self.embedding = embedding
        self.ngram_fields = ngram_fields
        self.dropout = nn.Dropout(0.5)
        self.elmo_encoder = elmo_encoder
        self.encoder = encoder
        self.attention = attention
        self.crf = crf

    def _embedding(self, tokens, lens):
        embed = self.embedding(tokens)
        if self.elmo_encoder is None:
            return embed

        forwards, backwards = self.elmo_encoder(embed, lens)
        return torch.cat([forwards, backwards], dim=-1)

    def _encode(self, batch: data.Batch):
        sens, lens = batch.text
        token_masks = make_masks(sens, lens)
        emb = self._embedding(sens, lens)

        hidden, _ = self.encoder(emb)
        if self.attention:
            hidden = self.attention(hidden, hidden, hidden, token_masks, batch_first=False)
        return hidden, token_masks

    def predict(self, batch: data.Batch):
        sens, lens = batch.text
        hiddens, token_masks = self._encode(batch)
        return self.crf(hiddens, lens)

    def predict_with_prob(self, batch: data.Batch):
        sens, lens = batch.text
        hiddens, token_masks = self._encode(batch)
        return self.crf.predict_with_prob(hiddens, lens)

    def nbest(self, batch: data.Batch):
        sens, lens = batch.text
        hiddens, token_masks = self._encode(batch)
        return self.crf.nbest(hiddens, lens, 5)

    def criterion(self, batch: data.Batch):
        sens, lens = batch.text
        tag_masks, tags = batch.tags
        hidden, token_masks = self._encode(batch)
        return self.crf.neg_log_likelihood(hidden, lens, tag_masks)

    def print_transition(self):
        print(tabulate([self.tags.itos] + self.crf.transition.tolist(),
                       headers="firstrow", tablefmt='grid', floatfmt='3.3g'))

    def print(self, batch: data.Batch):
        text, text_len = batch.text
        gold_masks, gold_tags = batch.tags
        for i in range(len(text_len)):
            length = text_len[i]
            print(' '.join([self.words.itos[w] + '#' + t
                            for w, t in zip(text[0:length, i].data.tolist(), gold_tags[i])]))

    def sample_nbest(self, batch: data.Batch):
        self.eval()
        text, text_len = batch.text
        gold_masks, gold_tags = batch.tags
        nbests = self.nbest(batch)

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
                pred_tag = ''.join(tostr(sen, [self.tags.itos[tag_id] for tag_id in pred_tag]))
                print('pred: %s\nscore: %f' % (pred_tag, score))

    def sample(self, batch: data.Batch):
        self.eval()
        text, text_len = batch.text
        gold_masks, gold_tags = batch.tags
        pred_tags = self.predict_with_prob(batch)

        results = []
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

            pred_tag = [self.tags.itos[tag_id] for tag_id in pred_tag]
            gold_tag = gold_tags[i][:text_len[i]]
            if pred_tag != gold_tag:
                pred_tag = ''.join(tostr(sen, pred_tag))
                gold_tag = ''.join(tostr(sen, gold_tag))
                print('\ngold: %s\npred: %s\nscore: %f' % (gold_tag, pred_tag, score))

        return results

    def evaluation_one(self, preds: List[str],
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

    def evaluation(self, data_it) -> Dict[str, float]:
        self.eval()
        correct_counts, gold_counts, pred_counts = Counter(), Counter(), Counter()
        for batch in tqdm(data_it, desc='eval', total=len(data_it)):
            _, text_len = batch.text
            masks, golds = batch.tags
            preds = self.predict(batch)

            # print(gold_tags)
            for i in range(len(text_len)):
                pred, score = preds[i]
                _correct_counts, _gold_counts, _pred_counts = self.evaluation_one(
                    [self.tags.itos[p] for p in pred[1:-1]], golds[i][1:text_len[i] - 1])

                correct_counts.update(_correct_counts)
                gold_counts.update(_gold_counts)
                pred_counts.update(_pred_counts)

        total_correct = sum(correct_counts.values())
        total_gold = sum(gold_counts.values())
        total_pred = sum(pred_counts.values())

        results = {'total_f1': total_correct * 2 / (total_gold + total_pred + 1e-5),
                   'total_prec': total_correct / (total_gold + 1e-5),
                   'total_recall': total_correct / (total_pred + 1e-5)}
        for name in gold_counts.keys():
            results['%s_f1' % name] = correct_counts[name] * 2 / (pred_counts[name] + gold_counts[name] + 1e-5)
            results['%s_prec' % name] = correct_counts[name] / (pred_counts[name] + 1e-5)
            results['%s_recall' % name] = correct_counts[name] / (gold_counts[name] + 1e-5)

        return results


class FineTrainer:
    def __init__(self, config, model: Tagger,
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

        return rloss, num_sen

    def valid(self, valid_it) -> Tuple[torch.Tensor, float]:
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

                del valid_batch

            # self.model.print_transition()

            return total_loss / num_samples, num_samples

    @staticmethod
    def tabulate_format(scores: Dict) -> str:
        return tabulate([[key, [(n.rsplit('_', maxsplit=1)[1], '%0.5f' % s) for n, s in values]]
                         for key, values in
                         itertools.groupby(scores.items(), key=lambda item: item[0].rsplit('_', maxsplit=1)[0])])

    def train(self):
        total_loss, total_sen, start = 0, 1e-10, time.time()
        checkpoint_losses = deque()

        for step, batch in tqdm(enumerate(self.train_it, start=1), total=len(self.train_it)):

            loss, num_sen = self.train_one(batch, scale=1.0)

            del batch

            total_loss += loss
            total_sen += num_sen

            if step % self.valid_step == 0:
                train_speed = total_sen / (time.time() - start)

                inference_start = time.time()
                valid_loss, valid_num_samples = self.valid(self.valid_it)

                self.checkpoint(checkpoint_losses, valid_loss)

                for field in self.model.ngram_fields:
                    field.tagger.print_stats()

                print("train loss=%.6f\t\tvalid loss=%.6f" % (total_loss / total_sen, valid_loss))
                print("speed:   train %.2f sentence/s  valid %.2f sentence/s\n\n" %
                      (train_speed, valid_num_samples / (time.time() - inference_start)))

                self.summary_writer.add_scalars('loss', {'train': total_loss / total_sen, 'valid': valid_loss}, step)

                total_loss, total_sen, start = 0, 1e-10, time.time()

            if self.train_it.iterations % (self.valid_step * 5) == 0:
                with torch.no_grad():
                    '''
                    print([(embedding.name, embedding.scale_ratio.item()) for embedding in self.model.tag_embeddings])
                    self.summary_writer.add_scalars(
                        'tag_weights',
                        {embedding.name: embedding.scale_ratio.item() for embedding in self.model.tag_embeddings},
                        step)
                    '''
                    valid_res = self.model.evaluation(self.valid_it)
                    print(self.tabulate_format(valid_res))
                    self.summary_writer.add_scalars('eval_valid', valid_res, step)

                    eval_res = self.model.evaluation(self.test_it)
                    print(self.tabulate_format(eval_res))
                    self.summary_writer.add_scalars('eval_test', eval_res, step)

    def checkpoint(self, checkpoint_losses, valid_loss):
        if len(checkpoint_losses) == 0 or checkpoint_losses[-1] > valid_loss:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

            checkpoint_losses.append(valid_loss)

            torch.save(self.state_dict(),
                       '%s/model-%0.4f' % (self.checkpoint_dir, valid_loss))

            if len(checkpoint_losses) > 5:
                removed = checkpoint_losses.popleft()
                os.remove('%s/model-%0.4f' % (self.checkpoint_dir, removed))

    def load_datasets(self, config, text_field: data.Field):

        if text_field is None:
            text_field = data.Field(include_lengths=True, pad_token=PAD_TOKEN)

        datasets = []
        for name, (train, valid, test) in config.datasets:
            tag_field = PartialField()
            train, valid, test = POSData.splits(
                name=name,
                path=config.data_dir,
                train=train,
                validation=valid,
                test=test,
                fields=[('text', text_field), ('tags', tag_field)]
            )
            tag_field.build_vocab(train, min_freq=config.tag_min_freq)
            datasets.append((name, text_field, tag_field, train, valid, test))

        text_field.build_vocab()

    @classmethod
    def create(cls, coarse_config, checkpoint, fine_config):
        if checkpoint is not None:
            TEXT, KEY_LABEL, ATTR_LABEL, SUB_LABEL, ENTITY_LABEL = Trainer.load_voc(coarse_config)
            model = Trainer.load_elmo_model(coarse_config, TEXT, KEY_LABEL, ATTR_LABEL, SUB_LABEL, ENTITY_LABEL)
            states = torch.load(checkpoint, map_location=fine_config.device)
            model.load_state_dict(states['model'])
            embedding, elmo_encoder = model.embedding, model.encoder
        else:
            TEXT = data.Field(include_lengths=True, pad_token=PAD_TOKEN)
            embedding, elmo_encoder = None, None

        tag_field = PartialField()

        train, valid, test = fine_config.dataset_class.splits(
            path=fine_config.data_dir,
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
                                  device=fine_config.device,
                                  sort_within_batch=True)

        train_it.repeat = True

        print('voc', '\n'.join(
            ['%s:%d %s' % (field.name, len(field.vocab), ' '.join(field.vocab.itos)) for field in char_fields]))

        if checkpoint is None:
            embedding = nn.Embedding(
                len(TEXT.vocab),
                fine_config.embedding_dim,
                padding_idx=TEXT.vocab.stoi[PAD_TOKEN])

            encoder = nn.LSTM(
                fine_config.embedding_dim,
                fine_config.encoder_hidden_dim // 2,
                fine_config.encoder_num_layers,
                bidirectional=True,
                # residual=fine_config.encoder_residual,
                dropout=0.5)
        else:
            encoder = nn.LSTM(
                coarse_config.encoder_hidden_dim * 2,
                fine_config.encoder_hidden_dim // 2,
                1,
                bidirectional=True,
                # residual=fine_config.encoder_residual,
                dropout=0.5)

        if fine_config.attention_num_heads:
            attention = MultiHeadedAttention(fine_config.attention_num_heads,
                                             fine_config.encoder_hidden_dim,
                                             dropout=0.3)
        else:
            attention = None

        crf = LinearCRF(fine_config.encoder_hidden_dim,
                        len(tag_field.vocab),
                        tag_field.vocab.transition_constraints,
                        dropout=0.3)
        fine_model = Tagger(TEXT.vocab, tag_field.vocab,
                            embedding,
                            elmo_encoder,
                            encoder,
                            attention,
                            crf)

        fine_model.to(fine_config.device)

        return cls(fine_config,
                   fine_model,
                   train_it, valid_it, test_it,
                   fine_config.valid_step,
                   fine_config.checkpoint_path)


class Config:
    def __init__(self, model_dir: str = None):
        self.data_dir = './all_pos/data'
        self.model_dir = model_dir if model_dir else './all_pos/model'
        os.makedirs(self.model_dir, exist_ok=True)

        self.train = 'train'
        self.valid = 'valid'
        self.test = 'gold'

        self.datasets = {
            'ctb': ('ctb.train', 'ctb.valid', 'ctb.gold'),
            'cnc': ('cnc.train', 'cnc.valid', 'cnc.gold'),
            'people2014': ('people2014.train', None, None),
            'msra': ('msra.train', 'msra.valid')
        }

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


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Preprocess baike corpus and save vocabulary')
    argparser.add_argument('--checkpoint', type=str, help='checkpoint path')
    argparser.add_argument('--task', type=str, help='task type, [ctb, ner, people, cnc]')

    args = argparser.parse_args()

    trainer = FineTrainer.create(Config(), args.checkpoint, Config())

    trainer.train()

