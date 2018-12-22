#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import os
import random
import time
from collections import defaultdict, OrderedDict, deque, Counter
from typing import List, Tuple, Dict
from tqdm import tqdm
import argparse


import torch
from torch import nn, optim
from torchtext import data, vocab
from torchtext.data.iterator import BucketIterator
from .encoder import StackLSTM
from ..transformer.crf import LinearCRF, MaskedCRF
from ..transformer.vocab import TagVocab
from ..transformer.field import PartialField
from .train import Trainer, CoarseConfig
from .base import INIT_TOKEN, EOS_TOKEN, MASK_TOKEN, PAD_TOKEN
from .embedding import WindowEmbedding

from tensorboardX import SummaryWriter


class NamedEntityData(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path: str, fields: List, **kwargs):
        examples = [data.Example.fromlist(self._toBMES(chars, types), fields)
                    for chars, types in self._getline(path)]
        super(NamedEntityData, self).__init__(examples, fields, **kwargs)

    def _getline(self, path):
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

    def _toBMES(self, chars, types):
        n_chars, n_types = [], []
        buffer_c, buffer_t = [], []
        for char, type in zip(chars, types):
            if type[0] in ['B', 'O'] and len(buffer_c) > 0:
                tag = buffer_t[0][2:] if buffer_t[0].startswith('B-') else buffer_t[0]

                if len(buffer_t) == 1:
                    buffer_t = ['S_' + tag]
                elif len(buffer_t) >= 2:
                    buffer_t = ['B_' + tag] + ['M_' + tag] * (len(buffer_t) - 2) + ['E_' + tag]

                n_chars.extend(buffer_c)
                n_types.extend(buffer_t)

                buffer_c, buffer_t = [], []

            if type[0] in ['B', 'O']:
                buffer_c, buffer_t = [char], [type]
            elif type[0] == 'I':
                buffer_c.append(char)
                buffer_t.append(type)
        return n_chars, n_types


class Tagger(nn.Module):
    def __init__(self, words: vocab.Vocab, tags: TagVocab,
                 embedding: nn.Embedding, encoder: StackLSTM, crf: LinearCRF):
        super(Tagger, self).__init__()
        self.words = words
        self.tags = tags

        self.embedding = embedding
        self.encoder = encoder
        self.crf = crf

    def _make_masks(self, sens, lens):
        masks = sens.new_ones(sens.size(), dtype=torch.uint8)
        for i, l in enumerate(lens):
            masks[l:, i] = 0
        return masks

    def _encode(self, batch: data.Batch):
        sens, lens = batch.text
        token_masks = self._make_masks(sens, lens)
        emb = self.embedding(sens)
        if hasattr(batch, 'lattices'):
            return self.encoder(emb, token_masks, batch.lattices), token_masks
        else:
            return self.encoder(emb, token_masks), token_masks

    def predict(self, batch: data.Batch):
        sens, lens = batch.text
        hiddens, token_masks = self._encode(batch)
        return self.crf(hiddens, lens, token_masks)

    def nbest(self, batch: data.Batch):
        sens, lens = batch.text
        hiddens, token_masks = self._encode(batch)
        return self.crf.nbest(hiddens, lens, token_masks, 5)

    def criterion(self, batch: data.Batch):
        sens, lens = batch.text
        tag_masks, tags = batch.tags
        hidden, token_masks = self._encode(batch)
        return self.crf.neg_log_likelihood(hidden, lens, token_masks, tag_masks, tags)

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
        pred_tags = self.predict(batch)

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

            #print(gold_tags)
            for i in range(len(text_len)):
                pred, score = preds[i]
                _correct_counts, _gold_counts, _pred_counts = self.evaluation_one(
                    [self.tags.itos[p] for p in pred[1:-1]], golds[i][1:text_len[i]-1])

                correct_counts.update(_correct_counts)
                gold_counts.update(_gold_counts)
                pred_counts.update(_pred_counts)

        total_correct = sum(correct_counts.values())
        total_gold = sum(gold_counts.values())
        total_pred = sum(pred_counts.values())

        results = {'total_f1': total_correct*2/(total_gold+total_pred+1e-5),
                   'total_prec': total_correct/(total_gold+1e-5),
                   'total_recall': total_correct/(total_pred+1e-5)}
        for name in gold_counts.keys():
            results['%s_f1' % name] = correct_counts[name]*2/(pred_counts[name]+gold_counts[name]+1e-5)
            results['%s_prec' % name] = correct_counts[name]/(pred_counts[name] + 1e-5)
            results['%s_recall' % name] = correct_counts[name]/(gold_counts[name] + 1e-5)

        return results

    def coarse_params(self):
        yield from self.embedding.parameters()
        yield from self.encoder.parameters()
        yield from self.crf.parameters()

    def fine_params(self):
        yield from self.embedding.parameters()
        yield from self.crf.parameters()


class FineTrainer:
    def __init__(self, config, model: Tagger,
                 partial_train_it, partial_valid_it, partial_test_it,
                 valid_step, checkpoint_dir):
        self.config = config
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.train_it, self.valid_it, self.test_it = \
            partial_train_it, partial_valid_it, partial_test_it

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

            return total_loss / num_samples, num_samples

    def train(self):
        total_loss, total_sen, start = 0, 1e-10, time.time()
        checkpoint_losses = deque()

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
            os.makedirs(self.checkpoint_dir, exist_ok=True)

            checkpoint_losses.append(valid_loss)

            torch.save(self.state_dict(),
                       '%s/ctb-pos-%0.4f' % (self.checkpoint_dir, valid_loss))

            if len(checkpoint_losses) > 5:
                removed = checkpoint_losses.popleft()
                os.remove('%s/ctb-pos-%0.4f' % (self.checkpoint_dir, removed))

    @classmethod
    def create(cls, coarse_config, checkpoint, fine_config):
        if checkpoint is not None:
            TEXT, KEY_LABEL, ATTR_LABEL, SUB_LABEL, ENTITY_LABEL = Trainer.load_voc(coarse_config)
            model = Trainer.load_model(coarse_config, TEXT, KEY_LABEL, ATTR_LABEL, SUB_LABEL, ENTITY_LABEL)
            states = torch.load(checkpoint)
            model.load_state_dict(states['model'])
            embedding, encoder = model.embedding, model.encoder
        else:
            TEXT = data.Field(include_lengths=True, init_token=INIT_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN)
            embedding, encoder = None, None

        tag_field = PartialField(init_token=INIT_TOKEN, eos_token=EOS_TOKEN)
        train, valid, test = NamedEntityData.splits(
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


class FineConfig:
    def __init__(self):
        self.partial_prefix = './ner/data'
        self.full_prefix = './ner/data'

        self.train = 'example.train'
        self.valid = 'example.dev'
        self.test = 'example.test'

        self.batch_sizes = [16, 32, 32]

        # vocabulary
        self.text_min_freq = 10
        # text_min_size = 50000

        self.tag_min_freq = 10
        # tag_min_size = 50000

        self.common_size = 1000

        # model
        self.vocab_size = 100
        self.embedding_size = 128
        self.encoder_size = 256
        self.attention_num_heads = 8
        self.encoder_depth = 2

        self.device = torch.device('cpu') # torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.cached_dataset_prefix = './ner/dataset'
        self.checkpoint_path = './ner/attention/model/mode_{}_emb_{}_hidden_{}_layer_{}_head_{}'.format(
            'lstm', self.embedding_size, self.encoder_size, self.encoder_depth, self.attention_num_heads)

        # summary parameters
        self.summary_dir = os.path.join('./ner/attention', 'summary')
        os.makedirs(self.summary_dir, exist_ok=True)

        self.valid_step = 100


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Preprocess baike corpus and save vocabulary')
    argparser.add_argument('--checkpoint', type=str, help='checkpoint path')

    args = argparser.parse_args()

    trainer = FineTrainer.create(CoarseConfig(), args.checkpoint, FineConfig())

    trainer.train()



