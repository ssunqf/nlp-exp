#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

# http://nlp.seas.harvard.edu/2018/04/03/attention.html


import torch
import torch.nn as nn
import math, copy
from torch.autograd import Variable
from .base import clones, AddNormLayer, PositionwiseFeedForward, PAD
from .lattice import LatticeEncoderLayer


#-------------------------------------------- embedding -------------------------------------------
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x


# token embedding  * math.sqrt(embed_dim) + pos_emb
class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx=None, dropout=0.3):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos = PositionalEncoding(embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_dim = embed_dim

    def forward(self, x):
        return self.dropout(self.pos(self.lut(x)))


# -------------------------------------------------------

# attention -> (add + norm) -> ffn -> (add + norm)
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout=0.3):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(AddNormLayer(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# ------------------------------------------------------- Encoder --------------------------------------



from torchtext.vocab import Vocab
from torchtext import data
from task.pretrained.transformer.vocab import TagVocab
from typing import List
from .crf import LinearCRF, MaskedCRF
from collections import defaultdict

from .attention import MultiHeadedAttention
from .encoder import Encoder, LatticeEncoder, LSTMEncoder

class Tagger(nn.Module):
    def __init__(self, words: Vocab, tags: TagVocab,
                 embedding: Embeddings, encoder: Encoder, crf: LinearCRF):
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

    def evaluation_one(self, preds: List[str], golds: List[str]):
        assert len(preds) == len(golds)
        correct = 0
        true = 0

        corrects = defaultdict(int)
        trues = defaultdict(int)
        begin = 0
        while begin < len(golds):
            end = begin
            while end < len(golds):
                if golds[end].startswith('S_') or golds[end].startswith('E_') or golds[end] in ['O', '*']:
                    end += 1
                    break
                else:
                    end += 1
            if not golds[begin].endswith('*'):
                tag_type = golds[begin][2:]
                trues[tag_type] += 1
                true += 1
                if preds[begin:end] == golds[begin:end]:
                    corrects[tag_type] += 1
                    correct += 1

            begin = end
        return correct, true

    def evaluation(self, data_it):
        self.eval()
        correct, true, pos = 0, 0, 0
        for batch in data_it:
            _, text_len = batch.text
            masks, golds = batch.tags
            preds = self.predict(batch)

            #print(gold_tags)
            for i in range(len(text_len)):
                pred, score = preds[i]
                c, t = self.evaluation_one([self.tags.itos[p] for p in pred], golds[i][0:text_len[i]])
                correct += c
                true += t

        recall = correct/float(true+1e-5)
        return {'recall':recall}

    def coarse_params(self):
        yield from self.embedding.parameters()
        yield from self.encoder.parameters()
        yield from self.crf.parameters()

    def fine_params(self):
        yield from self.embedding.parameters()
        yield from self.crf.parameters()

    @staticmethod
    def create(words: Vocab, tags: TagVocab,
               embedding_dim: int,
               encoder_dim: int, encoder_depth: int,
               attention_num_head: int, atten_window_size=None):
        embedding = Embeddings(len(words), embedding_dim, padding_idx=words.stoi[PAD], dropout=0.3)
        attention = MultiHeadedAttention(attention_num_head, encoder_dim, atten_window_size=atten_window_size, dropout=0.3)
        ffn = PositionwiseFeedForward(encoder_dim, encoder_dim, dropout=0.3)
        transformer = EncoderLayer(encoder_dim, attention, ffn, dropout=0.3)
        encoder = Encoder(transformer, encoder_depth)
        crf = MaskedCRF(encoder_dim, len(tags), tags.transition_constraints)

        return Tagger(words, tags, embedding, encoder, crf)

    @staticmethod
    def createLattice(words: Vocab, tags: TagVocab,
                      embedding_dim: int, encoder_dim: int, encoder_depth: int,
                      attention_num_head: int,
                      pretrained_emb: nn.Embedding, max_subword_len: int):

        char_embedding = Embeddings(len(words), embedding_dim, padding_idx=words.stoi[PAD], dropout=0.3)
        attention = MultiHeadedAttention(attention_num_head, encoder_dim, atten_window_size=None, dropout=0.3)
        ffn = PositionwiseFeedForward(encoder_dim, encoder_dim, dropout=0.3)
        transformer = EncoderLayer(encoder_dim, attention, ffn, dropout=0.3)

        latticeLayer = LatticeEncoderLayer(
            encoder_dim, pretrained_emb, max_subword_len, copy.deepcopy(ffn), dropout=0.3)
        encoder = LatticeEncoder(transformer, encoder_depth, latticeLayer)
        # crf = LinearCRF(encoder_dim, len(tags))
        crf = MaskedCRF(encoder_dim, len(tags), tags.transition_constraints)

        return Tagger(words, tags, char_embedding, encoder, crf)


    @staticmethod
    def createLSTM(words: Vocab, tags: TagVocab,
                      embedding_dim: int, encoder_dim: int, encoder_depth: int, attention_num_head: int,
                      pretrained_emb: nn.Embedding, max_subword_len: int):

        # char_embedding = Embeddings(len(words), embedding_dim, padding_idx=words.stoi[PAD])
        char_embedding = nn.Embedding(len(words), embedding_dim, padding_idx=words.stoi[PAD])
        attention = MultiHeadedAttention(attention_num_head, encoder_dim, atten_window_size=None, dropout=0.2)
        ffn = PositionwiseFeedForward(encoder_dim, encoder_dim, dropout=0.2)
        # transformer = EncoderLayer(encoder_dim, attention, ffn, dropout=0.2)
        encoder = LSTMEncoder(encoder_dim, encoder_depth, attention, 0.2)
        latticeLayer = LatticeEncoderLayer(
            encoder_dim, pretrained_emb, max_subword_len, copy.deepcopy(ffn), dropout=0.2)
        # encode = Encoder(lstm, encoder_depth, latticeLayer)
        # encode = Encoder(lstm, encoder_depth)
        # crf = LinearCRF(encoder_dim, len(tags))
        crf = MaskedCRF(encoder_dim, len(tags), tags.transition_constraints)

        return Tagger(words, tags, char_embedding, encoder, crf)

    @staticmethod
    def createCNN(words: Vocab, tags: TagVocab,
                   embedding_dim: int, encoder_dim: int, encoder_depth: int, attention_num_head: int):
        char_embedding = Embeddings(len(words), embedding_dim)

        from module.encoder import Encoder
        encoder = Encoder(embedding_dim, 'CNN', encoder_dim, encoder_depth, attention_num_head)
        # crf = LinearCRF(encoder_dim, len(tags))
        crf = MaskedCRF(encoder_dim, len(tags), tags.transition_constraints)

        return Tagger(words, tags, char_embedding, encoder, crf)


