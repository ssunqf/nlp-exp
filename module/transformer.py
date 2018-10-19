#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

# http://nlp.seas.harvard.edu/2018/04/03/attention.html


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

#-------------------------------------------- embedding -------------------------------------------

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

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
        return self.dropout(x)

# token embedding  * math.sqrt(embed_dim) + pos_emb
class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, embed_dim)
        self.pos = PositionalEncoding(embed_dim, dropout=0.1)
        self.embed_dim = embed_dim

    def forward(self, x):
        return self.pos(self.lut(x) * math.sqrt(self.embed_dim))


# ------------------------------------------------------ Attention -------------------------------------

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    head_dim = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_head, hidden_dim, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert hidden_dim % num_head == 0
        # We assume d_v always equals d_k
        self.head_dim = hidden_dim // num_head
        self.num_head = num_head
        self.linears = clones(nn.Linear(hidden_dim, hidden_dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1).unsqueeze(-1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from hidden_dim => num_head x head_dim
        query, key, value = \
            [l(x).view(nbatches, -1, self.num_head, self.head_dim).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.num_head * self.head_dim)
        return self.linears[-1](x)


# ------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, input_dim, ff_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(input_dim, ff_dim)
        self.w_2 = nn.Linear(ff_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# -------------------------------------------------------

# add and norm
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

# attention -> (add + norm) -> ffn -> (add + norm)
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# ------------------------------------------------------- Encoder --------------------------------------
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(dim))
        self.b_2 = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# -------------------------------------------------------- CRF -----------------------------------------
'''
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
'''

def log_sum_exp(vecs, axis=-1):
    max_val, _ = vecs.max(axis)
    vecs = vecs - max_val.unsqueeze(axis)
    out_val = vecs.exp().sum(axis).log()
    # print(max_val, out_val)
    return max_val + out_val


class CRFLayer(nn.Module):
    def __init__(self, hidden_dim, tag_size, begin_id, end_id):
        super(CRFLayer, self).__init__()
        self.hidden2emission = nn.Linear(hidden_dim, tag_size)
        self.transition = nn.Parameter(torch.randn(tag_size, tag_size))
        self.transition.data[begin_id, :] = -10000
        self.transition.data[:, end_id] = -10000
        self.hidden_dim = hidden_dim
        self.tag_size = tag_size
        self.begin_id = begin_id
        self.end_id = end_id

    def _forward_score(self, emissions, lens):
        batch_size, max_len, emission_size = emissions.size()
        assert emission_size == self.tag_size

        forward_var = self.transition[:, self.begin_id].expand(batch_size, -1)

        unfinished = sorted(enumerate(lens), key=lambda x: x[1], reverse=True)
        last_vars = []
        for time in range(max_len):
            # record finished
            while len(unfinished) > 0 and unfinished[-1][1]==time:
                seq_id, _ = unfinished.pop()
                last_vars.append((seq_id, forward_var[seq_id].clone()))

            forward_var = forward_var.unsqueeze(1).expand(batch_size, self.tag_size, self.tag_size)
            forward_var = log_sum_exp(forward_var + self.transition) + emissions[:, time]

        while len(unfinished) > 0:
            seq_id, _ = unfinished.pop()
            last_vars.append((seq_id, forward_var[seq_id].clone()))

        last_vars = torch.stack([score for _, score in sorted(last_vars, key=lambda x: x[0])])
        return log_sum_exp(last_vars + self.transition[self.end_id])

    def _transition_select(self, prev_tags, curr_tags):
        return self.transition.index_select(0, curr_tags).gather(1, prev_tags.unsqueeze(-1)).squeeze(-1)

    def _gold_score(self, emissions, lens, tags):
        '''
        :param emissions: Variable(FloatTensor([seq_len, batch, num_lables]))
        :param tags: Variable(FloatTensor([seq_len, batch]))
        :param lens: sentence lengths
        :return: Variable(FloatTensor([batch]))
        '''
        batch_size, max_len, emission_size = emissions.size()
        assert emission_size == self.tag_size

        # [seq_len, batch]
        emissions = emissions.gather(-1, tags.unsqueeze(-1)).squeeze(-1)

        tags = torch.cat([torch.LongTensor([[self.begin_id]] * batch_size), tags], dim=1)

        unfinished = sorted(enumerate(lens), key=lambda x: x[1], reverse=True)
        last = []
        scores = torch.zeros(batch_size)

        for i in range(max_len):
            # record finished
            while len(unfinished) > 0 and unfinished[-1][1]==i:
                seq_id, _ = unfinished.pop()
                last.append((seq_id, tags[seq_id, i], scores[seq_id].clone()))

            scores = scores + self._transition_select(tags[:, i], tags[:, i+1]) + emissions[:, i]

        while len(unfinished) > 0:
            finished = unfinished.pop()
            last.append((finished[0], tags[finished[0], -1], scores[finished[0]].clone()))

        finished = sorted(last, key=lambda x: x[0])

        last_scores = torch.stack([s for _, _, s in finished])
        last_tags = torch.stack([t for _, t, _ in finished])

        scores = last_scores + self.transition[self.end_id].gather(0, last_tags)
        return scores

    def neg_log_likelihood(self, hiddens, lens, tags):

        emissions = self.hidden2emission(hiddens)
        return self._forward_score(emissions, lens) - self._gold_score(emissions, lens, tags)

    def _viterbi_decode(self, emission):

        sen_len, _ = emission.size()

        backpointers = []
        forward_var = self.transition[:, self.begin_id] + emission[0]
        for time in range(1, emission.size(0), 1):
            max_var, max_id = (forward_var.unsqueeze(0) + self.transition).max(-1)
            backpointers.append(max_id.tolist())
            forward_var = max_var + emission[time]

        best_score, best_id = (forward_var + self.transition[self.end_id]).max(-1)
        best_path = [best_id.item()]
        for bp in reversed(backpointers):
            best_path.append(bp[best_path[-1]])

        best_path.reverse()
        return best_path, best_score

    def forward(self, hiddens, lens):
        '''

        :param hiddens: batch_size * len * hidden_dim
        :param lens: batch_size * len
        :return:
        '''
        emissions = self.hidden2emission(hiddens)
        return [self._viterbi_decode(emissions[sen_id, :, :lens[sen_id]]) for sen_id in range(emissions.size(0))]


class Tagger(nn.Module):
    def __init__(self, words, tags, embed, encoder, crf):
        super(Tagger, self).__init__()
        self.words = words
        self.tags = tags
        self.embed = embed
        self.encoder = encoder
        self.crf = crf

    def forward(self, sens, lens):
        emb = self.embed(sens)
        encode = self.encoder(emb, None)
        return self.crf(encode, lens)

    def _make_masks(self, sens, lens):

        masks = torch.ones(sens.size(), dtype=torch.uint8)
        for i, l in enumerate(lens):
            masks[i, l:] = 0
        return masks

    def loss(self, sens, lens, tags):
        emb = self.embed(sens)
        encode = self.encoder(emb, self._make_masks(sens, lens))
        return self.crf.neg_log_likelihood(encode, lens, tags)


# -------------------------------------------------- load --------------------------------------------
from torchtext import data
from typing import List

class TaggerDataset(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.text), len(ex.tags))

    def __init__(self, path: str, fields: List[data.Field],
                 **kwargs):
        if not isinstance(fields[0], (tuple, list)):
            fields = [('html2text', fields[0]), ('tags', fields[1])]

        examples = []

        with open(path) as file:
            for line in file:
                items = line.strip().split()
                if 0 < len(items) < 150:
                    items = [t.rsplit('#', maxsplit=1) for t in items]
                    tokens = [t[0] for t in items]
                    tags = [t[1][0:2] for t in items]
                    examples.append(data.Example.fromlist([tokens, tags], fields))

        super(TaggerDataset, self).__init__(examples, fields, **kwargs)


def load_dataset(path: str):
    text_field = data.Field(include_lengths=True, batch_first=True)
    tag_field = data.Field(include_lengths=False, batch_first=True)
    train_data, valid_data, test_data = TaggerDataset.splits(
        path=path, train='.train', valid='.valid', test='.test',
        fields=[text_field, tag_field])
    return text_field, tag_field, train_data, valid_data, test_data


class Config:
    partial_prefix = './tagger-data/baike.seg'
    full_prefix = './tagger-data/ctb.std'

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
    vocab_size = 100
    embed_dim = 100
    hidden_dim = 100
    num_head = 20
    hidden_depth = 5

    tag_size = 10
    begin_id = 0
    end_id = 9

    use_cuda = torch.cuda.is_available()

    cached_dataset_prefix = './tagger-data/dataset'
    checkpoint_path = './tagger-data/model/mode_{}_emb_{}_hidden_{}_layer_{}_head_{}'.format(
        hidden_mode, embedding_dim, hidden_dim, hidden_layers, atten_heads)

    valid_step = 500


def make_model():

    words = []
    tags = []
    vocab_size = 100
    embed_dim = 100
    hidden_dim = 100
    num_head = 20
    hidden_depth = 5

    tag_size = 10
    begin_id = 0
    end_id = 9

    embed = Embeddings(vocab_size, embed_dim)

    atten = MultiHeadedAttention(num_head, hidden_dim)
    ffn = PositionwiseFeedForward(hidden_dim, hidden_dim)
    transformer = EncoderLayer(hidden_dim, atten, ffn, dropout=0.1)
    encoder = Encoder(transformer, hidden_depth)

    crf = CRFLayer(hidden_dim, 10, begin_id, end_id)

    tagger = Tagger(words, tags, embed, encoder, crf)

    return tagger

tagger = make_model()

sens = torch.randint(low=0, high=100, size=(2, 10), dtype=torch.int64)
tags = torch.randint(low=0, high=10, size=(2, 10), dtype=torch.int64)

print(tagger(sens, torch.IntTensor([10, 10])))
print(tagger.loss(sens, torch.IntTensor([10, 10]), tags))
