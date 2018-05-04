#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import random

import torch
from torch import Tensor

from torchtext.data import Batch
from torchtext.vocab import Vocab

from .decoder import DecoderState, Decoder
from module.encoder import Encoder
from .reward import *
from module.utils import *


class Beam:
    def __init__(self, decoder_state: DecoderState, sos_index: int, eos_index: int, beam_size: int =10):

        self.decoder_state = decoder_state.repeat(beam_size)

        self.sos_index = sos_index
        self.eos_index = eos_index

        self.prev_index = []

        self.use_cuda = self.decoder_state.use_cuda()

        init_input = torch.LongTensor([sos_index] * beam_size)
        if self.use_cuda:
            init_input = init_input.cuda()

        self.next_input = [init_input]
        self.attn_scores = []

        self.scores = torch.zeros(beam_size)
        if self.use_cuda:
            self.scores = self.scores.cuda()

        # record last state (time_step, prev_index, score)
        self.finished = []

        self.beam_size = beam_size

    def update(self, log_probs: Tensor, output: Tensor, hidden_state: HiddenState, attn_scores):

        scores = log_probs + self.scores.unsqueeze(1).expand_as(log_probs)

        for i in range(self.next_input[-1].size(0)):
            if self.next_input[-1].data[i] == self.eos_index:
                scores[i] = -1e20

        topk_score, topk_index = scores.view(-1).topk(self.beam_size)

        vocab_size = log_probs.size(1)
        topk_prev = topk_index / vocab_size
        topk_input = topk_index % vocab_size

        self.next_input.append(topk_input)
        self.prev_index.append(topk_prev)
        self.scores = topk_score

        self.decoder_state.update(output, hidden_state, topk_prev)

        self.attn_scores.append(attn_scores.index_select(0, topk_prev))

        for i in range(self.next_input[-1].size(0)):
            if self.next_input[-1].data[i] == self.eos_index:
                self.finished.append((len(self.next_input), i, self.scores[i]))

    def is_finish(self) -> bool:
        return len(self.finished) >= self.beam_size

    def get_hypo(self, time_step: int, index: int) -> Tuple[List[int], Tensor]:

        hypo, attn = [], []
        for t in range(time_step-1, -1, -1):
            hypo.append(self.next_input[t].data[index])
            attn.append(self.attn_scores[t - 1][index])
            index = self.prev_index[t-1].data[index]

        return hypo[::-1], torch.stack(attn[::-1], 0)

    def get_best(self) -> Tuple[List[int], Tensor]:

        if len(self.finished) > 0:
            best_len, best_index, score = max(self.finished, key=lambda x: x[2].data/x[0])
        else:
            best_len = len(self.next_input)-1
            best_index = max(range(len(self.next_input[-1])), key=lambda i: self.scores.data[i])

        return self.get_hypo(best_len, best_index)

    def get_topk(self, topk) -> Tuple[List[int], Tensor, Tensor]:
        if len(self.finished) > 0:
            hypo_topk = sorted(self.finished, key=lambda x: x[2].data[0]/x[0], reverse=True)[0:topk]
            hypo_topk = [(*self.get_hypo(hypo_len, hypo_index), hypo_score) for hypo_len, hypo_index, hypo_score in hypo_topk]
        else:
            best_len = len(self.next_input)-1
            hypo_topk = [(*self.get_hypo(best_len, hypo_index), hypo_score)
                    for hypo_index, hypo_score in zip(range(min(len(self.next_input[-1]), topk)), self.scores[0:topk])]

        return hypo_topk

    @staticmethod
    def cat(beams):
        states = DecoderState.cat([b.decoder_state for b in beams])
        inputs = torch.cat([b.next_input[-1] for b in beams])

        return inputs, states


class Seq2Seq(nn.Module):
    def __init__(self, config,
                 src_voc: Vocab, trg_voc: Vocab,
                 src_embed, trg_embed, output_embed,
                 encoder, decoder):

        super(Seq2Seq, self).__init__()

        self.config = config
        self.src_voc = src_voc
        self.trg_voc = trg_voc

        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.output_embed = output_embed

        self.encoder = encoder
        self.decoder = decoder

        self.rewarder = GLEU()

        self.pad_index = trg_voc.stoi[PAD]
        self.sos_index = trg_voc.stoi[SOS]
        self.eos_index = trg_voc.stoi[EOS]

    def cuda(self, device=None):

        self.encoder.cuda(device)
        self.decoder.cuda(device)

        return self

    def _mask(self, batch_size: int, max_length: int, lengths: List[int]) -> Tensor:
        mask = torch.ByteTensor(batch_size, max_length).fill_(0)
        for i, l in enumerate(lengths):
            mask[i, 0:l] = 1

        return Tensor(mask)

    def forward(self, src: Tensor, src_lengths: List[int],
                trg: Tensor,
                src_embedding, trg_embedding, output_embedding,
                teacher_force_ratio: float = 0.5) -> Tuple[Tensor, Tensor]:
        """
        :param src: LongTensor[src_len, batch]
        :param src_lengths: [int] * batch
        :param trg: LongTensor[trg_len, batch]
        :param teacher_force_ratio: float
        :return:
            outputs: FloatTensor[trg_len, batch, vocab_size]
            attn_scores: FloatTensor[trg_len, batch, src_len]
        """

        src_len, batch_size = src.size()

        src_embed = src_embedding(src)

        # src_hidden [src_len, batch, dim]
        src_hidden, src_hidden_state = self.encoder(src_embed)

        hidden_state = src_hidden_state

        # [batch, dim]
        _, batch_size, dim = src_hidden.size()
        context = Tensor(src_hidden.data.new(batch_size, dim).zero_())

        # [batch, src_len, dim]
        src_hidden = src_hidden.transpose(0, 1)
        src_mask = self._mask(batch_size, src_len, src_lengths)
        if src_hidden.is_cuda:
            src_mask = src_mask.cuda(async=True)

        decoderState = DecoderState(src_hidden, src_mask, context, hidden_state)
        max_len = trg.size(0)

        log_probs = []
        attn_scores = []
        for t in range(max_len):
            teacher_force = (t == 0 or (self.training and random.random() < teacher_force_ratio))

            input = trg[t] if teacher_force else log_prob.max(-1)[1]

            input_embed = trg_embedding(input)

            context, hidden_state, score = self.decoder(input_embed, decoderState)
            decoderState.update(context, hidden_state)

            log_prob = F.log_softmax(output_embedding(context), -1)

            log_probs.append(log_prob)
            attn_scores.append(score)

        return torch.stack(log_probs), torch.stack(attn_scores)

    def ml_loss(self, batch: Batch, pool_state):

        batch, src_embed, trg_embed, output_embed = pool_state.forward(batch)

        src, src_lens = batch.src
        trg, trg_lens = batch.trg
        log_probs, attn_scores = self.forward(src, src_lens, trg, src_embed, trg_embed, output_embed)

        loss = F.nll_loss(log_probs[:-1].view(-1, log_probs.size(-1)),
                          trg[1:].view(-1),
                          ignore_index=pool_state.pad_index,
                          size_average=False)

        return loss, sum(trg_lens)


    def translate(self, src: Tensor, src_lengths: List[int],
                  src_embedding, trg_embedding, output_embedding) -> List[Beam]:
        """

        :param src: LongTensor[src_len, batch]
        :param src_lengths: List[int]
        :return:
        """

        src_len, batch_size = src.size()

        # src_output [src_len, batch, dim]
        src_output, src_hidden_state = self.encoder(src_embedding(src))

        hidden_state = src_hidden_state

        # [batch, dim]
        _, batch_size, dim = src_output.size()
        context = Tensor(src_output.data.new(batch_size, dim).zero_())

        # [batch, src_len, dim]
        src_output = src_output.transpose(0, 1)
        src_mask = self._mask(batch_size, src_len, src_lengths)
        if src_output.is_cuda:
            src_mask = src_mask.cuda(async=True)

        states = DecoderState.init_split(src_output, src_mask, context, hidden_state, 1)
        beams = [Beam(states[b], self.sos_index, self.eos_index) for b in range(batch_size)]

        max_len = src_len * 2

        for t in range(max_len):
            unfinished = [beam for beam in beams if not beam.is_finish()]
            if len(unfinished) == 0:
                break

            inputs, states = Beam.cat(unfinished)

            context, hidden_state, attn_score = self.decoder(trg_embedding(inputs), states)
            log_prob = F.log_softmax(output_embedding(context), -1)
            log_probs, outputs, hidden_states, attns = DecoderState.chunk(log_prob, context, hidden_state, attn_score, len(unfinished))
            for i in range(len(unfinished)):
                unfinished[i].update(log_probs[i], outputs[i], hidden_states[i], attns[i])

        return beams

    def rl_loss(self, batch: Batch, pool_state):

        batch, src_embed, trg_embed, output_embed = pool_state.forward(batch)

        src, src_lens = batch.src
        trg, trg_lens = batch.trg

        beams = self.translate(src, src_lens, src_embed, trg_embed, output_embed)

        losses = []
        for i in range(len(src_lens)):
            topk = beams[i].get_topk(10)

            rewards = self.rewarder.score(trg[:trg_lens[i], i].data.tolist(), [hypo for hypo, _, _ in topk])
            losses.append(sum([score * r for (_, _, score), r in zip(topk, rewards)])/len(topk))

        return sum(losses), sum(trg_lens)

    def sample(self, batch: Batch, pool_state) -> None:

        batch, src_embed, trg_embed, output_embed = pool_state.forward(batch)

        src, src_lens = batch.src
        trg, trg_lens = batch.trg

        beams = self.translate(src, src_lens, src_embed, trg_embed, output_embed)

        bests = [b.get_best() for b in beams]

        for i in range(src_lens.size(0)):
            print('src: %s' % (' '.join([pool_state.src_voc.itos[tok] for tok in src[:src_lens[i], i].cpu().data])))
            print('trg: %s' % (' '.join([pool_state.trg_voc.itos[tok] for tok in bests[i][0]])))


    @classmethod
    def create(cls, src_voc, trg_voc, config):

        config = config
        src_voc = src_voc
        trg_voc = trg_voc

        src_embed = nn.Embedding(len(src_voc), config.embedding_dim, padding_idx=src_voc.stoi[PAD])
        trg_embed = nn.Embedding(len(trg_voc), config.embedding_dim, padding_idx=trg_voc.stoi[PAD])
        output_embed = nn.Linear(config.hidden_dim, len(trg_voc), bias=False)

        encoder = Encoder(config.embedding_dim,
                          config.hidden_model, config.hidden_dim, config.num_layers, dropout=config.dropout)
        decoder = Decoder(config.embedding_dim,
                          config.hidden_model, config.hidden_dim, config.num_layers, dropout=config.dropout)

        return cls(config, src_voc, trg_voc, src_embed, trg_embed, output_embed, encoder, decoder)