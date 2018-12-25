#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import math
import time
from typing import List
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

from torchtext.vocab import Vocab
from torchtext import data
from task.pretrained.transformer.base import BOS, EOS, PAD

from task.util import utils


def get_dropout_mask(prob: float, tensor_to_mask: torch.Tensor):
    mask = torch.rand(tensor_to_mask.size(), device=tensor_to_mask.device) > prob
    return mask.float().div(1 - prob)

class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, go_forward: bool, dropout=0.2):
        super(LSTMLayer, self).__init__()
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.go_forward = go_forward
        self.recurrent_dropout_prob = dropout

    def forward(self, input: PackedSequence):

        input, batch_sizes = input

        seq_len = batch_sizes.size()[0]
        max_batch_size = batch_sizes[0]

        output = input.new_zeros(input.size(0), self.hidden_size)

        hidden_state = input.new_zeros(max_batch_size, self.hidden_size)
        cell_state = input.new_zeros(max_batch_size, self.hidden_size)

        recurrent_mask = get_dropout_mask(self.recurrent_dropout_prob, hidden_state) if self.training else None

        cumsum_sizes = torch.cumsum(batch_sizes, dim=0)
        for timestep in range(seq_len):
            timestep = timestep if self.go_forward else seq_len - timestep - 1
            len_t = batch_sizes[timestep]
            begin, end = (cumsum_sizes[timestep]-len_t, cumsum_sizes[timestep])

            input_t = input[begin:end]
            hidden_t, cell_t = self.cell(input_t, (hidden_state[0:len_t], cell_state[0:len_t]))

            if self.training:
                hidden_t = hidden_t * recurrent_mask[:len_t]

            output[begin:end] = hidden_t
            hidden_state = hidden_state.clone()
            cell_state = cell_state.clone()
            hidden_state[0:batch_sizes[timestep]] = hidden_t
            cell_state[0:batch_sizes[timestep]] = cell_t

        return PackedSequence(output, batch_sizes), (hidden_state, cell_state)


class ElmoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(ElmoLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        forwards, backwards = [], []
        for l in range(num_layers):
            forwards.append(LSTMLayer(input_size, hidden_size, True, dropout))
            backwards.append(LSTMLayer(input_size, hidden_size, False, dropout))
            input_size = hidden_size

        self.forwards = nn.ModuleList(forwards)
        self.backwards = nn.ModuleList(backwards)

    def forward(self, input: PackedSequence):

        forward_outputs = []
        backward_outputs = []

        forward_input, backward_input = input, input
        for l in range(self.num_layers):
            forward_output, _ = self.forwards[l](forward_input)
            backward_output, _ = self.backwards[l](backward_input)

            if l != 0:
                forward_output = PackedSequence(forward_output.data + forward_input.data, forward_output.batch_sizes)
                backward_output = PackedSequence(backward_output.data + backward_input.data, backward_output.batch_sizes)

            forward_outputs.append(forward_output)
            backward_outputs.append(backward_output)

            forward_input, backward_input = forward_output, backward_output

        forward_outputs = torch.stack([o.data for o in forward_outputs], dim=-2)
        backward_outputs = torch.stack([o.data for o in backward_outputs], dim=-2)

        return PackedSequence(forward_outputs, input.batch_sizes), \
               PackedSequence(backward_outputs, input.batch_sizes)


class CharElmo(nn.Module):
    def __init__(self, token_vocab: Vocab, embed_size, hidden_size, num_layers, dropout=0.2):
        super(CharElmo, self).__init__()
        self.vocab = token_vocab
        self.padding_idx = self.vocab.stoi[PAD]
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(len(token_vocab), embed_size, padding_idx=self.padding_idx)
        self.elmo = ElmoLSTM(embed_size, hidden_size, num_layers, dropout=dropout)

        self.forward_linear = nn.Linear(hidden_size, embed_size)
        self.backward_linear = nn.Linear(hidden_size, embed_size)

        self.output_embed = nn.Linear(embed_size, len(token_vocab))
        self.output_embed.weight = self.embed.weight

    def forward(self, input, lens):

        input = pack_padded_sequence(input, lens)

        embed = PackedSequence(self.embed(input.data), input.batch_sizes)

        hiddens = self.elmo(embed)

        return hiddens

    def loss(self, input, lens):

        input = pack_padded_sequence(input, lens)

        embed = PackedSequence(self.embed(input.data), input.batch_sizes)

        forwards, backwards = self.elmo(embed)
        '''
        last_forwards, _ = pad_packed_sequence(
            PackedSequence(self.output_embed(self.forward_linear(forwards.data[:, -1])), forwards.batch_sizes))
        last_backwards, _ = pad_packed_sequence(
            PackedSequence(self.output_embed(self.backward_linear(backwards.data[:, -1])), backwards.batch_sizes))

        forward_num_tokens, backward_num_tokens = input.batch_sizes[1:].sum(), input.batch_sizes[:-1].sum()

        pad_input, _ = pad_packed_sequence(input, padding_value=self.padding_idx)

        forward_loss = F.cross_entropy(last_forwards[0:-1].view(-1, len(self.vocab)),
                                       pad_input[1:].view(-1),
                                       ignore_index=self.padding_idx, reduction='sum')
        backward_loss = F.cross_entropy(last_backwards[1:].view(-1, len(self.vocab)),
                                        pad_input[:-1].view(-1),
                                        ignore_index=self.padding_idx, reduction='sum')
        '''
        batch_sizes = input.batch_sizes
        last_forwards, last_backwards = forwards.data[:, -1], backwards.data[:, -1]

        cumsum_sizes = torch.cumsum(batch_sizes, dim=0)

        max_seq_len = lens[0]
        forward_context, forward_target = [], []
        for t in range(max_seq_len - 1):
            cbegin = cumsum_sizes[t] - batch_sizes[t]
            cend = cbegin + batch_sizes[t+1]
            tbegin = cumsum_sizes[t]
            tend = tbegin + batch_sizes[t+1]
            forward_context.append(last_forwards[cbegin:cend])
            forward_target.append(input.data[tbegin:tend])

        forward_context = torch.cat(forward_context)
        forward_context = self.output_embed(self.forward_linear(forward_context))
        forward_target = torch.cat(forward_target)

        forward_loss = F.cross_entropy(forward_context, forward_target, reduction='sum')

        backward_context, backward_target = [], []
        for t in range(max_seq_len - 1):
            cbegin = cumsum_sizes[t]
            cend = cbegin + batch_sizes[t + 1]
            tbegin = cumsum_sizes[t] - batch_sizes[t]
            tend = tbegin + batch_sizes[t + 1]
            backward_context.append(last_backwards[cbegin:cend])
            backward_target.append(input.data[tbegin:tend])

        backward_context = torch.cat(backward_context)
        backward_context = self.output_embed(self.backward_linear(backward_context))
        backward_target = torch.cat(backward_target)

        backward_loss = F.cross_entropy(backward_context, backward_target, reduction='sum')

        return forward_loss, backward_loss, cumsum_sizes[-1].item() - len(lens)


class ElmoDataset(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path: str, fields: List, **kwargs):

        examples = []

        with open(path) as file:
            for line in file:
                line = line.strip()
                if 0 < len(line) < 150:
                    examples.append(data.Example.fromlist([line], fields))

        super(ElmoDataset, self).__init__(examples, fields, **kwargs)


class Trainer:
    def __init__(self, model: CharElmo, train_it, valid_it, valid_step=100):

        self.model = model
        self.optimizer = optim.Adam(self.model.parameters())

        self.train_it = train_it
        self.valid_it = valid_it
        self.valid_step = valid_step

    def train_one(self, data):
        self.model.train()
        self.model.zero_grad()

        ids, lens = data.text
        f_loss, b_loss, num_tokens = self.model.loss(ids, lens)
        (f_loss.div(num_tokens) + b_loss.div(num_tokens)).backward()

        # Step 3. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()
        return f_loss.item(), b_loss.item(), num_tokens

    def valid(self):
        self.model.eval()
        with torch.no_grad():
            floss, bloss, num_tokens = 0., 0., 0.
            for batch in self.valid_it:
                ids, lens = batch.text
                f_loss, b_loss, _num = self.model.loss(ids, lens)
                floss += f_loss.item()
                bloss += b_loss.item()
                num_tokens += _num

        return floss, bloss, num_tokens

    def train(self):
        train_f_loss, train_b_loss, train_num, start_time = 0., 0., 0., time.time()
        for step, batch in tqdm(enumerate(self.train_it, start=1)):

            f_loss, b_loss, _num = self.train_one(batch)

            train_f_loss += f_loss
            train_b_loss += b_loss
            train_num += _num

            if step % self.valid_step == 0:
                # (valid_f_loss, valid_f_num), (valid_b_loss, valid_b_num) = self.valid()
                train_f_loss /= train_num
                train_b_loss /= train_num

                # valid_f_loss /= valid_f_num
                # valid_b_loss /= valid_b_num
                print('train: forward loss = %.5f\tperplexity=%.5f\tbackward loss = %0.5f\tperplexity=%.5f' % (
                    train_f_loss, math.exp(train_f_loss), train_b_loss, math.exp(train_b_loss)))
                # print('valid: forward loss = %.5f\tperplexity=%.5f\tbackward loss = %0.5f\tperplexity=%.5f' % (
                #     valid_f_loss, math.exp(valid_f_loss), valid_b_loss, math.exp(valid_b_loss)))
                print('speed %.2f w/s' % (train_num / (time.time()-start_time)))

                train_f_loss, train_b_loss, train_num, start_time = 0., 0., 0., time.time()

    @staticmethod
    def create(config):
        def tokenize(text):
            return [w for t, w in utils.replace_entity(text)]

        token_field = data.Field(include_lengths=True, init_token=BOS, eos_token=EOS, tokenize=tokenize)
        train_data, valid_data = ElmoDataset.splits(
            config.data_prefix,
            train=config.train, validation=config.valid,
            fields=[('text', token_field)]
        )
        train_data.sort_key = lambda example: len(example.text)
        valid_data.sort_key = lambda example: len(example.text)
        token_field.build_vocab(train_data, valid_data, min_freq=config.token_min_freq)

        train_it, valid_it = data.BucketIterator.splits([train_data, valid_data],
                                                        batch_sizes=config.batch_sizes,
                                                        sort_within_batch=True,
                                                        device=torch.device('gpu') if config.use_cuda else torch.device('cpu'))
        train_it.repeat = True
        model = CharElmo(token_field.vocab,
                         config.embed_size,
                         config.hidden_size,
                         config.num_layers,
                         config.dropout)
        if config.use_cuda:
            model = model.cuda()
        return Trainer(model, train_it, valid_it, config.valid_step)


class Config:
    data_prefix = 'elmo/data'

    train, valid = 'train', 'valid'

    token_min_freq = 10
    batch_sizes = [16, 16]

    embed_size = 256
    hidden_size = 256
    num_layers = 5
    dropout = 0.2

    valid_step = 200

    use_cuda = torch.cuda.is_available()


if __name__ == '__main__':

    config = Config()
    trainer = Trainer.create(config)

    trainer.train()


