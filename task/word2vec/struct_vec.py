#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

# reference http://www.cs.cmu.edu/~lingwang/papers/naacl2015.pdf
import torch
import torch.nn as nn
import torch.nn.functional as F
from task.util import utils, Vocab
from collections import defaultdict
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import argparse


# reference https://github.com/kefirski/pytorch_NEG_loss/blob/master/NEG_loss/neg.py
class NegativeSamplingLoss(nn.Module):
    def __init__(self, num_class, embedding_dim, weights, size_average=False):
        super(NegativeSamplingLoss, self).__init__()
        self.num_class = num_class
        self.embedding_dim = embedding_dim

        self.output_embedding = nn.Embedding(self.num_class, self.embedding_dim)

        self.weights = torch.FloatTensor(weights)

        self.size_average = size_average

    def forward(self, contexts, outputs, num_sampled):
        """

        :param contexts: FloatTensor(batch, embedding_dim)
        :param outputs: FloatTensor(batch)
        :param num_sampled:
        :return:
        """
        batch_size = outputs.size(0)

        # [batch_size, embedding_dim]
        output_embs = self.output_embedding(outputs)

        # torch.LongTensor(batch_size, num_sampled)
        noises = torch.multinomial(self.weights.view(1, self.num_class).expand(batch_size, -1), num_sampled)

        # torch.FloatTensor(batch_size, num_sampled, embedding_dim)
        noise_embs = self.output_embedding(noises).neg()

        # [batch_size, num_sampled+1, embedding_dim]
        embds = torch.cat([output_embs.unsqueeze(1), noise_embs], 1)
        # [batch_size, num_sampled+1, embedding_dim] * [batch_size, embedding_dim, 1] -> [batch_size, num_sampled+1, 1]
        loss = -F.logsigmoid(torch.bmm(embds, contexts.unsqueeze(2))).sum()

        return loss/batch_size if self.size_average else loss


class NECBOW(nn.Module):
    def __init__(self, vocab, embedding_dim, window_size, weights):
        super(NECBOW, self).__init__()

        self.vocab = vocab
        self.num_class = len(vocab)
        self.embedding_dim = embedding_dim
        self.window_size = window_size

        self.embedding = nn.Embedding(self.num_class, self.embedding_dim)
        self.classifier = nn.Linear(self.embedding_dim*self.window_size*2, self.num_class)

        self.loss = NegativeSamplingLoss(self.num_class, self.embedding_dim*self.window_size*2, weights)

    def forward(self, windows, centers, num_sampled=10):

        windows = torch.cat(self.embedding(windows), -1)
        return self.loss(windows, centers, num_sampled), centers.data.nelement()

    def save(self, path):
        with open(path, 'w') as file:
            file.write('%d\t%d\n' % (self.num_class, self.embedding_dim))
            for index, word in self.vocab.index2words.items():
                file.write('%s\t%s\n' % (word, ' '.join([str(i) for i in self.embedding.weight[index].data])))


class NESkipGram(nn.Module):
    def __init__(self, vocab, embedding_dim, window_size, class_weights):
        super(NESkipGram, self).__init__()
        self.vocab = vocab
        self.num_class = len(vocab)
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.class_weights = class_weights

        self.embedding = nn.Embedding(self.num_class, self.embedding_dim)
        self.losses = nn.ModuleList(
            [NegativeSamplingLoss(self.num_class, self.embedding_dim, self.class_weights) for i in range(self.window_size*2)]
        )

    def forward(self, windows, centers, num_sampled=10):
        """

        :param centers: LongTensor(batch size)
        :return: FloatTensor(batch size, window size, word size)
        """
        centers = self.embedding(centers)

        # [batch size, window size, word size]
        return sum([loss(centers, windows[:, pos], num_sampled) for pos, loss in enumerate(self.losses)]), windows.data.nelement()

    def save(self, path):
        with open(path, 'w') as file:
            file.write('%d\t%d\n' % (self.num_class, self.embedding_dim))
            for index, word in self.vocab.index2words.items():
                file.write('%s\t%s\n' % (word, ' '.join([str(i) for i in self.embedding.weight[index].data])))


class SkipGram(nn.Module):
    def __init__(self, vocab, embedding_dim, window_size, dropout=0.1):
        super(SkipGram, self).__init__()
        self.vocab = vocab
        self.num_class = len(vocab)
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.dropout = dropout

        self.embedding = nn.Embedding(self.num_class, self.embedding_dim)
        self.classifier = nn.Linear(self.embedding_dim, self.num_class*self.window_size*2, bias=False)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, windows, centers):
        """
        :param windows: Variable(batch_size, window_size)
        :param centers: Variable(batch size)
        :return: Variable(1), count
        """

        batch_size = centers.size(0)
        centers = self.embedding(centers)
        losses = F.cross_entropy(self.classifier(centers).view(batch_size*self.window_size*2, self.num_class),
                                 windows.view(batch_size * self.window_size*2),
                                 reduce=False)

        return sum(self.dropout(losses)), windows.data.nelement()

    def save(self, path):
        with open(path, 'w') as file:
            file.write('%d\t%d\n' % (self.num_class, self.embedding_dim))
            for index, word in self.vocab.index2words.items():
                file.write('%s\t%s\n' % (word, ' '.join([str(i) for i in self.embedding.weight[index].data])))


arg_parser = argparse.ArgumentParser

arg_parser.add_argument("--corpus", type=str, default="/Users/sunqf/startup/corpus/en-zh/train.zh", help="corpus path")
arg_parser.add_argument("--min_count", type=int, default=10, help="minimum count of word")
arg_parser.add_argument("--word_freq_decay", type=float, default=0.75, help="decay ratio of word freqency")
arg_parser.add_argument("--batch_size", type=int, default=32, help="batch size")
arg_parser.add_argument("--window_size", type=int, default=4, help="window size of context")


def load(path, window_size, batch_size, word_freq_decay, min_count):
    with open(path) as file:

        data = []
        word_counts = defaultdict(int)
        total_words = 0
        for line in file:
            words = line.strip().split('\n')
            chars = [char if type == '@zh_char@' else type
                     for word in words for type, char in utils.replace_entity(word)]
            if len(chars) > 0:
                data.append(['<s>'] * window_size + chars + ['</s>'] * window_size)

            total_words += len(chars)
            for c in chars:
                word_counts[c] += 1
        word_counts = [(w, pow(c, word_freq_decay)) for w, c in word_counts.items() if c > min_count]
        dict = Vocab([w for w, c in word_counts])
        word_weights = torch.FloatTensor([min_count / total_words] + [c / total_words for w, c in word_counts])

        windows = []
        centers = []
        for sen in data:
            for i in range(window_size, len(sen) - window_size):
                windows.append(dict.convert(
                    [sen[i - offset] for offset in range(window_size)] + [sen[i + offset] for offset in
                                                                          range(window_size)]))
                centers.append(dict.convert(sen[i]))

        train_data = []

        for begin in range(0, len(windows), batch_size):
            train_data.append((torch.LongTensor(windows[begin:begin + batch_size]),
                               torch.LongTensor(centers[begin:begin + batch_size])))

        train_data, valid_data = train_test_split(train_data, test_size=5000 // batch_size)

        import random
        random.shuffle(train_data)

    return dict, train_data, valid_data


if __name__ == '__main__':

    args = arg_parser.parse_args()

    dict, train_data, valid_data = load(args.corpus, args.window_size, args.batch_size, args.word_freq_decay, args.min_count)
    model = SkipGram(dict, 128, 4, 0.2)
    optimizer = torch.optim.Adam(model.parameters())

    num_epoch = 10
    for epoch in tqdm(range(num_epoch), desc='epoch', total=num_epoch):
        total_loss = 0
        total_count = 0
        best_loss = 1e10
        for batch_id, (batch_window, batch_center) in tqdm(enumerate(train_data, start=1), desc='batch', total=len(train_data)):
                model.train()
                model.zero_grad()

                loss, count = model.forward(batch_window, batch_center)

                total_loss += loss.data[0]
                total_count += count

                (loss/count).backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)

                optimizer.step()

                if batch_id % 500 == 0:

                    model.eval()

                    valid_loss = 0
                    valid_count = 0
                    for w, c in valid_data:
                        loss, count = model.forward(w, c)
                        valid_loss += loss.data[0]
                        valid_count += count

                    print('train loss = %f\t\tvalid loss = %f' %
                          (total_loss/(total_count+1e-5), valid_loss/(valid_count+1e-5)))
                    total_count = 0
                    total_loss = 0

                    if best_loss > valid_loss:
                        model.save('char.skipgram-window-4')
                        best_loss = valid_loss