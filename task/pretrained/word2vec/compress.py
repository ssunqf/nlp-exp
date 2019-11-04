#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import torch
import time
import math
import os
import sys
import json
import random
import scipy.stats as stats
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from task.pretrained.baike.base import smart_open


def txt2npy(src):
    with smart_open(src, 'rt') as input, smart_open(src + '.word', 'wt') as word_file:
        voc_size, dim = map(int, input.readline().split())
        voc_size = min(voc_size, 2000000)
        matrix = np.zeros((voc_size, dim), dtype=np.float32)
        count = 0
        for line in tqdm(input, desc='load word embedding'):
            if count >= voc_size:
                break
            word, *values = line.rstrip().rsplit(maxsplit=dim)
            if len(values) == dim:
                word_file.write(word + '\n')
                vector = np.array([float(t) for t in values], dtype=np.float32)
                matrix[count] = vector

                count += 1


        np.save(src + '.npy', matrix[:count])


def load_vector(src):

    with smart_open(src + '.word') as input:
        id2word = [line.rstrip() for line in input.readlines()]

    matrix = np.load(src + '.npy', mmap_mode='r')

    assert len(id2word) == matrix.shape[0]

    word2id = {word: id for id, word in enumerate(id2word)}

    return id2word, word2id, matrix


def similarity(word2id, vectors):
    prefix = '../emb2bin/testsets/similarity/'
    total_score, total_count = 0, 0
    for testset in ['COS960_all.txt',
                    'COS960_adj.txt',
                    'COS960_noun.txt',
                    'COS960_verb.txt',
                    'pku-500.csv',
                    'sim-240.csv',
                    'sim-297.csv']:

        task_name = testset.rsplit('.', maxsplit=1)[0]
        with open(os.path.join(prefix, testset)) as file:
            testPairNum = 0
            skipPairNum = 0

            wordSimStd = []
            wordSimPre = []

            for line in file:
                if line.startswith('#'):
                    continue

                word1, word2, valStr = line.strip().split()[0:3]
                if (word1 in word2id) and (word2 in word2id):
                    testPairNum += 1
                    wordSimStd.append(float(valStr))
                    vector1 = vectors[word2id[word1]]
                    vector2 = vectors[word2id[word2]]
                    if vectors.dtype == torch.uint8:
                        # Sokal & Michener similarity function  sim(x, y) = (n11 + n00) / n
                        cosSim = (vector1 == vector2).float().sum() / vector1.size(0)
                    else:
                        cosSim = torch.dot(vector1, vector2) / vector1.norm() / vector2.norm()

                    wordSimPre.append(cosSim.item())
                else:
                    skipPairNum += 1
                    print('Skip:', word1, word2)

            corrCoef = np.corrcoef(wordSimStd, wordSimPre)[0, 1]
            SpearCoef = stats.spearmanr(wordSimStd, wordSimPre).correlation
            SqrtCoef = np.sqrt(corrCoef * SpearCoef)

            print("%s Pearson Score: %f" % (task_name, corrCoef))
            print("%s Spearman Score: %f" % (task_name, SpearCoef))
            print("%s Sqrt of r and rho Score: %f" % (task_name, SqrtCoef))
            print('TestPair:', testPairNum, 'SkipPair:', skipPairNum)
            print()

        total_score += SqrtCoef
        total_count += 1
    return total_score / total_count

def read_analogy(iw):
    analogy = {}
    analogy_type = ""
    for file in ['../emb2bin/testsets/analogy/CA8/morphological.txt',
                 '../emb2bin/testsets/analogy/CA8/semantic.txt']:
        with open(file) as f:
            for line in f:
                oov = 0
                if line.strip().split()[0] == ':':
                    analogy_type = line.strip().split()[1]
                    analogy[analogy_type] = {}
                    analogy[analogy_type]["questions"] = []
                    analogy[analogy_type]["total"] = 0
                    analogy[analogy_type]["seen"] = 0
                    continue
                analogy_question = line.strip().split()
                for w in analogy_question[:3]:
                    if w not in iw:
                        oov = 1
                if oov == 1:
                    analogy[analogy_type]["total"] += 1
                    continue
                analogy[analogy_type]["total"] += 1
                analogy[analogy_type]["seen"] += 1
                analogy[analogy_type]["questions"].append(analogy_question)

            for t in analogy:
                analogy[t]['iw'] = []
                analogy[t]['wi'] = {}
                for question in analogy[t]["questions"]:
                    for w in question:
                        if w not in analogy[t]['iw']:
                            analogy[t]['iw'].append(w)
                for i, w in enumerate(analogy[t]['iw']):
                    analogy[t]['wi'][w] = i
    return analogy


def guess(sims: torch.Tensor, analogy, analogy_type, iw, wi, word_a, word_b, word_c):
    sim_a = sims[analogy[analogy_type]["wi"][word_a]]
    sim_b = sims[analogy[analogy_type]["wi"][word_b]]
    sim_c = sims[analogy[analogy_type]["wi"][word_c]]

    add_sim = -sim_a+sim_b+sim_c
    add_sim[wi[word_a]] = 0
    add_sim[wi[word_b]] = 0
    add_sim[wi[word_c]] = 0
    guess_add = iw[torch.argmax(add_sim)]

    mul_sim = sim_b * sim_c * torch.reciprocal(sim_a + 1e-5)
    mul_sim[wi[word_a]] = 0
    mul_sim[wi[word_b]] = 0
    mul_sim[wi[word_c]] = 0
    guess_mul = iw[torch.argmax(mul_sim)]

    return guess_add, guess_mul


def analogy(analogy, word2id, id2word, matrix: torch.Tensor):

    if matrix.dtype not in [torch.uint8]:
        matrix = F.normalize(matrix, dim=-1)

    results = {}
    total_add, total_mul, total_seen = 0, 0, 0
    for analogy_type in analogy.keys():
        correct_add_num, correct_mul_num = 0, 0
        analogy_matrix = matrix.index_select(0,
            torch.tensor([word2id[w] if w in word2id else random.randint(0, len(word2id) - 1)
                             for w in analogy[analogy_type]["iw"]], dtype=torch.long))
        if matrix.dtype == torch.uint8:
            sum11 = torch.matmul(analogy_matrix.float(), matrix.t().float())
            sum00 = torch.matmul(1 - analogy_matrix.float(), 1 - matrix.t().float())
            sims = (sum11 + sum00)
        else:
            sims = torch.matmul(analogy_matrix, matrix.t())
            sims = (sims + 1) / 2  # Transform similarity scores to positive numbers (for mul evaluation)

        for question in analogy[analogy_type]["questions"]:  # Loop for each analogy question
            word_a, word_b, word_c, word_d = question
            guess_add, guess_mul = guess(sims, analogy, analogy_type, id2word, word2id, word_a, word_b, word_c)

            if guess_add == word_d:
                correct_add_num += 1
            if guess_mul == word_d:
                correct_mul_num += 1
        cov = float(analogy[analogy_type]["seen"]) / analogy[analogy_type]["total"]
        if analogy[analogy_type]["seen"] == 0:
            acc_add = 0
            acc_mul = 0
            print(analogy_type + " add/mul: " + str(round(0.0, 3)) + "/" + str(round(0.0, 3)))
        else:
            acc_add = float(correct_add_num) / analogy[analogy_type]["seen"]
            acc_mul = float(correct_mul_num) / analogy[analogy_type]["seen"]
            print(analogy_type + " add/mul: " + str(round(acc_add, 3)) + "/" + str(round(acc_mul, 3)))
        # Store the results
        results[analogy_type] = {}
        results[analogy_type]["coverage"] = [cov, analogy[analogy_type]["seen"], analogy[analogy_type]["total"]]
        results[analogy_type]["accuracy_add"] = [acc_add, correct_add_num, analogy[analogy_type]["seen"]]
        results[analogy_type]["accuracy_mul"] = [acc_mul, correct_mul_num, analogy[analogy_type]["seen"]]

        total_add += correct_add_num
        total_mul += correct_mul_num
        total_seen += analogy[analogy_type]["seen"]

    return math.sqrt((total_add / total_seen) * (total_mul / total_seen))


class BinaryCompress(nn.Module):
    def __init__(self, emb_dim, binary_dim, alpha=1):
        super(BinaryCompress, self).__init__()

        self.emb_dim = emb_dim
        self.binary_dim = binary_dim

        self.weight = nn.Parameter(torch.Tensor(emb_dim, binary_dim))

        self.bias = nn.Parameter(torch.Tensor(emb_dim))

        self.alpha = alpha

        self.momentum = 0.95

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def to_json(self):
        return {
            'emb_dim': self.emb_dim,
            'binary_dim': self.binary_dim,
            'alpha': self.alpha,
        }

    def encode(self, input):
        return torch.matmul(input, self.weight) > 0.

    def decode(self, code):
        return F.hardtanh(F.linear(code.float(), self.weight, self.bias))

    def rec_loss(self, input):
        return F.mse_loss(self.decode(self.encode(input)), input, reduction='sum') / (input.size(0) if input.dim() == 2 else 1)

    def reg_loss(self):
        return ((torch.matmul(self.weight.t(), self.weight)
                              - torch.diag(torch.ones(self.binary_dim))) ** 2).sum()


class Trainer:
    def __init__(self, model):
        self.model = model

    def train(self, config):
        id2word, word2id, matrix = load_vector(config.emb_path)

        matrix = torch.from_numpy(matrix.clip(-1, 1))

        assert matrix.dim() == 2

        train_size = min(config.train_size, len(id2word))
        train_matrix = matrix[:train_size]

        eval_size = min(config.eval_size, len(id2word))
        eval_matrix = matrix[:eval_size]
        eval_id2word = id2word[:eval_size]
        eval_word2id = {word:id for id, word in enumerate(eval_id2word)}
        self.eval(0, eval_matrix, eval_id2word, eval_word2id)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        for epoch in tqdm(range(1, config.max_epoch+1), desc='epoch'):

            train_rec_losses, train_reg_losses, train_losses = [], [], []

            shuffled_matrix = train_matrix.index_select(0, torch.randperm(train_size))

            start_time = time.time()
            for offset in tqdm(range(0, train_size, config.batch_size), desc='iter'):
                self.model.train()
                self.model.zero_grad()

                batch = shuffled_matrix[offset:offset+config.batch_size]

                rec_loss = self.model.rec_loss(batch)
                reg_loss = self.model.reg_loss()

                loss = rec_loss + self.model.alpha * reg_loss

                train_rec_losses.append(rec_loss.item())
                train_reg_losses.append(reg_loss.item())
                train_losses.append(loss.item())

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                optimizer.step()

            elapse_time = time.time() - start_time

            bps = len(train_losses) / elapse_time

            train_rec_loss, train_reg_loss, train_loss = np.mean(train_rec_losses), np.mean(train_reg_losses), np.mean(train_losses)

            self.model.eval()

            print('epoch %d: train (rec=%f, reg=%f, total=%f), bps = %f' %
                  (epoch,
                   train_rec_loss, train_reg_loss, train_loss,
                   bps))

            with torch.no_grad():
                bin_matrix = self.model.encode(eval_matrix)
                self.eval(epoch, bin_matrix, eval_id2word, eval_word2id)
                rec_matrix = self.model.decode(bin_matrix)
                self.eval(epoch, rec_matrix, eval_id2word, eval_word2id)

        self.export(id2word, matrix, config.output_path)

    def distance(self, matrix: torch.Tensor, batch_size=64):
        distances = []
        for offset in range(0, matrix.size(0), batch_size):
            original_vec = matrix[offset:offset+batch_size]
            resconstruct_vec = self.model.decode(self.model.encode(original_vec))

            distances.extend((resconstruct_vec - original_vec).norm(dim=-1).tolist())

        print('distance = %f' % np.mean(distances))

    def eval(self, epoch, matrix: torch.Tensor, id2word, word2id):
        if matrix.dtype == torch.float32:
            matrix = F.normalize(matrix, dim=-1) # matrix.index_select(0, torch.randperm(train_size))
        analogy_data = read_analogy(id2word)
        simscore = similarity(word2id, matrix)
        anascore = analogy(analogy_data, word2id, id2word, matrix)

        print('epoch %d sim = %f, ana = %f avg = %f' % (epoch, simscore, anascore, (simscore + anascore) / 2))

    def export(self, id2word, matrix, output):
        with open(output + '.bin', 'w') as bin_file, open(output + '.rec', 'w') as rec_file:
            bin_file.write('%d %d\n' % (len(id2word), self.model.binary_dim))
            rec_file.write('%d %d\n' % (len(id2word), self.model.emb_dim))

            for id, word in enumerate(id2word):
                bin_vec = self.model.encode(matrix[id])
                bin_file.write('%s %s\n' % (word, ' '.join(map(str, bin_vec.tolist()))))

                rec_vec = self.model.decode(bin_vec)
                rec_file.write('%s %s\n' % (word, ' '.join(['%.6f' % i for i in rec_vec.tolist()])))

    def save(self, model, path):
        torch.save(self.model.state_dict(), path + '.pth')


class Config:
    def __init__(self, config: dict):
        self.emb_path = config['emb_path']
        self.train_size = config['train_size']
        self.eval_size = config['eval_size']
        self.emb_dim = config['emb_dim']
        self.bin_dim = config['bin_dim']

        self.batch_size = config.get('batch_size', 64)
        self.max_epoch = config.get('max_epoch', 10)

        self.output_path = config['output_path']

    @classmethod
    def from_json(cls, file):
        with open(file) as input:
            return cls(json.load(input))

    def to_json(self):
        return json.dump(self.__dict__)


if __name__ == "__main__":

    config = Config.from_json(sys.argv[1])

    if not os.path.exists(config.emb_path + '.word') or not os.path.exists(config.emb_path + '.npy'):
        txt2npy(config.emb_path)

    compress = BinaryCompress(config.emb_dim, config.bin_dim)

    trainer = Trainer(compress)
    trainer.train(config)


