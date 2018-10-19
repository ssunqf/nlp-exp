#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
import torch.nn.functional as F
from .config import config
from tensorboardX import SummaryWriter
import gzip
from tqdm import tqdm

class Classifier(nn.Module):
    def __init__(self, feature_dim: int, num_label: int, dropout=0.3):
        super(Classifier, self).__init__()

        self.num_label = num_label

        self.model = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim//2),
            nn.BatchNorm1d(feature_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim//2, feature_dim//2),
            nn.BatchNorm1d(feature_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim//2, feature_dim//8),
            nn.BatchNorm1d(feature_dim//8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 8, num_label),
            nn.Sigmoid()
        )

    def forward(self, features):
        return self.model(features)

    def loss(self, features, label):

        pred = self.forward(features)
        return F.binary_cross_entropy(pred, label, size_average=False)


def load(path, label2id: dict, batch_size=16):
    with gzip.open(path, mode='rt', compresslevel=6) as f:
        data = []
        counts = [0] * len(label2id)
        total = 0
        for line in f:
            labels, features = line.split('\t')
            labels = labels.split()
            features = torch.Tensor([float(i) for i in features.strip().split()])
            labelv = torch.zeros(len(label2id), dtype=torch.float32)
            for l in labels:
                if l in label2id:
                    labelv[label2id[l]] = 1
                    counts[label2id[l]] += 1
            if labelv.sum().item() > 0.1:
                data.append((features, labelv))

            total += 1

            if len(data) >= batch_size:
                batch_feature = torch.stack([f for f, l in data])
                batch_label = torch.stack([l for f, l in data])
                yield batch_feature, batch_label
                data = []

        if len(data) > 0:
            batch_feature = torch.stack([f for f, l in data])
            batch_label = torch.stack([l for f, l in data])
            yield batch_feature, batch_label

        print('\n'.join([label + ': ' + str(counts[id]) for label, id in label2id.items()]))


def train(model: Classifier, train_data, valid_data, test_data, label2id: dict, max_epoch=10):
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5, amsgrad=True)

    writer = SummaryWriter(log_dir=config.summary_dir+'2')

    best_mean_f1 = 0
    for epoch in range(max_epoch):
        train_loss, train_count = 0., 0.
        for id, batch in tqdm(enumerate(train_data)):
            feature, label = batch
            model.train()
            model.zero_grad()

            loss = model.loss(feature, label)

            train_loss += loss
            train_count += label.size()[0]
            loss.backward()
            optimizer.step()

            writer.add_scalars('Loss', {'train': train_loss/(train_count+1e-5)}, epoch*len(train_data)+id)

            if id % 1000 == 0:
                valid_loss, valid_count = 0., 0.
                model.eval()
                for vfeature, vlabel in valid_data:
                    valid_loss += model.loss(vfeature, vlabel)
                    valid_count += vlabel.size()[0]
                writer.add_scalars('loss', {'valid': valid_loss/(valid_count+1e-5)}, epoch*len(train_data)+id)

                print('epoch %d\titeration %d\tTrain loss=%0.5f\t\tValid loss=%0.5f' % (
                    epoch, id, train_loss/(train_count+1e-5), valid_loss/(valid_count+1e-5)))

                train_loss = 0.
                train_count = 0.
            if id % 10000 == 0:
                # test
                model.eval()
                TP = torch.zeros(model.num_label, dtype=torch.int64)
                TP_FP = torch.zeros(model.num_label, dtype=torch.int64)
                TP_FN = torch.zeros(model.num_label, dtype=torch.int64)
                for feature, label in test_data:
                    pred = model(feature)
                    flag = pred > 0.5
                    label = label > 0.5
                    TP += (flag * label).sum(dim=0)
                    TP_FP += flag.sum(dim=0)
                    TP_FN += label.sum(dim=0)

                precision = TP.float()*100/(TP_FP.float()+1e-5)
                recall = TP.float()*100/(TP_FN.float()+1e-5)
                fscore = 2*precision*recall/(precision+recall+1e-5)
                print('\n'.join('%s:\tcount=%d\tPrecision=%0.3f%%\tRecall=%0.3f%%\tFscore=%0.3f%%' % (
                    label, TP_FN[id], precision[id], recall[id], fscore[id]) for label, id in label2id.items()))

                for label, id in label2id.items():
                    writer.add_scalars('Prec', label, precision[id], epoch*len(train_data)+id)
                    writer.add_scalars('Recall', label, recall[id], epoch*len(train_data)+id)
                    writer.add_scalars('F1', label, fscore[id], epoch*len(train_data)+id)

                mean_f = fscore.mean()
                if mean_f > best_mean_f1:
                    best_mean_f1 = mean_f
                    torch.save(model, '%s_%0.3f.pt' % (config.model_prefix, best_mean_f1))


if __name__ == '__main__':

    with open(config.label_dict) as f:
        labels = [l.strip() for l in f.readlines() if len(l.strip()) > 0 and not l.startswith('#')]
        label2id = dict((l, id)for id, l in enumerate(labels))
    train_data = list(load(config.train, label2id))
    valid_data = list(load(config.valid, label2id))
    test_data = list(load(config.test, label2id))

    model = Classifier(feature_dim=config.word_embed_dim*2*6, num_label=len(label2id))

    train(model, train_data, valid_data, test_data, label2id)










