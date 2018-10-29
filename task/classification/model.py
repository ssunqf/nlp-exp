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
        for line in tqdm(f):
            _, labels, features = line.split('\t')
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
            if config.use_cuda:
                batch_feature = batch_feature.cuda()
                batch_label = batch_label.cuda()
            yield batch_feature, batch_label

        print('\n'.join([label + ': ' + str(counts[id]) for label, id in label2id.items()]))


def train(model: Classifier, train_data, valid_data, test_data, label2id: dict, max_epoch=15, weight_decay=5e-5):
    optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay, amsgrad=True)

    writer = SummaryWriter(log_dir=config.summary_dir)

    best_mean_f1 = 0
    iter_step = 1
    for epoch in range(max_epoch):
        train_loss, train_count = 0., 0.
        for batch in tqdm(train_data, total=len(train_data)):
            iter_step += 1
            feature, label = batch
            model.train()
            model.zero_grad()

            if config.use_cuda:
                feature = feature.cuda()
                label = label.cuda()
            loss = model.loss(feature, label)

            train_loss += loss.item()
            train_count += label.size()[0]
            loss.backward()
            optimizer.step()

            writer.add_scalars('Loss', {'train-'+str(weight_decay): train_loss/(train_count+1e-5)}, iter_step)

            if iter_step % 1000 == 0:
                valid_loss, valid_count = 0., 0.
                model.eval()
                for vfeature, vlabel in valid_data:
                    if config.use_cuda:
                        vfeature = vfeature.cuda()
                        vlabel = vlabel.cuda()
                    valid_loss += model.loss(vfeature, vlabel).item()
                    valid_count += vlabel.size()[0]
                writer.add_scalars('Loss', {'valid-'+str(weight_decay): valid_loss/(valid_count+1e-5)}, iter_step)

                print('epoch %d\titeration %d\tTrain loss=%0.5f\t\tValid loss=%0.5f' % (
                    epoch, iter_step, train_loss/(train_count+1e-5), valid_loss/(valid_count+1e-5)))

                train_loss = 0.
                train_count = 0.
            if iter_step % 10000 == 0:
                # test
                model.eval()
                TP = torch.zeros(model.num_label, dtype=torch.int64)
                TP_FP = torch.zeros(model.num_label, dtype=torch.int64)
                TP_FN = torch.zeros(model.num_label, dtype=torch.int64)
                for feature, label in test_data:
                    if config.use_cuda:
                        feature = feature.cuda()
                        label = label.cuda()
                    pred = model(feature)
                    flag = pred > 0.5
                    label = label > 0.5
                    TP += (flag * label).sum(dim=0).cpu()
                    TP_FP += flag.sum(dim=0).cpu()
                    TP_FN += label.sum(dim=0).cpu()

                precision = TP.float()*100/(TP_FP.float()+1e-5)
                recall = TP.float()*100/(TP_FN.float()+1e-5)
                fscore = 2*precision*recall/(precision+recall+1e-5)
                print('\n'.join('%s:\tcount=%d\tPrecision=%0.3f%%\tRecall=%0.3f%%\tFscore=%0.3f%%' % (
                    label, TP_FN[id], precision[id], recall[id], fscore[id]) for label, id in label2id.items()))

                for label, id in label2id.items():
                    writer.add_scalars('Prec-'+str(weight_decay), {label: precision[id]}, iter_step)
                    writer.add_scalars('Recall-'+str(weight_decay), {label: recall[id]}, iter_step)
                    writer.add_scalars('F1-'+str(weight_decay), {label: fscore[id]}, iter_step)

                writer.add_scalars('avg-Prec', {str(weight_decay): precision.mean()}, iter_step)
                writer.add_scalars('avg-Recall', {str(weight_decay): recall.mean()}, iter_step)
                writer.add_scalars('avg-F1', {str(weight_decay): fscore.mean()}, iter_step)

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

    if config.use_cuda:
        model = model.cuda()

    weight_decays=[0.000004, 0.000003, 0.000002, 0.000001]
    for i in weight_decays:
        train(model, train_data, valid_data, test_data, label2id, max_epoch=30, weight_decay=i)










