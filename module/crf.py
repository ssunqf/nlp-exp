#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-


from typing import List

import torch
import torch.nn as nn

from overrides import overrides

# reference

def log_sum_exp(vecs, axis):
    max_val, _ = vecs.max(axis)
    vecs = vecs - max_val.unsqueeze(axis)
    out_val = vecs.exp().sum(axis).log()
    # print(max_val, out_val)
    return max_val + out_val


class CRF(nn.Module):
    def __init__(self, feature_dim, num_labels, dropout=0.5):
        super(CRF, self).__init__()
        self.hidden_dim = feature_dim
        self.num_labels = num_labels
        self.feature2labels = nn.Sequential(nn.Dropout(dropout),
                                            nn.Linear(feature_dim, num_labels))
        self.start_transition = nn.Parameter(torch.randn(self.num_labels))
        # tags[i + 1] -> tags[i]
        self.transitions = nn.Parameter(torch.randn(self.num_labels, self.num_labels))
        self.end_transition = nn.Parameter(torch.randn(self.num_labels))

    def forward_alg(self, emissions, lens):
        '''
        :param emissions: Variabel(FloatTensor([seq_len, batch, num_labels]))
        :return: Variable(FloatTensor([batch]))
        '''
        max_len, batch_size, num_labels = emissions.size()
        assert num_labels == self.num_labels

        # [batch, num_labels]
        forward_scores = emissions[0] + self.start_transition.unsqueeze(0).expand(batch_size, num_labels)
        last_scores = []
        unfinished = sorted(enumerate(lens), key=lambda x: x[1], reverse=True)

        for i in range(1, max_len, 1):
            # record finished
            while len(unfinished) > 0 and unfinished[-1][1]==i:
                finished = unfinished.pop()
                last_scores.append((finished[0], forward_scores[finished[0]].clone()))

            forward_scores = log_sum_exp(forward_scores.unsqueeze(1).expand(batch_size, num_labels, num_labels) +\
                                         self.transitions.unsqueeze(0), axis=-1) + emissions[i]

        while len(unfinished) > 0:
            finished = unfinished.pop()
            last_scores.append((finished[0], forward_scores[finished[0]].clone()))

        # [batch, num_labels]
        last_scores = torch.stack([score for _, score in sorted(last_scores, key=lambda x: x[0])])

        # [batch, num_labels]
        forward_scores = last_scores + self.end_transition.unsqueeze(0).expand(batch_size, num_labels)
        return log_sum_exp(forward_scores, axis=-1)

    def _transition_select(self, prev_tags, curr_tags):
        return self.transitions.index_select(0, curr_tags).gather(1, prev_tags.unsqueeze(-1)).squeeze(-1)

    def score_sentence(self, emissions, tags, lens: List[int]):
        '''

        :param emissions: Variable(FloatTensor([seq_len, batch, num_lables]))
        :param tags: Variable(FloatTensor([seq_len, batch]))
        :param lens: sentence lengths
        :return: Variable(FloatTensor([batch]))
        '''
        max_seq, batch, num_lables = emissions.size()
        assert num_lables == self.num_labels

        # [seq_len, batch]
        emissions = emissions.gather(-1, tags.unsqueeze(-1)).squeeze(-1)

        unfinished = sorted(enumerate(lens), key=lambda x: x[1], reverse=True)
        last = []
        scores = self.start_transition.gather(0, tags[0]) + emissions[0]

        for i in range(1, max_seq, 1):
            # record finished
            while len(unfinished) > 0 and unfinished[-1][1]==i:
                finished = unfinished.pop()
                last.append((finished[0], tags[i-1, finished[0]], scores[finished[0]].clone()))

            scores = scores + self._transition_select(tags[i-1], tags[i]) + emissions[i]

        while len(unfinished) > 0:
            finished = unfinished.pop()
            last.append((finished[0], tags[-1, finished[0]], scores[finished[0]].clone()))

        finished = sorted(last, key=lambda x: x[0])
        last_scores = torch.cat([s for _, _, s in finished])
        last_tags = torch.cat([t for _, t, _ in finished])

        scores = last_scores + self.end_transition.gather(0, last_tags)
        return scores

    def _viterbi_decode(self, emissions):
        '''

        :param emissions: [seq_len, num_labels]
        :return:
        '''

        back_pointers = []

        # [num_labels]
        scores = self.start_transition + emissions[0]

        for i in range(1, emissions.size(0)):
            # [num_labels, num_labels]
            scores_with_transitions = scores.unsqueeze(0).expand_as(self.transitions) + self.transitions
            max_scores, max_index = scores_with_transitions.max(-1)
            back_pointers.append(max_index.data.tolist())
            scores = emissions[i] + max_scores

        scores = scores + self.end_transition
        max_score, max_index = scores.max(-1)

        viterbi = [max_index.data[0]]
        for bp in reversed(back_pointers):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()
        return max_score.data[0], viterbi

    def criterion(self, feats, golds, lens):
        '''

        :param feats: PackedSequence
        :param golds: PackedSequence
        :return:
        '''
        # [seq_len, batch, num_labels]
        emissions = self.feature2labels(feats)
        forward_score = self.forward_alg(emissions, lens)
        gold_score = self.score_sentence(emissions, golds, lens)

        return (forward_score - gold_score).sum(), len(lens)

    @overrides
    def forward(self, feats, lens: List[int]):
        '''
        unsupported batch process
        :param feats: Variable(FloatTensor(seq_len, batch, feat_dim))
        :return: score, tag sequence
        '''
        # [seq_len, batch, num_labels]
        emissions = self.feature2labels(feats)
        return [self._viterbi_decode(emissions[0:lens[i], i]) for i in range(len(lens))]


'''
    处理部分标注的数据
'''
class PartialCRF(nn.Module):
    def __init__(self, feature_dim,
                 num_labels,
                 begin_constraints,
                 end_constraints,
                 transition_constraints,
                 dropout=0.5):
        super(PartialCRF, self).__init__()
        self.hidden_dim = feature_dim
        self.num_labels = num_labels
        self.transition_constraints = transition_constraints
        self.feature2labels = nn.Sequential(nn.Dropout(dropout),
                                            nn.Linear(feature_dim, num_labels))
        self.begin_transition = nn.Parameter(torch.full([self.num_labels], 1./self.num_labels))
        self.begin_constraints = begin_constraints
        # tags[i + 1] -> tags[i]
        self.transitions = nn.Parameter(torch.full([self.num_labels, self.num_labels], 1./self.num_labels))
        self.transition_constraints = transition_constraints

        self.end_transition = nn.Parameter(torch.full([self.num_labels], 1./self.num_labels))
        self.end_constraints = end_constraints

    @overrides
    def to(self, device: torch.device, dtype: torch.dtype):
        self.begin_constraints = self.begin_constraints.to(device)
        self.end_constraints = self.end_constraints.to(device)
        self.transition_constraints = self.transition_constraints.to(device)

        return super(PartialCRF, self).to(device, dtype)

    def forward_alg(self, emissions, lens):
        '''
        :param emissions: Variabel(FloatTensor([seq_len, batch, num_labels]))
        :return: Variable(FloatTensor([batch]))
        '''
        max_len, batch_size, num_labels = emissions.size()
        assert num_labels == self.num_labels

        # [batch, num_labels]
        forward_scores = emissions[0] + self.begin_transition.unsqueeze(0).expand(batch_size, num_labels)
        last_scores = []
        unfinished = sorted(enumerate(lens), key=lambda x: x[1], reverse=True)

        for i in range(1, max_len, 1):
            # record finished
            while len(unfinished) > 0 and unfinished[-1][1]==i:
                finished = unfinished.pop()
                last_scores.append((finished[0], forward_scores[finished[0]].clone()))

            forward_scores = log_sum_exp(forward_scores.unsqueeze(1).expand(batch_size, num_labels, num_labels) +\
                                         self.transitions.unsqueeze(0), axis=-1) + emissions[i]

        while len(unfinished) > 0:
            finished = unfinished.pop()
            last_scores.append((finished[0], forward_scores[finished[0]].clone()))

        # [batch, num_labels]
        last_scores = torch.stack([score for _, score in sorted(last_scores, key=lambda x: x[0])])

        # [batch, num_labels]
        forward_scores = last_scores + self.end_transition.unsqueeze(0).expand(batch_size, num_labels)
        return log_sum_exp(forward_scores, axis=-1)

    def partial_score(self, emissions, tags, lens: List[int]):
        '''
        :param emissions: Variabel(FloatTensor([seq_len, batch, num_labels]))
        :return: Variable(FloatTensor([batch]))
        '''
        max_len, batch_size, num_labels = emissions.size()
        assert num_labels == self.num_labels

        # [batch, num_labels]
        forward_scores = emissions[0] + self.begin_transition.masked_fill(self.begin_constraints, -1e20) \
            .unsqueeze(0).expand(batch_size, num_labels)
        forward_scores = forward_scores.masked_fill(tags[0], -1e20)

        last_scores = []
        unfinished = sorted(enumerate(lens), key=lambda x: x[1], reverse=True)

        for i in range(1, max_len, 1):
            # record finished
            while len(unfinished) > 0 and unfinished[-1][1]==i:
                finished = unfinished.pop()
                last_scores.append((finished[0], forward_scores[finished[0]].clone()))

            forward_scores = log_sum_exp(forward_scores.unsqueeze(1).expand(batch_size, num_labels, num_labels) \
                             + self.transitions.masked_fill(self.transition_constraints, -1e20).unsqueeze(0), axis=-1) \
                             + emissions[i]

            forward_scores = forward_scores.masked_fill(tags[i], -1e20)

        while len(unfinished) > 0:
            finished = unfinished.pop()
            last_scores.append((finished[0], forward_scores[finished[0]].clone()))

        # [batch, num_labels]
        last_scores = torch.stack([score for _, score in sorted(last_scores, key=lambda x: x[0])])

        # [batch, num_labels]
        forward_scores = last_scores + \
                         self.end_transition.masked_fill(self.end_constraints, -1e20)\
                             .unsqueeze(0).expand(batch_size, num_labels)

        return log_sum_exp(forward_scores, axis=-1)

    def _viterbi_decode(self, emissions):
        '''
        :param emissions: [seq_len, num_labels]
        :return:
        '''
        back_pointers = []

        # [num_labels]
        scores = self.begin_transition + emissions[0]
        for i in range(1, emissions.size(0)):
            # [num_labels, num_labels]
            scores_with_transitions = scores.unsqueeze(0).expand_as(self.transitions) + self.transitions
            max_scores, max_index = scores_with_transitions.max(-1)
            back_pointers.append(max_index.data.tolist())
            scores = emissions[i] + max_scores
        scores = scores + self.end_transition
        max_score, max_index = scores.max(-1)

        viterbi = [max_index.item()]
        for bp in reversed(back_pointers):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()
        return max_score.item(), viterbi

    def criterion(self, feats, golds, lens):
        '''

        :param feats: PackedSequence
        :param golds: PackedSequence
        :return:
        '''
        # [seq_len, batch, num_labels]
        emissions = self.feature2labels(feats)
        forward_score = self.forward_alg(emissions, lens)
        partial_score = self.partial_score(emissions, golds, lens)

        # print(partial_score, forward_score)
        return (forward_score - partial_score).sum(), len(lens)

    @overrides
    def forward(self, feats, lens: List[int]):
        '''
        unsupported batch process
        :param feats: Variable(FloatTensor(seq_len, batch, feat_dim))
        :return: score, tag sequence
        '''
        # [seq_len, batch, num_labels]
        emissions = self.feature2labels(feats)
        return [self._viterbi_decode(emissions[0:lens[i], i]) for i in range(len(lens))]



