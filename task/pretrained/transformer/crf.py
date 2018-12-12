#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import torch
from torch import nn
from collections import namedtuple
from .base import MIN_SCORE
from .attention import MultiHeadedAttention, TransformerLayer


class LinearCRF(nn.Module):
    def __init__(self, hidden_size, num_tags, attention_num_heads=None, dropout=0.3):
        super(LinearCRF, self).__init__()

        self.attention = None if attention_num_heads is None \
            else TransformerLayer(hidden_size, attention_num_heads)

        self.hidden2emission = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, num_tags)
        )
        self._transition = nn.Parameter(torch.randn(num_tags, num_tags))
        self.hidden_size = hidden_size
        self.num_tags = num_tags

        self.reset_parameters()

    @property
    def transition(self):
        return self._transition

    def reset_parameters(self):
        nn.init.xavier_normal_(self._transition)

    def _forward_score(self, emissions, lens):
        max_len, batch_size, emission_size = emissions.size()
        assert emission_size == self.num_tags

        transition = self.transition.unsqueeze(0).expand(batch_size, -1, -1)
        forward_0 = emissions[0]

        forward_vars = [forward_0]
        for time in range(1, max_len, 1):
            forward_t = forward_vars[-1].unsqueeze(1).expand(-1, self.num_tags, -1)
            forward_t = (forward_t + transition).logsumexp(-1) + emissions[time]
            forward_vars.append(forward_t)

        return torch.stack([forward_vars[lens[b]-1][b] for b in range(batch_size)]).logsumexp(-1)

    def _transition_select(self, prev_tags, curr_tags):
        return self.transition.index_select(0, curr_tags).gather(1, prev_tags.unsqueeze(-1)).squeeze(-1)

    def _gold_score(self, emissions, lens, tags):
        '''
        :param emissions: Variable(FloatTensor(seq_len, batch, num_lables))
        :param tags: LongTensor(seq_len, batch)
        :param lens: sentence lengths
        :return: Variable(FloatTensor([batch]))
        '''
        max_len, batch_size, emission_size = emissions.size()
        assert emission_size == self.num_tags

        # [seq_len, batch]
        emissions = emissions.gather(-1, tags.unsqueeze(-1)).squeeze(-1)

        scores = [emissions[0]]

        for i in range(1, max_len, 1):
            scores.append(scores[-1] + self._transition_select(tags[i - 1], tags[i]) + emissions[i])

        return torch.stack([scores[lens[b]-1][b] for b in range(batch_size)])

    def neg_log_likelihood(self, hidden, lens, masks, tags):
        if self.attention is not None:
            hidden = self.attention(hidden, masks, batch_first=False)
        emissions = self.hidden2emission(hidden)
        forward_score = self._forward_score(emissions, lens)
        gold_score = self._gold_score(emissions, lens, masks.max(-1)[1])
        return (forward_score - gold_score).sum(), len(lens)

    def _viterbi_decode(self, emission):
        sen_len, _ = emission.size()

        backpointers = []
        forward_var = emission[0]
        for time in range(1, emission.size(0), 1):
            max_var, max_id = (forward_var.unsqueeze(0) + self.transition).max(-1)
            backpointers.append(max_id.tolist())
            forward_var = max_var + emission[time]

        best_score, best_id = forward_var.max(-1)
        best_path = [best_id.item()]
        for bp in reversed(backpointers):
            best_path.append(bp[best_path[-1]])

        best_path.reverse()
        return best_path, best_score

    def forward(self, hidden, lens, masks):
        '''
        :param hidden: len * batch_size * hidden_dim
        :param lens: batch_size * len
        :return:
        '''
        if self.attention is not None:
            hidden = self.attention(hidden, masks, batch_first=False)
        emissions = self.hidden2emission(hidden)
        return [self._viterbi_decode(emissions[:lens[sen_id], sen_id]) for sen_id in range(emissions.size(1))]

    # ------------------- only for test ---------------------
    def _valid_forward_score(self, emissions):
        seq_len, dim = emissions.size()

        forward_var = emissions[0]

        for time in range(1, seq_len, 1):
            forward_var = (forward_var.unsqueeze(0).expand(self.num_tags, -1) + self.transition).logsumexp(-1) + emissions[time]

        return forward_var

    def _valid_gold_score(self, emissions, tags):
        seq_len, dim = emissions.size()
        forward_var = emissions[0, tags[0]]
        for t in range(1, seq_len, 1):
            forward_var = forward_var + emissions[t, tags[t]] + self.transition[tags[t], tags[t-1]]

        return forward_var

    def valid_neg_log_likelihood(self, hiddens, lens, masks, tags):
        emissions = self.hidden2emission(self.dropout(hiddens))
        golds = masks.max(-1)[1]
        return sum(
            [self._valid_forward_score(emissions[:lens[b], b]) - self._valid_gold_score(emissions[:lens[b], b], golds[:lens[b], b])
             for b in range(emissions.size(1))]), len(lens)

    def _nbest(self, emission, topk):
        seq_len, _ = emission.size()
        Node = namedtuple('Node', ['tag', 'prev', 'score'])

        beam_t = [[Node(t, None, em)] for t, em in enumerate(emission[0].tolist())]
        beams = [beam_t]
        for time in range(1, seq_len, 1):
            beam_t = [
                [Node(curr_tag, prev, prev.score + emission[time, curr_tag].item() + self.transition[curr_tag, prev.tag].item())
                 for sub in beams[-1] for prev in sub]
                for curr_tag in range(self.num_tags)
            ]
            beam_t = [sorted(curr_tag, key=lambda node: node.score, reverse=True)[0:topk] for curr_tag in beam_t]
            beams.append(beam_t)

        last = sorted([node for curr_tag in beams[-1] for node in curr_tag], key=lambda n: n.score, reverse=True)[0:topk]

        print(last)

        paths = []
        scores = []
        for node in last:
            scores.append(node.score)
            path = []
            while node is not None:
                path.append(node.tag)
                node = node.prev
            path.reverse()
            paths.append(path)

        return list(zip(paths, scores))

    def nbest(self, hiddens, lens, masks, topk=5):
        if self.attention is not None:
            hiddens = self.attention(hiddens, masks, batch_first=False)
        emissions = self.hidden2emission(hiddens)
        return [self._nbest(emissions[:lens[sen_id], sen_id], topk) for sen_id in range(emissions.size(1))]


class MaskedCRF(LinearCRF):
    def __init__(self, hidden_size, num_tags, transition_constraints, attention_num_heads=None, dropout=0.3):
        super(MaskedCRF, self).__init__(hidden_size, num_tags, attention_num_heads=attention_num_heads)
        self.register_buffer('transition_constraints', transition_constraints)

    @property
    def transition(self):
        return self._transition.masked_fill(self.transition_constraints == 0, MIN_SCORE)

    def _mask_score(self, emissions, lens, masks, tags):
        '''
        :param emissions: FloatTensor(seq_len, batch_size, tag_size)
        :param lens: sentence lengths
        :param masks: ByteTensor(seq_len, batch_size, tag_size)
        :return: FloatTensor(batch_size)
        '''
        max_len, batch_size, emission_size = emissions.size()
        assert emission_size == self.num_tags

        transition = self.transition.unsqueeze(0)

        forward_0 = emissions[0].masked_fill(masks[0] == 0, MIN_SCORE)

        forward_vars = [forward_0]
        for time in range(1, max_len, 1):
            forward_t = forward_vars[-1].unsqueeze(1).expand(-1, self.num_tags, -1)
            forward_t = (forward_t + transition).logsumexp(-1) + emissions[time]
            forward_vars.append(forward_t.masked_fill(masks[time] == 0, MIN_SCORE))

        return torch.stack([forward_vars[lens[b] - 1][b] for b in range(batch_size)]).logsumexp(-1)

    def neg_log_likelihood(self, hiddens, lens, h_masks, tag_masks, tags):
        if self.attention:
            hiddens = self.attention(hiddens, h_masks, batch_first=False)

        emissions = self.hidden2emission(hiddens)
        forward_score = self._forward_score(emissions, lens)
        mask_score = self._mask_score(emissions, lens, tag_masks, tags)
        return (forward_score - mask_score).sum(), len(lens)

