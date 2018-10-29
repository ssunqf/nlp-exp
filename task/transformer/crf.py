#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import torch
from torch import nn
from collections import namedtuple


class LinearCRF(nn.Module):
    def __init__(self, hidden_dim, tag_size):
        super(LinearCRF, self).__init__()
        self.hidden2emission = nn.Linear(hidden_dim, tag_size)
        self.begin_transition = nn.Parameter(torch.randn(tag_size))
        self.transition = nn.Parameter(torch.randn(tag_size, tag_size))
        self.end_transition = nn.Parameter(torch.randn(tag_size))
        self.hidden_dim = hidden_dim
        self.tag_size = tag_size

    def _forward_score(self, emissions, lens):
        max_len, batch_size, emission_size = emissions.size()
        assert emission_size == self.tag_size

        transition = self.transition.unsqueeze(0).expand(batch_size, -1, -1)
        forward_0 = emissions[0] + self.begin_transition.unsqueeze(0).expand(batch_size, -1)

        forward_vars = [forward_0]
        for time in range(1, max_len, 1):
            forward_t = forward_vars[-1].unsqueeze(1).expand(-1, self.tag_size, -1)
            forward_t = (forward_t + transition).logsumexp(-1) + emissions[time]
            forward_vars.append(forward_t)

        last_vars = torch.stack([forward_vars[lens[b]-1][b] for b in range(batch_size)])
        return (last_vars + self.end_transition.unsqueeze(0).expand(batch_size, -1)).logsumexp(-1)

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
        assert emission_size == self.tag_size

        # [seq_len, batch]
        emissions = emissions.gather(-1, tags.unsqueeze(-1)).squeeze(-1)

        scores = [self.begin_transition.gather(0, tags[0]) + emissions[0]]

        for i in range(1, max_len, 1):
            scores.append(scores[-1] + self._transition_select(tags[i - 1], tags[i]) + emissions[i])

        last_scores = torch.stack([scores[lens[b]-1][b] for b in range(batch_size)])
        last_tags = emissions.new_tensor([tags[lens[b]-1, b] for b in range(batch_size)], dtype=torch.long)

        return last_scores + self.end_transition.gather(0, last_tags)

    def neg_log_likelihood(self, hiddens, lens, masks, tags):
        emissions = self.hidden2emission(hiddens)
        forward_score = self._forward_score(emissions, lens)
        gold_score = self._gold_score(emissions, lens, masks.max(-1)[1])
        return (forward_score - gold_score).sum(), len(lens)

    def _viterbi_decode(self, emission):
        sen_len, _ = emission.size()

        backpointers = []
        forward_var = self.begin_transition + emission[0]
        for time in range(1, emission.size(0), 1):
            max_var, max_id = (forward_var.unsqueeze(0) + self.transition).max(-1)
            backpointers.append(max_id.tolist())
            forward_var = max_var + emission[time]

        best_score, best_id = (forward_var + self.end_transition).max(-1)
        best_path = [best_id.item()]
        for bp in reversed(backpointers):
            best_path.append(bp[best_path[-1]])

        best_path.reverse()
        return best_path, best_score

    def forward(self, hiddens, lens):
        '''
        :param hiddens: len * batch_size * hidden_dim
        :param lens: batch_size * len
        :return:
        '''
        emissions = self.hidden2emission(hiddens)
        return [self._viterbi_decode(emissions[:lens[sen_id], sen_id]) for sen_id in range(emissions.size(1))]

    # ------------------- only for test ---------------------
    def _valid_forward_score(self, emissions):
        seq_len, dim = emissions.size()

        forward_var = emissions[0] + self.begin_transition

        for time in range(1, seq_len, 1):
            forward_var = (forward_var.unsqueeze(0).expand(self.tag_size, -1) + self.transition).logsumexp(-1) + emissions[time]

        return (forward_var + self.end_transition).logsumexp(-1)

    def _valid_gold_score(self, emissions, tags):
        seq_len, dim = emissions.size()
        forward_var = emissions[0, tags[0]] + self.begin_transition[tags[0]]
        for t in range(1, seq_len, 1):
            forward_var = forward_var + emissions[t, tags[t]] + self.transition[tags[t], tags[t-1]]

        return forward_var + self.end_transition[tags[seq_len-1]]

    def valid_neg_log_likelihood(self, hiddens, lens, masks, tags):
        emissions = self.hidden2emission(hiddens)
        golds = masks.max(-1)[1]
        return sum(
            [self._valid_forward_score(emissions[:lens[b], b]) - self._valid_gold_score(emissions[:lens[b], b], golds[:lens[b], b])
             for b in range(emissions.size(1))]), len(lens)

    def _nbest(self, emission, topk):
        seq_len, _ = emission.size()
        Node = namedtuple('Node', ['tag', 'prev', 'score'])

        stack = [[Node(t, None, emission[0, t] + self.begin_transition[t])] for t in range(self.tag_size)]
        stacks = [stack]
        for time in range(1, seq_len, 1):
            stack = [
                [Node(curr_tag, node, node.score + emission[time, curr_tag] + self.transition[curr_tag, node.tag])
                 for sub in stacks[-1] for node in sub]
                for curr_tag in range(self.tag_size)
            ]
            stack = [sorted(curr_tag, key=lambda node: node.score, reverse=True)[0:topk] for curr_tag in stack]
            stacks.append(stack)

        last = [Node(-1, node, node.score + self.end_transition[node.tag])
                for curr_tag in stacks[-1] for node in curr_tag]
        last = sorted(last, key=lambda n: n.score, reverse=True)[0:topk]

        paths = []
        scores = []
        for node in last:
            scores.append(node.score.item())
            path = []
            node = node.prev
            while True:
                path.append(node.tag)
                if node.prev is None:
                    break
                node = node.prev
            path.reverse()
            paths.append(path)

        return list(zip(paths, scores))

    def nbest(self, hiddens, lens, topk=5):
        emissions = self.hidden2emission(hiddens)
        return [self._nbest(emissions[:lens[sen_id], sen_id], topk) for sen_id in range(emissions.size(1))]


class MaskedCRF(LinearCRF):
    def __init__(self, hidden_dim, tag_size, begin_constraints, transition_constraints, end_constraints):
        super(MaskedCRF, self).__init__(hidden_dim, tag_size)
        self.begin_constraints = nn.Parameter(begin_constraints, requires_grad=False)
        self.transition_constraints = nn.Parameter(transition_constraints, requires_grad=False)
        self.end_constraints = nn.Parameter(end_constraints, requires_grad=False)

    def _mask_score(self, emissions, lens, masks, tags):
        '''
        :param emissions: FloatTensor(seq_len, batch_size, tag_size)
        :param lens: sentence lengths
        :param masks: ByteTensor(seq_len, batch_size, tag_size)
        :return: FloatTensor(batch_size)
        '''
        max_len, batch_size, emission_size = emissions.size()
        assert emission_size == self.tag_size

        begin_transition = self.begin_transition.masked_fill(self.begin_constraints == 0, -1e10).unsqueeze(0)
        transition = self.transition.masked_fill(self.transition_constraints == 0, -1e10).unsqueeze(0)
        end_transition = self.end_transition.masked_fill(self.end_constraints == 0, -1e10).unsqueeze(0)

        forward_0 = (emissions[0] + begin_transition).masked_fill(masks[0] == 0, -1e10)

        forward_vars = [forward_0]
        for time in range(1, max_len, 1):
            forward_t = forward_vars[-1].unsqueeze(1).expand(-1, self.tag_size, -1)
            forward_t = (forward_t + transition).logsumexp(-1) + emissions[time]
            forward_vars.append(forward_t.masked_fill(masks[time] == 0, -1e10))

        last_vars = torch.stack([forward_vars[lens[b] - 1][b] for b in range(batch_size)])

        return (last_vars + end_transition).logsumexp(-1)

    def neg_log_likelihood(self, hiddens, lens, masks, tags):
        emissions = self.hidden2emission(hiddens)
        forward_score = self._forward_score(emissions, lens)
        mask_score = self._mask_score(emissions, lens, masks, tags)

        return (forward_score - mask_score).sum(), len(lens)

