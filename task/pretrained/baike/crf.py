#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import math
import torch
from torch import nn
from collections import namedtuple
from .base import MIN_SCORE
from .attention import MultiHeadedAttention


def make_mask(lens: torch.Tensor):
    masks = torch.ones(max(lens), lens.size(0), dtype=torch.uint8, device=lens.device)
    for i, l in enumerate(lens):
        masks[l:, i] = 0
    return masks


class EmissionLayer(nn.Module):
    def __init__(self, in_size: int, out_size: int, kernel_size: int=1, dropout=0.3):
        super(EmissionLayer, self).__init__()
        assert kernel_size % 2 == 1

        self.model = nn.Sequential(
            nn.Conv1d(in_size, in_size, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(in_size, out_size, kernel_size=1)
        )

    def forward(self, input: torch.Tensor, batch_first=False):
        input = input.permute(0, 2, 1) if batch_first else input.permute(1, 2, 0)

        output = self.model(input)

        output = output.permute(0, 2, 1) if batch_first else output.permute(2, 0, 1)

        return output


class LinearCRF(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_tags: int,
                 transition_constraints: torch.Tensor,
                 attention_num_heads=None,
                 dropout=0.1):
        super(LinearCRF, self).__init__()

        self.hidden2emission = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_tags)
        )

        self.attention = MultiHeadedAttention(
            attention_num_heads, hidden_size, dropout=dropout) if attention_num_heads else None

        self._transition = nn.Parameter(
            torch.randn(num_tags, num_tags).masked_fill_(transition_constraints == 0, MIN_SCORE))
        self.hidden_size = hidden_size
        self.num_tags = num_tags

        self.transition_constraints = nn.Parameter(transition_constraints, requires_grad=False)


    @property
    def transition(self):
        return self._transition.masked_fill(self.transition_constraints == 0, MIN_SCORE)

    def _transition_select(self, prev_tags, curr_tags):
        return self.transition.index_select(0, curr_tags).gather(1, prev_tags.unsqueeze(-1)).squeeze(-1)

    def _emission(self, hidden, lens):
        if self.attention:
            hidden, _ = self.attention(hidden, hidden, hidden, make_mask(lens))

        emissions = self.hidden2emission(hidden)

        return emissions

    def _forward_score(self, emissions, lens):
        max_len, batch_size, emission_size = emissions.size()
        assert emission_size == self.num_tags

        transition = self._transition

        forward_vars = [emissions[0]]
        for time in range(1, max_len, 1):
            forward_t = forward_vars[-1].unsqueeze(1).expand(-1, self.num_tags, -1)
            forward_t = (forward_t + transition.unsqueeze(0)).logsumexp(-1) + emissions[time]
            forward_vars.append(forward_t)

        return torch.stack([forward_vars[blen-1][bid] for bid, blen in enumerate(lens)]).logsumexp(-1)

    def _gold_score(self, emissions: torch.Tensor, lens: torch.Tensor, tags: torch.Tensor):
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

        return torch.stack([scores[blen-1][bid] for bid, blen in enumerate(lens)])

    def neg_log_likelihood(self,
                           hidden: torch.Tensor,
                           lens: torch.Tensor,
                           masks: torch.Tensor) -> torch.Tensor:
        emissions = self._emission(hidden, lens)
        forward_score = self._forward_score(emissions, lens)
        gold_score = self._gold_score(emissions, lens, masks.max(-1)[1])
        return (forward_score - gold_score).sum()

    def focal_loss(self, hidden: torch.Tensor, lens: torch.Tensor, masks: torch.Tensor, gamma=2):

        emissions = self._emission(hidden, lens)
        forward_score = self._forward_score(emissions, lens)
        gold_score = self._gold_score(emissions, lens, masks.max(-1)[1])

        logp = gold_score - forward_score
        probs = logp.exp()
        weights = (1 - probs).pow(gamma)
        return -torch.dot(weights, logp), len(lens)

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

    def batch_viterbi_decode(self, emission, lens):

        seq_len, batch_size, _ = emission.size()
        transition = self.transition.unsqueeze(0)
        backpointers = []
        forwards = [emission[0]]
        for time in range(1, seq_len, 1):
            max_var, max_id = (forwards[-1].unsqueeze(1) + transition).max(-1)
            backpointers.append(max_id)
            forwards.append(max_var + emission[time])

        results = []
        for sid, slen in enumerate(lens):
            best_score, best_id = forwards[slen-1][sid].max(-1)
            best_path = [best_id.item()]
            for time in range(slen-2, -1, -1):
                best_path.append(backpointers[time][sid, best_path[-1]].item())

            best_path.reverse()

            results.append((best_path, best_score))
        return results

    def forward(self, hidden, lens):
        '''
        :param hidden: len * batch_size * hidden_dim
        :param lens: batch_size * len
        :return:
        '''
        emissions = self._emission(hidden, lens)
        return self.batch_viterbi_decode(emissions, lens)
        # results = [self._viterbi_decode(emissions[:blen, bid]) for bid, blen in enumerate(lens)]
        # assert (b == e for b, e in zip(batch_results, results))
        # return [self._viterbi_decode(emissions[:blen, bid]) for bid, blen in enumerate(lens)]

    def predict_with_prob(self, hidden, lens):
        emissions = self._emission(hidden, lens)
        forward_score = self._forward_score(emissions, lens)
        predicted = self.batch_viterbi_decode(emissions, lens)
        # org_predicted = [self._viterbi_decode(emissions[:blen, bid]) for bid, blen in enumerate(lens)]
        return [(p, math.exp(logs - logz)) for (p, logs), logz in zip(predicted, forward_score.tolist())]

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

    def valid_neg_log_likelihood(self, hidden, lens, masks, tags):
        emissions = self._emission(hidden, lens)
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

    def nbest(self, hidden, lens, topk=5):
        emissions = self._emission(hidden, lens)
        forward_score = self._forward_score(emissions, lens)
        return [
            [(path, math.exp(logs - forward_score[sen_id].item()))
             for path, logs in self._nbest(emissions[:lens[sen_id], sen_id], topk)]
            for sen_id in range(emissions.size(1))]


class MaskedCRF(LinearCRF):
    def __init__(self, *args, **kwargs):
        super(MaskedCRF, self).__init__(*args, **kwargs)

    def _mask_score(self, emissions, lens, masks):
        '''
        :param emissions: FloatTensor(seq_len, batch_size, tag_size)
        :param lens: sentence lengths
        :param masks: ByteTensor(seq_len, batch_size, tag_size)
        :return: FloatTensor(batch_size)
        '''
        max_len, batch_size, emission_size = emissions.size()
        assert emission_size == self.num_tags

        forward_0 = emissions[0].masked_fill(masks[0] == 0, MIN_SCORE)

        forward_vars = [forward_0]
        for time in range(1, max_len, 1):
            forward_t = forward_vars[-1].unsqueeze(1).expand(-1, self.num_tags, -1)
            forward_t = (forward_t + self.transition.unsqueeze(0)).logsumexp(-1) + emissions[time]
            forward_vars.append(forward_t.masked_fill(masks[time] == 0, MIN_SCORE))

        return torch.stack([forward_vars[blen - 1][bid] for bid, blen in enumerate(lens)]).logsumexp(-1)

    def neg_log_likelihood(self, hidden: torch.Tensor, lens: torch.Tensor, tag_masks: torch.Tensor):
        emissions = self._emission(hidden, lens)
        forward_score = self._forward_score(emissions, lens)
        mask_score = self._mask_score(emissions, lens, tag_masks)
        return (forward_score - mask_score).sum()

    def focal_loss(self, hidden: torch.Tensor, lens: torch.Tensor, tag_masks: torch.Tensor, gamma=2):
        emissions = self._emission(hidden, lens)
        forward_score = self._forward_score(emissions, lens)
        mask_score = self._mask_score(emissions, lens, tag_masks)

        logp = mask_score - forward_score
        probs = logp.exp()
        weights = (1 - probs).pow(gamma)
        return -torch.dot(weights, logp), len(lens)

