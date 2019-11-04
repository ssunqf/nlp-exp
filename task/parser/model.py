#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import copy
import itertools
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence


def bundle(lstm_iter):
    if lstm_iter is None:
        return None
    lstm_iter = tuple(lstm_iter)
    if lstm_iter[0] is None:
        return None
    return torch.cat(lstm_iter, 0).chunk(2, 1)


def unbundle(state):
    if state is None:
        return itertools.repeat(None)
    return torch.split(torch.cat(state, 1), 1, 0)


class BoundaryLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(BoundaryLSTM, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.dropout_layer = nn.Dropout(self.dropout)
        self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim, num_layers)

        self._begin = torch.nn.init.uniform(nn.Parameter(torch.FloatTensor(1, hidden_dim*2)), -1, 1)

        self._end = torch.nn.init.uniform(nn.Parameter(torch.FloatTensor(1, hidden_dim*2)), -1, 1)

    def begin(self):
        return self._begin

    def end(self):
        return self._end

    def forward_end(self, hidden):
        return self.forward([self._end]*len(hidden), hidden)

    def forward(self, input, hidden=None):
        """
        :param input: [tensor(1, input_dim)] * batch_size or list
        :param hidden: [tensor(1, hidden_dim*2)] * batch_size or None
        :return: [tensor(1, hidden_dim*2)] * batch_size,
        """
        if isinstance(input, list):
            input = torch.cat(input, 0)

        if hidden is None:
            hidden = [self._begin] * input.size(0)
        if isinstance(hidden, list):
            hidden = bundle(hidden)

        new_hx, new_cx = self.lstm_cell(self.dropout_layer(input), hidden)

        return unbundle([new_hx, new_cx])


# 依存树的组合函数
class Composition(nn.Module):
    def __init__(self, node_dim, relation_dim, dropout=0.2):
        super(Composition, self).__init__()

        self.node_dim = node_dim
        self.relation_dim = relation_dim
        self.dropout = dropout
        self.model = nn.Sequential(nn.Dropout(self.dropout),
                                   nn.Linear(node_dim * 2 + relation_dim, node_dim),
                                   nn.Tanh())

    def forward(self, heads, modifiers, relations):
        '''
        :param heads: [tensor(1, self.node_dim)] * batch_size  or  tensor(batch_size, self.node_dim)
        :param modifiers:  [tensor(1, self.node_dim)] * batch_size   or  tensor(batch_size, self.node_dim)
        :param relations: [tensor(1, self.relation_dim)] * batch_size   or  tensor(batch_size, self.relation_dim)
        :return: [tensor(1, self.node_dim)] * batch_size
        '''
        if isinstance(heads, list):
            heads = torch.cat(heads, 0)
        if isinstance(modifiers, list):
            modifiers = torch.cat(modifiers, 0)
        if isinstance(relations, list):
            relations = torch.cat(relations, 0)

        comp = self.model(torch.cat([heads, modifiers, relations], 1))

        return comp.split(1, 0)


class Node:
    def __init__(self, index, chars, pos=None, head_index=-1, relation=None, lefts=None, rights=None):
        self.index = index
        self.chars = chars if chars else []
        self.pos = pos
        self.head_index = head_index
        self.relation = relation
        self.lefts = lefts if lefts else []
        self.rights = rights if rights else []

    def __len__(self):
        return len(self.chars)

    def __str__(self):
        return '(%s\t%s\t%s)' % ('\t'.join([str(left) for relation, left in self.lefts]),
                                 ''.join(self.chars),
                                 '\t'.join([str(right) for relation, right in self.rights]))

    def __deepcopy__(self, memodict={}):

        index = copy.copy(self.index)
        chars = copy.copy(self.chars)
        pos = copy.copy(self.pos)
        head_index = copy.copy(self.head_index)
        relation = copy.copy(self.relation)
        lefts = copy.deepcopy(self.lefts, memodict)
        rights = copy.deepcopy(self.rights, memodict)
        new_node = Node(index, chars, pos, head_index, relation, lefts, rights)
        memodict[id(self)] = new_node
        return new_node


class Actions:
    SHIFT = 0
    APPEND = 1
    ARC_LEFT = 2
    ARC_RIGHT = 3
    #ROOT = 4
    max_len = 4


class Transition:
    def __init__(self, action, label):
        self.action = action
        self.label = label

    def __eq__(self, other):
        return hash(self) == hash(other) and self.action == other.action and self.label == other.label

    def __hash__(self):
        return hash(self.action) + hash(self.label)

    def __str__(self):
        return '%d,%s' % (self.action, self.label)


class UDTree:
    def __init__(self, nodes, root):
        self.nodes = nodes
        self.root = root

    def linearize(self):
        chars = list(itertools.chain.from_iterable([node.chars for node in self.nodes]))

        def dfs(node):
            for left in node.lefts:
                yield from dfs(left)

            yield Transition(Actions.SHIFT, node.pos)
            for c in node.chars[1:]:
                yield Transition(Actions.APPEND, None)

            for l in node.lefts[::-1]:
                yield Transition(Actions.ARC_LEFT, l.relation)

            for right in node.rights:
                yield from dfs(right)
                yield Transition(Actions.ARC_RIGHT, right.relation)


        return chars, list(dfs(self.root))

    def get_words(self):
        return [''.join(node.chars) for node in self.nodes]

    def to_line(self):
        return '\t'.join(['%s#%s#%d#%s' % (''.join(node.chars), node.pos, node.head_index, node.relation) for node in self.nodes])

    @staticmethod
    def create(chars, transitons):
        nodes = []
        stack = []
        char_index = 0
        word_index = 0
        for tran in transitons:
            if tran.action == Actions.SHIFT:
                new_node = Node(word_index, [chars[char_index]], tran.label)
                stack.append(new_node)
                nodes.append(new_node)
                char_index += 1
                word_index += 1
            elif tran.action == Actions.APPEND:
                stack[-1].chars.append(chars[char_index])
                char_index += 1
            elif tran.action == Actions.ARC_LEFT:
                head = stack.pop()
                modifier = stack.pop()
                modifier.head_index = head.index
                modifier.relation = tran.label
                head.lefts.append(modifier)
                stack.append(head)
            elif tran.action == Actions.ARC_RIGHT:
                modifier = stack.pop()
                head = stack[-1]
                head.rights.append(modifier)
                modifier.head_index = head.index
                modifier.relation = tran.label

        for node in nodes:
            node.lefts = list(reversed(node.lefts))

        stack[-1].relation = 'root'

        return UDTree(nodes, stack[-1])

    @staticmethod
    def parse_stanford_format(tokens):
        nodes = [Node(index, chars, pos, int(parent_id)-1, relation) for index, (chars, pos, parent_id, relation) in enumerate(tokens)]

        root_id = 0
        for id, (token, pos, parent_id, relation) in enumerate(tokens):
            parent_id = int(parent_id) - 1

            if parent_id == -1:
                root_id = id
            elif parent_id > id:
                nodes[parent_id].lefts.append(nodes[id])
            elif parent_id < id:
                nodes[parent_id].rights.append(nodes[id])

        def validate(tree):
            gold = tree.to_line()

            chars, trans = tree.linearize()
            new_tree = UDTree.create(chars, trans)
            new_line = new_tree.to_line()

            return gold == new_line

        tree = UDTree(nodes, nodes[root_id])

        if validate(tree) is False:
            print('invalid tree:')
            print(tree.to_line())
            return None

        return UDTree(nodes, nodes[root_id])


class TransitionClassifier(nn.Module):

    def __init__(self, input_dim, num_transition, dropout=0.2):
        super(TransitionClassifier, self).__init__()

        self.num_transition = num_transition
        self.input_dim = input_dim

        self.dropout = dropout

        self.ffn = nn.Sequential(nn.Dropout(dropout),
                                 nn.Linear(self.input_dim, self.input_dim//2),
                                 nn.Sigmoid(),
                                 nn.Dropout(dropout),
                                 nn.Linear(self.input_dim//2, self.input_dim//2),
                                 nn.Sigmoid(),
                                 nn.Linear(self.input_dim//2, num_transition))

    def forward(self, buffer_hiddens, stack_hiddens, transition_hiddens, mask):

        buffers = bundle([b[-1] for b in buffer_hiddens])[0]
        stacks1 = bundle([s[-1] for s in stack_hiddens])[0]
        stacks2 = bundle([s[-2] for s in stack_hiddens])[0]
        transitions = bundle([t[-1] for t in transition_hiddens])[0]
        features = torch.cat([buffers, stacks1, stacks2, transitions], 1)

        return F.log_softmax(self.ffn(features).masked_fill(mask == 0, -1e15), 1)


class State:
    def __init__(self, nodes,
                 buffer, buffer_hidden,
                 stack, stack_hidden,
                 transitions, transition_hidden,
                 score=0.0):
        self.nodes = nodes
        self.buffer = buffer
        self.buffer_hidden = buffer_hidden
        self.stack = stack
        self.stack_hidden = stack_hidden
        self.transitions = transitions
        self.transitions_hidden = transition_hidden
        self.total_score = score
        self.scores = []  # score list for every transition

    def __deepcopy__(self, memodict={}):
        nodes = copy.deepcopy(self.nodes, memodict)
        buffer = copy.copy(self.buffer)
        buffer_hidden = copy.copy(self.buffer_hidden)
        stack = copy.copy(self.stack)
        stack_hidden = copy.copy(self.stack_hidden)
        transitions = copy.copy(self.transitions)
        transitions_hidden = copy.copy(self.transitions_hidden)
        total_score = copy.copy(self.total_score)
        scores = copy.copy(self.scores)

        new_state = State(nodes,
                          buffer, buffer_hidden,
                          stack, stack_hidden,
                          transitions, transitions_hidden,
                          total_score,
                          scores)
        memodict[id(self)] = new_state
        return new_state


# reference https://www.researchgate.net/profile/Yue_Zhang4/publication/266376262_Character-Level_Chinese_Dependency_Parsing/links/542e18030cf277d58e8e9908/Character-Level-Chinese-Dependency-Parsing.pdf
class ArcStandard(nn.Module):
    import math
    MIN_LOG_PROB = math.log(1e-2)

    def __init__(self, input_dim, hidden_dim,
                 transition_dict, relation_dict, pos_dict,
                 dropout=0.2, teacher_forcing_ratio=0.5):
        super(ArcStandard, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.transition_dict = transition_dict
        self.relation_dict = relation_dict
        self.pos_dict = pos_dict
        self.dropout = dropout

        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.topk = 3

        # buffer hidden
        self.buffer_dim = self.hidden_dim
        self.buffer_lstm = BoundaryLSTM(input_dim, self.buffer_dim, 1, self.dropout)

        #self.buffer_lstm = nn.LSTM(input_dim, self.buffer_dim, 1, dropout=self.dropout, bidirectional=False)

        # word embedding from char rnn
        self.word_dim = self.hidden_dim
        self.pos_emb_dim = self.word_dim * 2
        self.pos_emb = nn.Embedding(len(self.pos_dict), self.pos_emb_dim)
        self.word_lstm = BoundaryLSTM(input_dim, self.word_dim, 1, self.dropout)


        # stack hidden
        self.stack_dim = self.hidden_dim
        self.stack_lstm = BoundaryLSTM(self.word_dim, self.stack_dim, 1, self.dropout)

        self.relation_emb_dim = self.word_dim // 2
        self.relation_emb = nn.Embedding(len(self.relation_dict), self.relation_emb_dim)
        # compose head and modifier

        self.dependency_compsition = Composition(self.word_dim, self.relation_emb_dim)

        # transition id to embedding
        self.transition_emb_dim = self.hidden_dim // 4
        self.transition_emb = nn.Embedding(len(self.transition_dict), self.transition_emb_dim)
        # transition hidden
        self.transition_lstm_dim = self.hidden_dim
        self.transition_lstm = BoundaryLSTM(self.transition_emb_dim, self.transition_lstm_dim, 1, self.dropout)
        #self.transition_lstm = BoundaryLSTM(len(self.transition_dict), self.transition_dim, self.dropout)

        #self.encode = QRNN(input_dim, feature_dim, 1,
        #                    window_sizes=3, dropout=dropout)
        self.transition_classifier = TransitionClassifier(self.buffer_dim + self.stack_dim * 2 + self.transition_lstm_dim,
                                                          len(self.transition_dict), self.dropout)

    def _buffer(self, sentences):
        if isinstance(sentences, PackedSequence):
            buffers, lengths = pad_packed_sequence(sentences, batch_first=False)
        else:
            raise Exception("buffers must be PackedSequence")

        buffers = [list(torch.split(b.squeeze(1)[:length], 1, 0))
                   for b, length in zip(torch.split(buffers, 1, 1), lengths)]

        buffers = [list(reversed(b)) for b in buffers]

        buffer_hiddens = [[self.buffer_lstm.begin()] for b in buffers]
        max_length = lengths[0]
        for t in range(max_length):
            indexes = [i for i, _ in enumerate(buffers) if len(buffers[i]) > t]
            inputs = [buffers[i][t] for i in indexes]
            hidden = [buffer_hiddens[i][-1] for i in indexes]
            new_hiddens = self.buffer_lstm(inputs, hidden)

            for i, new in zip(indexes, new_hiddens):
                buffer_hiddens[i].append(new)

        return buffers, buffer_hiddens, lengths

    def _update_state(self, state_t, transition_id_t, transition_log_probs):
        """
        :param state_t: [state] * batch_size
        :param transition_id_t: LongTensor(batch_size)
        :param scores: [float] * batch_size
        :return:
        """
        def word_emb(chars):
            return chars[-1][:, 0:self.word_dim]

        transition_t = self.transition_dict.get_word(transition_id_t.data)
        # update char rnn
        update_nodes, update_char_inputs, update_pos, update_char_hiddens = [], [], [], []
        for state, t in zip(state_t, transition_t):
            if t.action == Actions.SHIFT:
                next_char = state.buffer.pop()
                state.buffer_hidden.pop()
                pos_id = torch.LongTensor([self.pos_dict.convert(t.label)])
                if self.use_cuda:
                    pos_id = pos_id.cuda()
                pos_emb = self.pos_emb(pos_id)
                node = Node(len(state.nodes), [pos_emb], t.label)
                state.nodes.append(node)

                update_nodes.append(node)
                update_char_inputs.append(next_char)
                update_char_hiddens.append(node.chars[-1])

            elif t.action == Actions.APPEND:
                next_char = state.buffer.pop()
                state.buffer_hidden.pop()
                node = state.nodes[-1]

                update_nodes.append(node)
                update_char_inputs.append(next_char)
                update_char_hiddens.append(node.chars[-1])

        if len(update_nodes) > 0:
            new_char_hidden = self.word_lstm(update_char_inputs, update_char_hiddens)
            for node, new in zip(update_nodes, new_char_hidden):
                node.chars.append(new)

        need_comp_states, need_comp_heads, need_comp_modifiers, need_comp_relations = [], [], [], []
        for state, t, tid, log_prob in zip(state_t, transition_t, transition_id_t.data, transition_log_probs.data):
            state.total_score += log_prob[tid]
            state.scores.append(log_prob[tid])
            if t.action == Actions.SHIFT:
                state.stack.append(word_emb(state.nodes[-1].chars))
            elif t.action == Actions.APPEND:
                state.stack.pop()
                state.stack_hidden.pop()
                state.stack.append(word_emb(state.nodes[-1].chars))
            elif t.action == Actions.ARC_LEFT:
                head = state.stack.pop()
                modifier = state.stack.pop()
                head_hidden = state.stack_hidden.pop()
                modifier_hidden = state.stack_hidden.pop()
                need_comp_states.append(state)
                need_comp_heads.append(head)
                need_comp_modifiers.append(modifier)
                need_comp_relations.append(t.label)
            elif t.action == Actions.ARC_RIGHT:
                modifier = state.stack.pop()
                head = state.stack.pop()
                modifier_hidden = state.stack_hidden.pop()
                head_hidden = state.stack_hidden.pop()
                need_comp_states.append(state)
                need_comp_heads.append(head)
                need_comp_modifiers.append(modifier)
                need_comp_relations.append(t.label)

        # update composition node
        if len(need_comp_states) > 0:
            need_comp_relations = torch.LongTensor(self.relation_dict.convert(need_comp_relations))
            if self.use_cuda:
                need_comp_relations = need_comp_relations.cuda()

            relation_emb = self.relation_emb(need_comp_relations)
            new_heads = self.dependency_compsition(need_comp_heads, need_comp_modifiers, relation_emb)
            for state, new in zip(need_comp_states, new_heads):
                state.stack.append(new)

        # update stack hidden
        new_stack_hiddens = self.stack_lstm([state.stack[-1] for state in state_t],
                                            [state.stack_hidden[-1] for state in state_t])

        for state, stack_hidden in zip(state_t, new_stack_hiddens):
            state.stack_hidden.append(stack_hidden)

        # update transition hidden
        no_teacher_forcing = True if self.training and random.random() < self.teacher_forcing_ratio else False
        if no_teacher_forcing:
            _, transition_id_t = transition_log_probs.max(1)
        transition_embs = self.transition_emb(transition_id_t)
        new_transition_hiddens = self.transition_lstm(transition_embs,
                                                      [state.transitions_hidden[-1] for state in state_t])

        for state, transition, transition_hidden in zip(state_t, transition_t, new_transition_hiddens):
            state.transitions.append(transition)
            state.transitions_hidden.append(transition_hidden)

        return state_t

    def loss(self, sentences, gold_transitions):
        '''
        :param sentences: [length, batch, dim]
        :param gold_transitions: [transition] * batch_size
        :return:
        '''

        #sentences, _ = self.encode(sentences)

        buffers, buffer_hiddens, lengths = self._buffer(sentences)

        batch_size = len(lengths)
        # [batch_size * [stack_size * [node]]]
        stacks = [[]] * batch_size

        states = [State([], buffer, buffer_hidden,
                        [], [self.stack_lstm.begin(), self.stack_lstm.begin()],
                        [], [self.transition_lstm.begin()])
                  for buffer, buffer_hidden, stack in zip(buffers, buffer_hiddens, stacks)]

        if gold_transitions is None:
            max_transition_length = lengths[0] * 2 - 1
        else:
            max_transition_length = gold_transitions.size(0)

        transition_loss = torch.FloatTensor([0])
        if self.use_cuda:
            transition_loss = transition_loss.cuda()

        transition_correct = 0
        transition_count = 1e-5

        pos_loss = [0]
        pos_correct = 0
        pos_count = 1e-5

        seg_loss = []
        seg_correct = 0
        seg_count = 1e-5

        for t in range(max_transition_length):
            state_t, transition_id_t, mask_t = [], [], []
            if gold_transitions is None:
                for state in states:
                    if len(state.buffer) > 1 or len(state.stack) > 1:
                        state_t.append(state)
                        mask_t.append(self.mask(len(state.buffer), len(state.stack),
                                                state.transitions[-1] if len(state.transitions) > 0 else None))

            else:

                for state, transition_id in zip(states, gold_transitions[t].data):
                    if transition_id >= 0:
                        state_t.append(state)
                        transition_id_t.append(transition_id)
                        transition = self.transition_dict.get_word(transition_id)
                        #assert self.check(state.buffer, state.stack, state.transitions, transition)
                        mask_t.append(self.mask(len(state.buffer), len(state.stack),
                                                state.transitions[-1] if len(state.transitions) > 0 else None))
                transition_id_t = torch.LongTensor(transition_id_t)
                if self.use_cuda:
                    transition_id_t = transition_id_t.cuda()

            if len(state_t) == 0:
                break

            transition_mask = torch.stack(mask_t)
            transition_log_prob = self.transition_classifier([state.buffer_hidden for state in state_t],
                                                         [state.stack_hidden for state in state_t],
                                                         [state.transitions_hidden for state in state_t],
                                                         transition_mask)

            # caculate loss
            if gold_transitions is None:
                _, transition_argmax = transition_log_prob.max(1)
                transition_loss += F.nll_loss(transition_log_prob,
                                              transition_argmax,
                                              size_average=False)
            else:
                transition_loss += F.nll_loss(transition_log_prob,
                                                   transition_id_t,
                                                   size_average=False)

            transition_count += transition_id_t.data.nelement()

            self._update_state(state_t, transition_id_t, transition_log_prob)
        return (transition_loss, transition_correct, transition_count), (sum(pos_loss)/pos_count, pos_correct, pos_count)

    def mask(self, buffer_len, stack_len, last_transition):
        action_blacklist = set()

        if buffer_len == 0:
            action_blacklist.add(Actions.SHIFT)
            action_blacklist.add(Actions.APPEND)

        if stack_len == 0:
            action_blacklist.add(Actions.APPEND)

        if last_transition is None:
            action_blacklist.update([Actions.APPEND, Actions.ARC_LEFT, Actions.ARC_RIGHT])
        elif last_transition.action == Actions.ARC_LEFT or last_transition.action == Actions.ARC_RIGHT:
            action_blacklist.add(Actions.APPEND)

        if stack_len < 2:
            action_blacklist.add(Actions.ARC_LEFT)
            action_blacklist.add(Actions.ARC_RIGHT)

        def to_mask(t):
            if t.action in action_blacklist:
                return 0
            return 1

        masked = torch.ByteTensor([to_mask(t) for t in self.transition_dict.words])
        if self.use_cuda:
            masked = masked.cuda()
        return masked

    @staticmethod
    def check(buffer, stack, prev_transitions, curr_transition):

        if len(prev_transitions) == 0:
            if curr_transition.action != Actions.SHIFT:
                return False
        else:
            '''
            if prev_transitions[-1].action in [Actions.SHIFT, Actions.APPEND] \
                    and curr_transition.action == Actions.APPEND \
                    and prev_transitions[-1].label != curr_transition.label:
                return False
            '''

            prev_action = prev_transitions[-1].action
            curr_action = curr_transition.action

            if curr_action == Actions.SHIFT:
                if len(buffer) == 0:
                    return False
            elif curr_action == Actions.APPEND:
                if prev_action != Actions.SHIFT and prev_action != Actions.APPEND:
                    return False
                if len(buffer) == 0 or len(stack) == 0:
                    return False

            elif curr_action == Actions.ARC_LEFT or curr_action == Actions.ARC_RIGHT:
                if len(stack) < 2:
                    return False
        return True

    def parse(self, sentences, beam_size=10, append_scale_ratio=1.0):

        #sentences, _ = self.encode(sentences)

        buffers, buffer_hiddens, lengths = self._buffer(sentences)

        result = [self.parse_one(buffer[0:length], buffer_hidden, beam_size, append_scale_ratio)
                for buffer, buffer_hidden, length in zip(buffers, buffer_hiddens, lengths)]

    def parse_one(self, sentence, buffer_hidden, beam_size=10, append_scale_ratio=1.0):

        length = len(sentence) * 2 - 1

        topK = []
        topK.append(State([], sentence, buffer_hidden,
                          [], [self.stack_lstm.begin(), self.stack_lstm.begin()],
                          [], [self.transition_lstm.begin()]))

        next1 = []

        for step in range(length):
            transition_mask = torch.stack([self.mask(len(state.buffer),
                                                   len(state.stack),
                                                   state.transitions[-1] if len(state.transitions) else None) for state in topK],
                                        0)
            transition_log_prob = self.transition_classifier([state.buffer_hidden for state in topK],
                                                         [state.stack_hidden for state in topK],
                                                         [state.transitions_hidden for state in topK],
                                                         transition_mask)

            step1, step2 = [], []
            for state, log_probs in zip(topK, transition_log_prob.data):
                for transition_id, log_prob in enumerate(log_probs):
                    transition = self.transition_dict.get_word(transition_id)
                    if self.check(state.buffer, state.stack, state.transitions, transition):
                        if transition.action in [Actions.SHIFT, Actions.ARC_LEFT, Actions.ARC_RIGHT]:
                            step1.append((state, transition, log_probs, state.score+log_prob))
                        else:
                            # append = shift + reduce
                            step2.append((state, transition, log_probs, state.score+log_prob))

            sorted_cands = sorted(step1 + next1, key=lambda c: c[-1], reverse=True)

            topK = sorted_cands[0: beam_size]

            transition_id_t = torch.LongTensor(self.transition_dict.convert([transition for _, transition, _, _ in topK]))
            if self.use_cuda:
                transition_id_t = transition_id_t.cuda()

            transition_log_prob = torch.stack([log_prob for _, _, log_prob, _ in topK])

            topK = self._update_state([copy.deepcopy(state) for state, _, _, _ in topK],
                                      transition_id_t,
                                      transition_log_prob)
            next1 = step2

        return topK
