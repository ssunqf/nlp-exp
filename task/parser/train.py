#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from collections import defaultdict

from .model import Actions, UDTree, ArcStandard, Transition

import torch
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pack_padded_sequence
from task.util.vocab import Vocab
from task.util.utils import replace_entity

class ParserConfig:

    def __init__(self, name, train_paths, test_paths):
        self.name = name
        self.train_paths = train_paths
        self.test_paths = test_paths

        self.hidden_dim = 128
        self.dropout = 0.2
        self.shared_weight_decay = 1e-6
        self.task_weight_decay = 1e-6

        self._loader = None

    def loader(self):
        if self._loader is None:
            self._loader = CTBParseData(self.train_paths, self.test_paths)

        return self._loader

    def create_task(self, shared_vocab, shared_encoder):

        loader = self.loader()

        return ParserTask(self.name,
                          shared_encoder, shared_vocab,
                          loader.transition_dict, loader.relation_dict, loader.pos_dict,
                          self.hidden_dim,
                          self.dropout,
                          self.shared_weight_decay,
                          self.task_weight_decay)


class ParserTask:
    def __init__(self, name, encoder, vocab,
                 transition_dict, relation_dict, pos_dict,
                 hidden_dim,
                 dropout,
                 shared_weight_decay,
                 task_weight_decay):

        self.name = name
        self.vocab = vocab
        self.transition_dict = transition_dict
        self.relation_dict = relation_dict
        self.pos_dict = pos_dict
        self.shared_encoder = encoder

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.shared_weight_decay = shared_weight_decay
        self.task_weight_decay = task_weight_decay

        self.use_cuda = False

        self.beam_size = 10
        self.append_scale_ratio = 1.0

        self.parser = ArcStandard(self.shared_encoder.output_dim(),
                                  self.hidden_dim,
                                  self.transition_dict,
                                  self.relation_dict,
                                  self.pos_dict,
                                  self.dropout)


        self.params = [{'params': self.shared_encoder.parameters(), 'weight_decay': self.shared_weight_decay},
                       #{'params': self.task_encoder.parameters(), 'weight_decay': task_weight_decay},
                       {'params': self.parser.parameters(), 'weight_decay': self.task_weight_decay}]

    def forward(self, sentences, transitions):
        sentences, gazetteers = sentences

        feature = self.shared_encoder(sentences, gazetteers)
        return self.parser.loss(feature, transitions)

    def _to_cuda(self, batch_data):

        (sentences, gazes), transitions = batch_data

        return ((PackedSequence(sentences.data.cuda(), sentences.batch_sizes),
                 PackedSequence(gazes.data.cuda(), gazes.batch_sizes)),
                transitions.cuda() if transitions else None)

    def loss(self, batch_data):

        if self.use_cuda:
            batch_data = self._to_cuda(batch_data)

        sentences, reference = batch_data
        (loss, acc, count), _ = self.forward(sentences, reference)

        return loss/count

    def parse(self, sentences):
        sentences, gazetteers = sentences

        features = self.shared_encoder(sentences, gazetteers)

        return self.parser.parse(features, self.beam_size, self.append_scale_ratio)

    def sample(self, batch_data):

        if self.use_cuda:
            batch_data = self._to_cuda(batch_data)

        sentences, gold_trans = batch_data
        topks = self.parse(sentences)

        sentences, lengths = pad_packed_sequence(sentences[0], batch_first=True, padding_value=-1)
        sentences = [self.vocab.get_word(sentence[0, :length].data.numpy())
                     for sentence, length in zip(sentences.split(1, 0), lengths)]

        if gold_trans is None:
            return ['score=%f\t%s\n' % (topk[0].total_score, UDTree.create(sentence, topk[0].transitions).to_line())
                    for sentence, topk in zip(sentences, topks)]
        else:
            gold_trans = [gold_trans[:, sen_id] for sen_id in range(len(sentences))]
            gold_trans = [t.masked_select(t >= 0) for t in gold_trans]

            samples = [(topk[0].total_score, UDTree.create(sentence, self.transition_dict.get_word(gold.data)).to_line(),
                        UDTree.create(sentence, topk[0].transitions).to_line())
                       for sentence, gold, topk in zip(sentences, gold_trans, topks)]

            return ['score=%f\nref:  %s\npred: %s\n' % (score, gold, pred) for score, gold, pred in samples if gold != pred]

    def evaluation(self, data):

        seg_correct = 0
        pos_correct = 0

        uas_correct = 0
        las_correct = 0

        gold_count = 0
        pred_count = 0

        labeled_arc_correct = 0
        unlabeled_arc_correct = 0


        for batch in data:
            if self.use_cuda:
                batch = self._to_cuda(batch)

            sentences, transitions = batch
            topks = self.parse(sentences)

            sentences, lengths = pad_packed_sequence(sentences[0], batch_first=True, padding_value=-1)
            sentences = [sentence[0, :length] for sentence, length in zip(sentences.split(1, 0), lengths)]
            for sen_id, (sentence, topk) in enumerate(zip(sentences, topks)):
                sen_gold_trans = transitions[:, sen_id]
                sen_gold_trans = sen_gold_trans.masked_select(sen_gold_trans >= 0)

                gold_tree = UDTree.create(self.vocab.get_word(sentence.data.numpy()), self.transition_dict.get_word(sen_gold_trans.data))
                pred_tree = UDTree.create(self.vocab.get_word(sentence.data.numpy()), topk[0].transitions)

                gold_nodes = gold_tree.nodes
                pred_nodes = pred_tree.nodes

                gold_count += len(gold_nodes)
                pred_count += len(pred_nodes)

                gold_char_index, gold_word_index = 0, 0
                pred_char_index, pred_word_index = 0, 0

                matched_nodes = []

                while gold_word_index < len(gold_nodes) and pred_word_index < len(pred_nodes):

                    if gold_nodes[gold_word_index].chars == pred_nodes[pred_word_index].chars:
                        seg_correct += 1
                        matched_nodes.append((gold_word_index, pred_word_index))
                        if gold_nodes[gold_word_index].pos == pred_nodes[pred_word_index].pos:
                            pos_correct += 1

                    gold_char_index += len(gold_nodes[gold_word_index])
                    pred_char_index += len(pred_nodes[pred_word_index])

                    gold_word_index += 1
                    pred_word_index += 1

                    while gold_char_index != pred_char_index:
                        if gold_char_index < pred_char_index:
                            gold_char_index += len(gold_nodes[gold_word_index])
                            gold_word_index += 1
                        elif gold_char_index > pred_char_index:
                            pred_char_index += len(pred_nodes[pred_word_index])
                            pred_word_index += 1


                pred2gold = {pred: gold for gold, pred in matched_nodes}

                for gold_modifier, pred_modifier in matched_nodes:
                    pred_relation = pred_nodes[pred_modifier].relation
                    pred_head = pred_nodes[pred_modifier].head_index

                    gold_head = gold_nodes[gold_modifier].head_index
                    gold_relation = gold_nodes[gold_modifier].relation

                    if (pred_head == gold_head == -1) or pred2gold.get(pred_head, -100) == gold_head:
                        unlabeled_arc_correct += 1
                        if gold_relation == pred_relation:
                            labeled_arc_correct += 1

        safe_div = lambda a, b : a / (b + 1e-10)
        seg_prec = safe_div(seg_correct, pred_count)
        seg_recall = safe_div(seg_correct, gold_count)
        seg_f = safe_div(2*seg_prec*seg_recall, seg_prec+seg_recall)

        pos_prec = safe_div(pos_correct, pred_count)
        pos_recall = safe_div(pos_correct, gold_count)
        pos_f = safe_div(2*pos_prec*pos_recall, pos_prec+pos_recall)

        unlabeled_prec = safe_div(unlabeled_arc_correct, pred_count)
        unlabeled_recall = safe_div(unlabeled_arc_correct, gold_count)
        unlabeled_f = safe_div(2*unlabeled_prec*unlabeled_recall, unlabeled_prec+unlabeled_recall)

        labeled_prec = safe_div(labeled_arc_correct, pred_count)
        labeled_recall = safe_div(labeled_arc_correct, gold_count)
        labeled_f = safe_div(2*labeled_prec*labeled_recall, labeled_prec+labeled_recall)


        return {'seg prec':seg_prec, 'seg recall':seg_recall, 'seg F-score':seg_f,
                'pos pred':pos_prec, 'pos recall':pos_recall, 'pos F-score':pos_f,
                'ua prec':unlabeled_prec, 'ua recall':unlabeled_recall, 'ua F-score':unlabeled_f,
                'la prec':labeled_prec, 'la recall': labeled_recall, 'la F-score':labeled_f
                }


class CTBParseData:
    def __init__(self, train_paths, test_paths):
        self.train_data = self.load(train_paths)

        self.word_counts = defaultdict(int)
        self.pos_counts = defaultdict(int)
        self.transition_counts = defaultdict(int)
        self.relation_counts = defaultdict(int)

        for chars, transitions in self.train_data:
            for t in transitions:
                self.transition_counts[t] += 1

                if t.action == Actions.SHIFT:
                    self.pos_counts[t.label] += 1
                elif t.action in [Actions.ARC_LEFT, Actions.ARC_RIGHT]:
                    self.relation_counts[t.label] += 1

                self.action_counts[t.action] += 1

            for char in chars:
                self.word_counts[char] += 1

        self.test_data = self.load(test_paths)

        self.train_data = sorted(self.train_data, key=lambda item: len(item[0]), reverse=True)
        self.test_data = sorted(self.test_data, key=lambda item: len(item[0]), reverse=True)

        self.min_count = 5

        transitions = defaultdict(int)

        def convert(t):
            if t.action == Actions.SHIFT:
                if self.pos_counts[t.label] > self.min_count:
                    return t
                else:
                    return Transition(t.action, 'UNK_POS')
            elif t.action in [Actions.ARC_LEFT, Actions.ARC_RIGHT]:
                if self.relation_counts[t.label] > self.min_count:
                    return t
                else:
                    return Transition(t.action, 'UNK_RELATION')
            else:
                return t

        for t, count in self.transition_counts.items():
            transitions[convert(t)] += count

        max_count = max(transitions.items(), key=lambda t:t[1])
        transition_weights = [(t, math.log(max_count/count)) for t, count in transitions.items]
        self.transition_dict = Vocab([t for t, _ in transition_weights], unk=None)
        self.pos_dict = Vocab([k for k, v in self.pos_counts.items() if v > self.min_count], unk='UNK_POS')

        relations = set([t.label for t, _ in transition_weights if t.action in [Actions.ARC_LEFT, Actions.ARC_RIGHT]])
        self.relation_dict = Vocab(relations, unk=None)

    def load(self, paths):
        data = []
        bad_data_count = 0
        for path in paths:
            with open(path) as file:
                sentences = file.read().strip().split('\n\n')
                # uniq operation
                #sentences = set(sentences)
                for sentence in sentences:
                    words = sentence.strip().split('\n')
                    if len(words) > 0:
                        words = [word.split('\t') for word in words]
                        words = [([char if type == '@zh_char@' else type for type, char in replace_entity(word)], pos, parent_id, relation)
                                 for _, word, _, pos, _, _, parent_id, relation, _, _ in words]
                        tree = UDTree.parse_stanford_format(words)

                        if tree is None:
                            bad_data_count += 1
                        else:
                            data.append(tree.linearize())

        print(bad_data_count)
        return data

    def _batch(self, data, vocab, gazetteers, batch_size):

        gazetteers_dim = sum([c.length() for c in gazetteers])

        for begin in range(0, len(data), batch_size):
            batch = data[begin:begin+batch_size]

            batch = sorted(batch, key=lambda item: len(item[0]), reverse=True)
            sen_lens = [len(s) for s, _ in batch]
            max_sen_len = max(sen_lens)
            max_tran_len = max([len(trans) for _, trans in batch])
            sentences = torch.LongTensor(max_sen_len, len(batch)).fill_(0)
            gazes = torch.FloatTensor(max_sen_len, len(batch), gazetteers_dim).fill_(0)
            transitions = torch.LongTensor(max_tran_len, len(batch)).fill_(-1)

            for id, (words, trans) in enumerate(batch):
                sen_len = len(words)
                sentences[:sen_len, id] = torch.LongTensor(vocab.convert(words))
                gazes[0:sen_len, id] = torch.cat([torch.FloatTensor(gazetteer.convert(words)) for gazetteer in gazetteers], -1)
                tran_len = len(trans)
                transitions[:tran_len, id] = torch.LongTensor(self.transition_dict.convert(trans))

            yield ((pack_padded_sequence(sentences, sen_lens),
                    pack_padded_sequence(gazes, sen_lens)),
                   transitions)

    def batch_train(self, vocab, gazetteers, batch_size):
        return self._batch(self.train_data, vocab, gazetteers, batch_size)

    def batch_test(self, vocab, gazetteers, batch_size):
        return self._batch(self.test_data, vocab, gazetteers, batch_size)


class RawData:
    def __init__(self, train_paths, test_paths):
        self.train_data = self.load(train_paths)

        self.word_counts = defaultdict(int)
        for chars in self.train_data:
            for char in chars:
                self.word_counts[char] += 1

        self.test_data = self.load(test_paths)

        self.train_data = sorted(self.train_data, key=lambda item: len(item), reverse=True)
        self.test_data = sorted(self.test_data, key=lambda item: len(item), reverse=True)

        self.num_pos = 10
        pos_list = ['pos-%d' % p for p in range(self.num_pos)]

        self.num_relation = 10
        relation_list = ['rel-%d' % r for r in range(self.num_relation)]
        transitions = list(itertools.chain.from_iterable([
            [Transition(Actions.SHIFT, p) for p in pos_list],
            [Transition(Actions.APPEND, None)],
            [Transition(Actions.ARC_LEFT, r) for r in relation_list],
            [Transition(Actions.ARC_RIGHT, r) for r in relation_list]
        ]))

        self.transition_dict = Vocab(transitions, unk=None)

        self.pos_dict = Vocab(pos_list, unk=None)

        self.relation_dict = Vocab(relation_list, unk=None)

    def load(self, paths):
        data = []
        bad_data_count = 0
        for path in paths:
            with open(path) as file:
                # uniq operation
                #sentences = set(sentences)
                for line in file:
                    words = line.strip().split('\n')
                    chars = [char if type == '@zh_char@' else type
                             for word in words for type, char in replace_entity(word)]
                    if len(chars) > 0:
                        data.append(chars)
        return data

    def _batch(self, data, vocab, gazetteers, batch_size):

        gazetteers_dim = sum([c.length() for c in gazetteers])

        for begin in range(0, len(data), batch_size):
            batch = data[begin:begin+batch_size]

            batch = sorted(batch, key=lambda item: len(item), reverse=True)
            sen_lens = [len(s) for s in batch]
            max_sen_len = max(sen_lens)
            sentences = torch.LongTensor(max_sen_len, len(batch)).fill_(0)
            gazes = torch.FloatTensor(max_sen_len, len(batch), gazetteers_dim).fill_(0)

            for id, words in enumerate(batch):
                sen_len = len(words)
                sentences[:sen_len, id] = torch.LongTensor(vocab.convert(words))
                gazes[0:sen_len, id] = torch.cat([torch.FloatTensor(gazetteer.convert(words)) for gazetteer in gazetteers], -1)

            yield ((pack_padded_sequence(sentences, sen_lens),
                    pack_padded_sequence(gazes, sen_lens)),
                   None)

    def batch_train(self, vocab, gazetteers, batch_size):
        return self._batch(self.train_data, vocab, gazetteers, batch_size)

    def batch_test(self, vocab, gazetteers, batch_size):
        return self._batch(self.test_data, vocab, gazetteers, batch_size)


class SelfParserConfig:

    def __init__(self, name, train_paths, test_paths):
        self.name = name
        self.train_paths = train_paths
        self.test_paths = test_paths

        self.hidden_dim = 128
        self.dropout = 0.2
        self.shared_weight_decay = 1e-6
        self.task_weight_decay = 1e-6

        self._loader = None

    def loader(self):
        if self._loader is None:
            self._loader = RawData(self.train_paths, self.test_paths)

        return self._loader

    def create_task(self, shared_vocab, shared_encoder):

        loader = self.loader()

        return ParserTask(self.name,
                          shared_encoder, shared_vocab,
                          loader.transition_dict, loader.relation_dict, loader.pos_dict,
                          self.hidden_dim,
                          self.dropout,
                          self.shared_weight_decay,
                          self.task_weight_decay)
