#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import math
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import itertools

import torch
from torch.nn import Module, Parameter, init, Embedding, ModuleList, Dropout

class Node:
    def __init__(self, inputs: List[int], outputs: List[int], **kwargs):
        self.inputs = inputs
        self.outputs = outputs
        self.__dict__.update(**kwargs)


class Edge:
    def __init__(self, begin: int, end: int, **kwargs):
        self.begin = begin
        self.end = end
        self.attrs = dict(kwargs)


class DirectLattice:
    def __init__(self, edges, nexts: Dict[int, List[int]], prevs: Dict[int, List[int]]):
        self.edges = edges
        self.nexts = nexts
        self.prevs = prevs

    def seq_len(self):
        return len(self.nexts)

    def edge_len(self):
        return len(self.edges)

class Lattice:
    def __init__(self, length: int, edges: List[Edge]):
        self.length = length
        self.edges = edges

    def left2right(self):

        nexts = defaultdict(lambda: [])
        prevs = defaultdict(lambda: [])
        for edge_id, edge in enumerate(self.edges):
            nexts[edge.begin].append(edge_id)
            prevs[edge.end-1].append(edge_id)

        return DirectLattice(self.edges, nexts, prevs)

    def right2left(self):

        nexts = defaultdict(lambda: [])
        prevs = defaultdict(lambda: [])
        for edge_id, edge in enumerate(self.edges):
            nexts[self.length - edge.end].append(edge_id)
            prevs[self.length - edge.begin - 1].append(edge_id)

        return DirectLattice(self.edges, nexts, prevs)


class EdgeCell(Module):
    def __init__(self, input_size, hidden_size, use_bias=True):
        super(EdgeCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias

        self.weight_ih = Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, 4 * hidden_size))

        if self.use_bias:
            self.bias = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_buffer('bias', None)

    def forward(self, node: Tuple[torch.Tensor, torch.Tensor], edge: Tuple[torch.Tensor, List[int]]):

        node_h, node_c = node
        edge_i, edge_sizes = edge

        assert node_h.size() == node_c.size()
        assert node_h.size(0) == len(edge_sizes)
        assert sum(edge_sizes) == edge_i.size(0)

        edge_ih = torch.matmul(edge_i, self.weight_ih)
        node_hh = torch.matmul(node_h, self.weight_hh)

        if self.use_bias:
            node_hh = node_hh + self.bias

        edge_h = []
        edge_c0 = []

        for bid, (ih, hh, c0) in enumerate(
                zip(torch.split(edge_ih, edge_sizes, dim=0),
                    torch.unbind(node_hh, dim=0),
                    torch.unbind(node_c, dim=0))):
            edge_h.append(ih + hh)
            edge_c0.append(c0.unsqueeze(0).expand(ih.size(0), -1))

        i, f, g, o = torch.split(torch.cat(edge_h, dim=0), self.hidden_size, dim=-1)

        c = f.sigmoid() * torch.cat(edge_c0, dim=0) + i.sigmoid() * g.tanh()
        h = o.sigmoid() * o.tanh()

        return h, c


class LatticeLSTMCell(Module):
    def __init__(self, input_dim, hidden_dim, use_bias=True):
        super(LatticeLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.use_bias = use_bias

        self.edge_cell = EdgeCell(input_dim, hidden_dim)

    def forward(self,
                lattices: List[DirectLattice],
                edge_input: List[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        '''

        :param lattices:  sort descending by sequence length
        :return:
        '''
        max_seq_length = lattices[0].seq_len()
        node_h = [torch.zeros(lattice.seq_len(), self.input_dim) for lattice in lattices]
        node_c = [torch.zeros(lattice.seq_len(), self.input_dim) for lattice in lattices]
        edge_h = [torch.zeros(lattice.edge_len(), self.hidden_dim) for lattice in lattices]
        edge_c = [torch.zeros(lattice.edge_len(), self.hidden_dim) for lattice in lattices]
        for step in range(max_seq_length):
            # next
            batch_h, batch_c = [], []
            batch_input, batch_input_sizes = [], []
            batch_str = []
            for seq_id, lattice in enumerate(lattices):
                if lattice.seq_len() < step:
                    break
                if step > 0:
                    batch_h.append(node_h[seq_id][step])
                    batch_c.append(node_c[seq_id][step])
                else:
                    batch_h.append(torch.zeros(self.hidden_dim))
                    batch_c.append(torch.zeros(self.hidden_dim))
                next_edge = [edge_input[seq_id][edge_id] for edge_id in lattice.nexts[step]]
                batch_input.extend(next_edge)
                batch_input_sizes.append(len(next_edge))
                batch_str = [lattice.edges[edge_id].attrs['token'] for edge_id in lattice.nexts[step]]
            print('next', step, batch_str)
            batch_h = torch.stack(batch_h, dim=0)
            batch_c = torch.stack(batch_c, dim=0)
            batch_input = torch.stack(batch_input, dim=0)

            next_edge_h, next_edge_c = self.edge_cell(
                (batch_h, batch_c),
                (batch_input, batch_input_sizes))

            for seq_id, (_edge_h, _edge_c) in enumerate(zip(next_edge_h.split(batch_input_sizes, dim=0),
                                                          next_edge_c.split(batch_input_sizes, dim=0))):
                for offset, edge_id in enumerate(lattices[seq_id].nexts[step]):
                    edge_h[seq_id][edge_id] = _edge_h[offset]
                    edge_c[seq_id][edge_id] = _edge_c[offset]

            # aggregate by average
            for seq_id, lattice in enumerate(lattices):
                if step < lattice.seq_len():
                    print('agg', step, [lattice.edges[prev].attrs['token'] for prev in lattice.prevs[step]])
                    node_h[seq_id][step] = torch.stack(
                        [edge_h[seq_id][prev] for prev in lattice.prevs[step]], dim=0).mean(dim=0)
                    node_c[seq_id][step] = torch.stack(
                        [edge_c[seq_id][prev] for prev in lattice.prevs[step]], dim=0).mean(dim=0)

        return list(zip(node_h, node_c, edge_h, edge_c))


class LatticeLSTM(Module):
    def __init__(self, input_dim, hidden_dim, num_layer, dropout=0.3):
        super(LatticeLSTM, self).__init__()

        self.num_layer = num_layer
        self.dropout = Dropout(dropout)
        self.forward_cells = ModuleList(
            LatticeLSTMCell(input_dim if layer_id == 0 else hidden_dim, hidden_dim//2)
            for layer_id in range(num_layer))
        self.backward_cells = ModuleList(
            LatticeLSTMCell(input_dim if layer_id == 0 else hidden_dim, hidden_dim//2)
            for layer_id in range(num_layer))

    def forward(self, lattices: List[Lattice], edge_input: List[torch.Tensor]) -> List[torch.Tensor]:

        left2rights = [lattice.left2right() for lattice in lattices]
        right2lefts = [lattice.right2left() for lattice in lattices]
        input = edge_input

        for layer_id in range(self.num_layer):
            f_h = self.forward_cells[layer_id](left2rights, input)
            b_h = self.backward_cells[layer_id](right2lefts, input)

            if layer_id + 1 == self.num_layer:
                return [torch.cat((f_n_h, b_n_h.flip([0])), dim=-1)
                        for (f_n_h, f_n_c, f_e_h, f_e_c), (b_n_h, b_n_c, b_e_h, b_e_c) in zip(f_h, b_h)]
            else:
                input = [torch.cat((f_e_h, b_e_h.flip([0])), dim=-1)
                         for (f_n_h, f_n_c, f_e_h, f_e_c), (b_n_h, b_n_c, b_e_h, b_e_c) in zip(f_h, b_h)]


class LatticeEncoder(Module):
    def __init__(self,
                 num_vocab: int, embedding_dim:int, hidden_dim: int, num_layer: int, dropout = 0.3):
        super(LatticeEncoder, self).__init__()
        self.embedding = Embedding(num_vocab, embedding_dim)

        self.lstm = LatticeLSTM(embedding_dim, hidden_dim, num_layer, dropout=dropout)

    def forward(self, lattices: List[Lattice]) -> List[torch.Tensor]:

        edge_emb = [self.embedding(torch.tensor([edge.attrs['id'] for edge in lattice.edges], dtype=torch.Long))
                    for lattice in lattices]

        hidden = self.lstm(lattices, edge_emb)

        return hidden
