#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from typing import List
import torch
from torch import nn

class Edge:
    def __init__(self, begin: int, end: int, token):
        self.begin = begin
        self.end = end
        self.token = token

class Node:
    def __init__(self, inputs: List, outputs: List):
        self.inputs = inputs
        self.outputs = outputs


class Graph:
    def __init__(self, nodes: List[Node], edges: List[Edge]):
        self.nodes = nodes
        self.edges - edges



class Merge(nn.Module):
    def __init__(self, hidden_size):
        super(Merge, self).__init__()


class GraphLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GraphLSTM, self).__init__()

        self.node2edge = nn.LSTMCell(input_size, hidden_size)

        self.edge2node =