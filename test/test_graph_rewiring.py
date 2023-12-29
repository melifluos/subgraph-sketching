"""
testing grape, which is still changing, so keeping this light
"""
import unittest
from argparse import Namespace

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils.random import barabasi_albert_graph
from torch_geometric.utils import add_self_loops, to_undirected
from embiggen.embedders.ensmallen_embedders.hyper_sketching import HyperSketching
from grape import Graph
import pandas as pd
import numpy as np
from torch import tensor
import scipy.sparse as ssp
import networkx as nx
from torch_geometric.utils import to_undirected, is_undirected, from_networkx
from src.graph_rewiring import EdgeBuilder
from test_params import OPT

class GraphRewiringTests(unittest.TestCase):
    def setUp(self):
        self.n_nodes = 30
        self.n_edges = 33  # deliberately odd to ensure undirected edges are handled correctly
        degree = 5  # number of edges to attach to each new node, not the degree at the end of the process
        self.x = torch.rand((self.n_nodes, 2))
        edge_index = barabasi_albert_graph(self.n_nodes, degree)
        indices = torch.randperm(edge_index.shape[1])[:self.n_edges]
        self.pos_edges = edge_index[:, indices].T
        # self.pos_edges = to_undirected(pos_edges)
        self.edge_index = to_undirected(edge_index)
        edge_weight = torch.ones(self.edge_index.size(1), dtype=torch.int)
        self.A = ssp.csr_matrix((edge_weight, (self.edge_index[0], self.edge_index[1])),
                                shape=(self.n_nodes, self.n_nodes))
        self.neg_edges = torch.randint(self.n_nodes, (self.n_edges, 2))
        self.links = torch.cat([self.pos_edges, self.neg_edges])
        self.labels = torch.cat([torch.ones(self.n_edges), torch.zeros(self.n_edges)])
        self.args = Namespace(**OPT)

    def test_EdgeBuilder(self):
        eb = EdgeBuilder(self.args)
        new_edge_index, _ = eb.rewire_graph(self.links, self.labels, self.edge_index)
        self.assertTrue(torch.all(torch.eq(new_edge_index, self.edge_index)))


    def test_remove_target_links(self):
        self.args.remove_target_links = True
        eb = EdgeBuilder(self.args)
        new_edge_index, _ = eb.rewire_graph(self.links, self.labels, self.edge_index)
        n_new_edges = new_edge_index.shape[1]
        n_edges = self.edge_index.shape[1]
        # the factor of two is because the graph is undirected
        print(n_new_edges, n_edges, self.n_edges, self.args.edge_dropout)
        self.assertTrue(n_new_edges == n_edges - 2 * int(self.n_edges * self.args.edge_dropout))

    def test_add_negative_links(self):
        self.args.add_negative_links = True
        eb = EdgeBuilder(self.args)
        new_edge_index, _ = eb.rewire_graph(self.links, self.labels, self.edge_index)

    def test_drop_message_passing_links(self):
        self.args.drop_message_passing_links = True
        eb = EdgeBuilder(self.args)
        new_edge_index, _ = eb.rewire_graph(self.links, self.labels, self.edge_index)

# this is undirected
