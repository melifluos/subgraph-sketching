"""
testing data reader and preprocessing
"""
import unittest
import os
from argparse import Namespace

import torch
from torch import tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, is_undirected
from ogb.linkproppred import PygLinkPropPredDataset

from src.data import get_data, get_pos_neg_edges, get_ogb_train_negs, make_obg_supervision_edges, get_ogb_data
from src.utils import ROOT_DIR
from test_params import OPT

class DataTests(unittest.TestCase):
  def setUp(self):
    self.edge_index = tensor([[0, 2, 2, 1], [1, 0, 1, 2]]).t()
    self.edge_weight = torch.ones(self.edge_index.size(0), dtype=int)
    self.test_edges = tensor([[0, 1], [1, 2]]).t()
    self.num_nodes = 3
    self.neg_test_edges = tensor([[0, 1], [2, 0]]).t()
    # create a graph with 2 isomorphic nodes 2 & 3
    self.iso_edge_index = tensor([[2, 2, 3, 3, 4, 0], [1, 4, 1, 4, 0, 1]]).t()
    self.iso_edge_weight = torch.ones(self.iso_edge_index.size(0), dtype=int)
    self.iso_test_edges = tensor([[2, 3], [0, 0]]).t()
    self.iso_num_nodes = 5

  def test_get_data(self):
    """
    We use the pyg RandomLinkSplit object to create train / val / test splits. For link prediction edges play 2 roles
    1/ for message passing 2/ as supervision
    :return:
    """
    opt = {'sample_size': None, 'dataset_name': 'Cora', 'num_hops': 2, 'max_dist': 10, 'max_nodes_per_hop': 10,
           'data_appendix': None, 'val_pct': 0.1, 'test_pct': 0.2, 'train_sample': 1, 'dynamic_train': True,
           'dynamic_val': True, 'model': 'linear', 'dynamic_test': True, 'node_label': 'drnl', 'ratio_per_hop': 1}
    opt = {**OPT, **opt}
    args = Namespace(**opt)
    dataset, splits, directed, eval_metric = get_data(args)
    train, val, test = splits['train'], splits['valid'], splits['test']
    train_pos_edges, train_neg_edges = get_pos_neg_edges(train)
    # the default behaviour is 1 negative edge for each positive edge
    self.assertTrue(train_pos_edges.shape == train_neg_edges.shape)
    val_pos_edges, val_neg_edges = get_pos_neg_edges(val)
    self.assertTrue(val_pos_edges.shape == val_neg_edges.shape)
    test_pos_edges, test_neg_edges = get_pos_neg_edges(test)
    self.assertTrue(test_pos_edges.shape == test_neg_edges.shape)