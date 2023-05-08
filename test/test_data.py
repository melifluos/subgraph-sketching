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

from src.data import get_data, get_ogb_train_negs, make_obg_supervision_edges, get_ogb_data, get_loaders
from src.utils import ROOT_DIR, get_pos_neg_edges
from test_params import OPT


class DataTests(unittest.TestCase):
    def setUp(self):
        self.edge_index = tensor([[0, 2, 2, 1], [1, 0, 1, 2]]).t()
        self.edge_weight = torch.ones(self.edge_index.size(0), dtype=int)
        self.test_edges = tensor([[0, 1], [1, 2]]).t()
        self.num_nodes = 3
        self.neg_test_edges = tensor([[0, 1], [2, 0]]).t()
        edges = 8
        self.train = {'edge': torch.randint(0, edges, size=(edges, 2))}
        self.valid = {'edge': torch.randint(0, edges, size=(edges, 2)), 'edge_neg': torch.randint(0, edges, size=(edges, 2))}
        self.test = {'edge': torch.randint(0, edges, size=(edges, 2)), 'edge_neg': torch.randint(0, edges, size=(edges, 2))}
        self.split_edge = {'train': self.train, 'valid': self.valid, 'test': self.test}

    def test_make_obg_supervision_edges(self):
        labels, edges = make_obg_supervision_edges(self.split_edge, 'test', neg_edges=None)
        n_pos, n_neg = self.test['edge'].shape[0], self.test['edge_neg'].shape[0]
        n_edges = self.train['edge'].shape[0] + self.valid['edge'].shape[0]
        self.assertTrue(labels.shape[0] == n_pos + n_neg)
        self.assertTrue(edges.shape == (2, n_edges))

    def test_get_ogb_data(self):
        ei = torch.randint(0, 10, size=(2, 20))
        ew = torch.ones(10)
        self.valid['weight'] = ew
        x = torch.rand(size=(10, 3))
        data = Data(x=x, edge_index=ei, edge_weight=ew.unsqueeze(-1))
        splits = get_ogb_data(data, self.split_edge, dataset_name='test_dataset')
        self.assertTrue('test' in splits.keys())
        self.assertTrue(len(splits.keys()) == 3)
        self.assertTrue(torch.all(torch.eq(splits['train'].edge_index, splits['valid'].edge_index)))
        self.assertTrue(torch.all(torch.eq(splits['test'].x, splits['valid'].x)))

    def test_use_val_edges(self):
        """
        to get good performance on ogb data, the val edges need to be used at test time. However, the val edges don't come
        undirected
        @return:
        """
        name = 'ogbl-collab'
        path = os.path.join(ROOT_DIR, 'dataset', name)
        dataset = PygLinkPropPredDataset(name=name, root=path)
        split_edge = dataset.get_edge_split()
        val_edges = split_edge['valid']['edge']
        self.assertTrue(not is_undirected(val_edges))  # no reason for supervision edges to be directed

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

    def test_get_ogb_train_negs(self):
        num_nodes = 10
        ei = torch.randint(0, 10, size=(2, 20))
        split_edge = self.split_edge
        train_negs = get_ogb_train_negs(split_edge, ei, num_nodes)
        self.assertTrue(train_negs.shape == split_edge['train']['edge'].shape)
        train_negs = get_ogb_train_negs(split_edge, ei, num_nodes, dataset_name='ogbl-citation2')
        # in this case the src node for the neg and pos edges should be the same
        self.assertTrue(torch.all(torch.eq(train_negs[:, 0], self.train['edge'][:, 0])))
        train_negs = get_ogb_train_negs(split_edge, ei, num_nodes, num_negs=2, dataset_name='ogbl-citation2')
        self.assertTrue(train_negs.shape[0] == 2 * split_edge['train']['edge'].shape[0])
        self.assertTrue(train_negs[0][0] == train_negs[1][0])

    def test_get_loaders(self):
        opt = {'sample_size': None, 'dataset_name': 'Cora', 'num_hops': 2, 'max_dist': 10, 'max_nodes_per_hop': 10,
               'data_appendix': None, 'val_pct': 0.1, 'test_pct': 0.2, 'train_sample': 1, 'dynamic_train': True,
               'dynamic_val': True, 'model': 'linear', 'dynamic_test': True, 'node_label': 'drnl', 'ratio_per_hop': 1}
        opt = {**OPT, **opt}
        args = Namespace(**opt)
        dataset, splits, directed, eval_metric = get_data(args)
        train_loader, train_eval_loader, val_loader, test_loader = get_loaders(args, dataset, splits, directed)
        # todo finish writing this test
