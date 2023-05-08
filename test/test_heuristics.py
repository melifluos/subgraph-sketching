"""
testing the simple heuristics Personalized PageRank, Adamic Adar and Common Neighbours

This is the directed test graph
   -> 0 -> 1 <-
   |          |
    --- 2 <---
"""

import unittest
import math

import torch
from torch import tensor
import scipy.sparse as ssp
import numpy as np

from src.heuristics import AA, PPR, CN, RA


class HeuristicTests(unittest.TestCase):
    def setUp(self):
        self.edge_index = tensor([[0, 2, 2, 1], [1, 0, 1, 2]]).t()
        self.edge_weight = torch.ones(self.edge_index.size(0), dtype=torch.int)
        self.test_edges = tensor([[0, 1], [1, 2]]).t()
        self.num_nodes = 3
        self.neg_test_edges = tensor([[0, 1], [2, 0]]).t()
        self.A = ssp.csr_matrix((self.edge_weight, (self.edge_index[:, 0], self.edge_index[:, 1])),
                                shape=(self.num_nodes, self.num_nodes), dtype=float)
        # create a graph with 2 isomorphic nodes 2 & 3
        self.iso_edge_index = tensor([[2, 2, 3, 3, 4, 0], [1, 4, 1, 4, 0, 1]]).t()
        self.iso_edge_weight = torch.ones(self.iso_edge_index.size(0), dtype=int)
        self.iso_test_edges = tensor([[2, 3], [0, 0]]).t()
        self.iso_num_nodes = 5

        square1 = np.array([[0, 1, 1, 2, 2, 3, 3, 0],
                            [1, 0, 2, 1, 3, 2, 0, 3]])
        square2 = square1 + 4
        bridge = np.array([[0, 4],
                           [4, 0]])

        common_neigbours = np.array([[0, 9, 9, 4, 0, 8, 4, 8],
                                     [9, 0, 4, 9, 8, 0, 8, 4]])

        self.ei = torch.tensor(np.concatenate([square1, square2, bridge, common_neigbours], axis=1), dtype=torch.long)
        ew = torch.ones(self.ei.size(1), dtype=int)
        num_nodes = 10
        self.A1 = ssp.csr_matrix((ew, (self.ei[0], self.ei[1])), shape=(num_nodes, num_nodes))

    def test_CN(self):
        train_scores, edge_index = CN(self.A, self.edge_index)
        self.assertTrue(np.array_equal(train_scores, np.array([0, 1, 0, 0])))
        neg_scores, edge_index = CN(self.A, self.neg_test_edges)
        self.assertTrue(np.array_equal(neg_scores, np.array([1, 0])))
        pos_scores, edge_index = CN(self.A, self.test_edges)
        self.assertTrue(np.array_equal(pos_scores, np.array([0, 0])))

    def test_AA(self):
        train_scores, edge_index = AA(self.A, self.edge_index)
        print(train_scores)
        self.assertTrue(np.allclose(train_scores, np.array([0, 1 / math.log(2), 0, 0])))
        neg_scores, edge_index = AA(self.A, self.neg_test_edges)
        self.assertTrue(np.allclose(neg_scores, np.array([1 / math.log(2), 0])))
        pos_scores, edge_index = AA(self.A, self.test_edges)
        self.assertTrue(np.allclose(pos_scores, np.array([0, 0])))

    def test_RA(self):
        train_scores, edge_index = RA(self.A, self.edge_index)
        print(train_scores)
        self.assertTrue(np.allclose(train_scores, np.array([0, 1 / 2, 0, 0])))
        neg_scores, edge_index = RA(self.A, self.neg_test_edges)
        self.assertTrue(np.allclose(neg_scores, np.array([1 / 2, 0])))
        pos_scores, edge_index = RA(self.A, self.test_edges)
        self.assertTrue(np.allclose(pos_scores, np.array([0, 0])))

    def test_iso_graph(self):
        A = ssp.csr_matrix((self.iso_edge_weight, (self.iso_edge_index[:, 0], self.iso_edge_index[:, 1])),
                           shape=(self.iso_num_nodes, self.iso_num_nodes))
        aa_test_scores, edge_index = AA(A, self.iso_test_edges)
        print(aa_test_scores)
        self.assertTrue(aa_test_scores[0] == aa_test_scores[1])
        cn_test_scores, edge_index = CN(A, self.iso_test_edges)
        print(cn_test_scores)
        self.assertTrue(cn_test_scores[0] == cn_test_scores[1])
        ppr_test_scores, edge_index = PPR(A, self.iso_test_edges)
        print(ppr_test_scores)
        self.assertTrue(ppr_test_scores[0] == ppr_test_scores[1])
