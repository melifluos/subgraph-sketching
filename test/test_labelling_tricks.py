import unittest

import torch
from torch import tensor
import scipy.sparse as ssp
import numpy as np

from src.labelling_tricks import de_plus_node_labeling, de_node_labeling, drnl_node_labeling, drnl_hash_function, \
    get_drnl_lookup


class LabelingTrickTests(unittest.TestCase):
    def setUp(self):
        num_node_features = 2
        # test graph is two connected squares
        square1 = np.array([[0, 1, 1, 2, 2, 3, 3, 0],
                            [1, 0, 2, 1, 3, 2, 0, 3]])
        square2 = square1 + 4
        bridge = np.array([[0, 4],
                           [4, 0]])
        self.edge_index = torch.tensor(np.concatenate([square1, square2, bridge], axis=1), dtype=torch.long)
        self.edge_set = {(0, 1), (2, 0), (2, 1), (1, 2)}
        self.target_edges = [(0, 1), (1, 2), (0, 2), (1, 0)]
        self.edge_weight = torch.ones(self.edge_index.size(1), dtype=int)
        self.test_edges = tensor([[0, 1], [1, 2]])
        self.num_nodes = 8
        self.A = ssp.csr_matrix((self.edge_weight, (self.edge_index[0], self.edge_index[1])),
                                shape=(self.num_nodes, self.num_nodes))
        # add one more edge connecting the squares
        bridge1 = torch.tensor([[5, 1], [1, 5]])
        ei = torch.cat([self.edge_index, bridge1], dim=1)
        ew = torch.ones(ei.size(1), dtype=int)
        self.A1 = ssp.csr_matrix((ew, (ei[0], ei[1])),
                                 shape=(self.num_nodes, self.num_nodes))
        # add one more edge connecting the squares
        bridge2 = torch.tensor([[1, 3, 5, 7], [3, 1, 7, 5]])
        ei1 = torch.cat([self.edge_index, bridge2], dim=1)
        ew1 = torch.ones(ei1.size(1), dtype=int)
        self.A2 = ssp.csr_matrix((ew1, (ei1[0], ei1[1])),
                                 shape=(self.num_nodes, self.num_nodes))

    def test_de_node_labeling(self):
        labels = de_node_labeling(self.A, 0, 4, max_dist=3)
        square1_truth = torch.tensor([[0, 1], [1, 2], [2, 3], [1, 2]])
        square2_truth = square1_truth.flip(dims=[1])
        truth = torch.cat([square1_truth, square2_truth], dim=0)
        self.assertTrue(torch.all(torch.eq(labels, truth)))
        labels = de_node_labeling(self.A1, 0, 4, max_dist=3)
        # nothing should change as DE does not mask the src / dst edges when calculating distances
        self.assertTrue(torch.all(torch.eq(labels, truth)))

    def test_drnl_node_labeling(self):
        labels = drnl_node_labeling(self.A, 0, 4, max_dist=10)
        truth = torch.tensor([1, 27, 33, 27, 1, 27, 33, 27])
        self.assertTrue(torch.all(torch.eq(labels, truth)))
        labels = drnl_node_labeling(self.A1, 0, 4, max_dist=10)
        truth1 = torch.tensor([1, 3, 7, 6, 1, 3, 7, 6])
        self.assertTrue(torch.all(torch.eq(labels, truth1)))
        labels = drnl_node_labeling(self.A2, 0, 4, max_dist=10)
        self.assertTrue(torch.all(torch.eq(labels, truth)))

    def test_de_plus_node_labeling(self):
        md = 10
        labels = de_plus_node_labeling(self.A, 0, 4, max_dist=md)
        square1_truth = torch.tensor([[0, 1], [1, md], [2, md], [1, md]])
        square2_truth = square1_truth.flip(dims=[1])
        truth = torch.cat([square1_truth, square2_truth], dim=0)
        self.assertTrue(torch.all(torch.eq(labels, truth)))
        labels = de_plus_node_labeling(self.A1, 0, 4, max_dist=md)
        square1_truth = torch.tensor([[0, 1], [1, 2], [2, 3], [1, 4]])
        square2_truth = square1_truth.flip(dims=[1])
        truth = torch.cat([square1_truth, square2_truth], dim=0)
        self.assertTrue(torch.all(torch.eq(labels, truth)))

    def test_drnl_hash_function(self):
        hash1 = drnl_hash_function(tensor([1]), tensor([1]))
        self.assertTrue(2 == hash1.item())
        hash2 = drnl_hash_function(tensor([1]), tensor([2]))
        hash3 = drnl_hash_function(tensor([2]), tensor([1]))
        self.assertTrue(hash2 == hash3)
        hash4 = drnl_hash_function(tensor([9]), tensor([9]))
        self.assertTrue(hash4.item() == 82)
        hash5 = drnl_hash_function(tensor([20]), tensor([20]))
        self.assertTrue(hash5.item() == 401)
        hash6 = drnl_hash_function(tensor([0]), tensor([0]))
        self.assertTrue(hash6.item() == 1)
        hash7 = drnl_hash_function(tensor([0]), tensor([1]))
        self.assertTrue(hash7.item() == 1)
        hash8 = drnl_hash_function(tensor([10]), tensor([0]))
        self.assertTrue(hash8.item() == 1)

    def test_get_drnl_lookup(self):
        max_dist = 10
        num_hops = 2
        z_to_idx, idx_to_dst = get_drnl_lookup(num_hops, max_dist)
        self.assertTrue(len(z_to_idx) == len(idx_to_dst))
        self.assertTrue(len(z_to_idx) == 20)
        z = torch.tensor([1, 1, 2, 3])
        z.apply_(lambda x: z_to_idx[x])
        self.assertTrue(torch.all(torch.eq(z, torch.tensor([0, 0, 1, 2]))))
