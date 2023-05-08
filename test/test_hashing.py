import unittest
from argparse import Namespace
import random
from math import isclose

from torch_geometric.nn import MessagePassing
from torch_geometric.utils.random import barabasi_albert_graph
from torch_geometric.utils import add_self_loops, to_undirected

import numpy as np
import scipy.sparse as ssp
import torch
from datasketch import MinHash, HyperLogLogPlusPlus

from src.hashing import ElphHashes, LABEL_LOOKUP
from src.datasets.seal import neighbors
from test_params import OPT, setup_seed


class HashingTests(unittest.TestCase):
    def setUp(self):
        self.n_nodes = 30
        self.n_edges = 100
        degree = 5  # number of edges to attach to each new node, not the degree at the end of the process
        self.x = torch.rand((self.n_nodes, 2))
        edge_index = barabasi_albert_graph(self.n_nodes, degree)
        edge_index = to_undirected(edge_index)
        self.edge_index, _ = add_self_loops(edge_index)
        edge_weight = torch.ones(self.edge_index.size(1), dtype=int)
        self.A = ssp.csr_matrix((edge_weight, (self.edge_index[0], self.edge_index[1])),
                                shape=(self.n_nodes, self.n_nodes))
        self.args = Namespace(**OPT)
        setup_seed(0)

    def test_minhash(self):
        src = [0]
        dst = [1]
        src_n = neighbors(src, self.A)
        dst_n = neighbors(dst, self.A)
        m1, m2 = MinHash(), MinHash()
        for elem in src_n:
            m1.update(elem)
        for elem in dst_n:
            m2.update(elem)
        print("Estimated Jaccard for data1 and data2 is", m1.jaccard(m2))
        jaccard = len(src_n.intersection(dst_n)) / len(src_n.union(dst_n))
        print(f'true Jaccard: {jaccard}')

    def test_hyperloglog(self):
        src = [0]
        dst = [1]
        src_n = list(neighbors(src, self.A))
        dst_n = list(neighbors(dst, self.A))
        union = src_n + dst_n
        hpp = HyperLogLogPlusPlus(p=8)
        for count, elem in enumerate(union):
            hpp.update(elem)
            if count % 2 == 0:
                print(f'estimate cardinality: {hpp.count()}')
        print(f'true size is {len(set(union))}')

    def test_build_hash_tables(self):
        eh = ElphHashes(self.args)
        eh.max_hops = 2
        hash_tables, cards = eh.build_hash_tables(self.n_nodes, self.edge_index)
        self.assertTrue(len(hash_tables[0]['minhash']) == self.n_nodes)
        self.assertTrue(len(hash_tables) == eh.max_hops + 1)
        eh.max_hops = 3
        hash_tables, cards = eh.build_hash_tables(self.n_nodes, self.edge_index)
        self.assertTrue(len(hash_tables[0]['minhash']) == self.n_nodes)
        self.assertTrue(len(hash_tables) == eh.max_hops + 1)

    def test_find_intersections(self):
        max_hops = 2
        self.args.max_hash_hops = max_hops
        eh = ElphHashes(self.args)
        hash_tables, cards = eh.build_hash_tables(self.n_nodes, self.edge_index)
        node1 = 0
        node2 = 1
        features = eh._get_intersections(torch.tensor([[node1, node2]]), hash_tables)
        self.assertTrue(len(features) == max_hops ** 2)
        max_hops = 3
        self.args.max_hash_hops = max_hops
        eh = ElphHashes(self.args)
        hash_tables, cards = eh.build_hash_tables(self.n_nodes, self.edge_index)
        features = eh._get_intersections(torch.tensor([[node1, node2]]), hash_tables)
        self.assertTrue(len(features) == max_hops ** 2)

    def test_find_neighbourhood_cardinality(self):
        # max_hops = 2
        eh = ElphHashes(self.args)
        hash_table, cards = eh.build_hash_tables(self.n_nodes, self.edge_index)
        node = 0
        # cards = find_neighbourhood_cardinality(node, hash_table, max_hops=2)
        neighbors1 = neighbors([node], self.A)
        neighbors2 = neighbors(neighbors1, self.A)
        n1, n2 = len(neighbors1) + 1, len(neighbors2) + 1  # node is counted by find_neighbourhood_cardinality
        self.assertTrue((cards[node, 0] > n1 - 1) and (cards[node, 0] < n1 + 1))
        self.assertTrue((cards[node, 1] > n2 - 2) and (cards[node, 1] < n2 + 2))

    def test_get_features(self):
        """
        These tests are stochastic, sometimes we get unlucky and one fails, which is generally nothing to worry about
        @return:
        """
        setup_seed(0)
        max_hops = 3
        self.args.max_hash_hops = max_hops
        #  some of these tests fail due to negative feature values, but this doesn't seem to be a problem for larger datasets
        self.args.floor_sf = True  # make subgraph features >= 0
        self.args.hll_p = 16  # choose a high value for more accuracy
        eh = ElphHashes(self.args)
        hash_tables, cards = eh.build_hash_tables(self.n_nodes, self.edge_index)
        node1 = 0
        node2 = 1
        # get the neigbhours<node_id,num_hops>. The cards include the central nodes, so need to add them here too
        neighbors11 = neighbors([node1], self.A).union({node1})
        neighbors21 = neighbors([node2], self.A).union({node2})
        neighbors12 = neighbors(neighbors11, self.A).union(neighbors11)
        neighbors22 = neighbors(neighbors21, self.A).union(neighbors21)
        neighbors13 = neighbors(neighbors12, self.A).union(neighbors12)
        neighbors23 = neighbors(neighbors22, self.A).union(neighbors22)
        # check the cardinality estimates
        cards1 = cards[node1].numpy()
        cards2 = cards[node2].numpy()
        # cards counts the central node whereas neighbours does not, so need to add 1
        self.assertAlmostEqual(cards1[0], len(neighbors11), 1)
        self.assertAlmostEqual(cards1[1], len(neighbors12), 1)
        self.assertAlmostEqual(cards1[2], len(neighbors13), 1)
        self.assertAlmostEqual(cards2[0], len(neighbors21), 1)
        self.assertAlmostEqual(cards2[1], len(neighbors22), 1)
        self.assertAlmostEqual(cards2[2], len(neighbors23), 1)

        tmp_features = eh.get_subgraph_features(torch.tensor([[node1, node2]]), hash_tables, cards).squeeze()
        self.assertTrue(torch.all(tmp_features >= 0))
        self.assertTrue(len(tmp_features) == max_hops * (max_hops + 2))
        # this is weird, but I changed features from dict -> tensor and this was the fastest way to change the tests
        features = {eh.label_lookup[idx]: val for idx, val in enumerate(tmp_features)}
        # test (1,1) features
        int11 = neighbors11.intersection(neighbors21)
        self.assertTrue(isclose(len(int11), features[(1, 1)], abs_tol=1))
        # test (2,1) features
        int21 = neighbors12.intersection(neighbors21)
        feat21 = int21.difference(int11)
        self.assertTrue(isclose(len(feat21), features[(2, 1)], abs_tol=1))
        # test (1,2) features
        int12 = neighbors11.intersection(neighbors22)
        feat12 = int12.difference(int11)
        self.assertTrue(isclose(len(feat12), features[(1, 2)], abs_tol=1))
        # test (2,2) features
        int22 = neighbors12.intersection(neighbors22)
        feat22 = int22.difference(feat12 | feat21 | int11)
        self.assertTrue(isclose(len(feat22), features[(2, 2)], abs_tol=2))
        # TEST ORDER 3 FROM HERE
        #  (3,1)
        int31 = neighbors13.intersection(neighbors21)
        feat31 = int31.difference(int11 | feat21)
        self.assertTrue(isclose(len(feat31), features[(3, 1)], abs_tol=1))
        # (1,3)
        int13 = neighbors11.intersection(neighbors23)
        feat13 = int13.difference(int11 | feat12)
        self.assertTrue(isclose(len(feat13), features[(1, 3)], abs_tol=1.5))
        # (3,2)
        int32 = neighbors13.intersection(neighbors22)
        feat32 = int32.difference(int11 | feat21 | feat12 | feat22 | feat31)
        self.assertTrue(isclose(len(feat32), features[(3, 2)], abs_tol=2))
        # (2, 3)
        int23 = neighbors12.intersection(neighbors23)
        feat23 = int23.difference(int11 | feat21 | feat12 | feat22 | feat13)
        self.assertTrue(isclose(len(feat23), features[(2, 3)], abs_tol=2))
        # (3, 3)
        int33 = neighbors13.intersection(neighbors23)
        feat33 = int33.difference(int11 | feat21 | feat12 | feat22 | feat31 | feat13 | feat23 | feat32)
        self.assertTrue(isclose(len(feat33), features[(3, 3)], abs_tol=2))
        # (0,1)
        feat01 = neighbors21.difference(int11 | feat21 | feat31)
        self.assertTrue(isclose(len(feat01), features[(0, 1)], abs_tol=2))

    def test_get_subgraph_features(self):
        max_hops = 2
        self.args.max_hash_hops = max_hops
        eh = ElphHashes(self.args)
        hashes, cards = eh.build_hash_tables(self.n_nodes, self.edge_index)
        links = torch.randint(self.n_nodes, (10, 2))
        sf = eh.get_subgraph_features(links, hashes, cards)
        self.assertTrue(sf.shape == torch.Size((len(links), len(LABEL_LOOKUP[2]))))
        for link, features in zip(links, sf):
            features_test = eh.get_subgraph_features(link, hashes, cards)
            self.assertTrue(torch.all(torch.eq(features, features_test)))
        sf[:, [4, 5]] = 0  # knock out the zero-one features
        for link, features in zip(links, sf):
            eh.use_zero_one = False
            features_test = eh.get_subgraph_features(link, hashes, cards)
            self.assertTrue(torch.all(torch.eq(features, features_test)))

    def test_label_lookup(self):
        for key, val in LABEL_LOOKUP.items():
            self.assertTrue(len(val) == key * (key + 2))

    def test_hll_p(self):
        max_hops = 2
        node1 = 0
        node2 = 1
        self.args.hll_p = 4
        n_links = 6
        eh = ElphHashes(self.args)
        hash_table, cards = eh.build_hash_tables(self.n_nodes, self.edge_index)
        intersections = eh._get_intersections(torch.tensor([[node1, node2]]), hash_table)
        self.assertTrue(len(intersections) == max_hops ** 2)
        cards1 = cards[node1]
        self.assertTrue(len(cards1) == max_hops)
        links = torch.randint(self.n_nodes, (n_links, 2))
        sf = eh.get_subgraph_features(links, hash_table, cards)
        self.assertTrue(sf.shape == (n_links, max_hops * (max_hops + 2)))

    def test_hll_counts(self):
        self.args.hll_p = 4
        eh = ElphHashes(self.args)
        n_registers = 10
        registers = torch.randint(high=2, size=(n_registers, 16))
        counts = eh.hll_count(registers)
        self.assertTrue(len(counts) == n_registers)
        hashes, cards = eh.build_hash_tables(self.n_nodes, self.edge_index)
        one_hop_counts = eh.hll_count(hashes[1]['hll'])
        two_hop_counts = eh.hll_count(hashes[2]['hll'])
        self.assertTrue(torch.allclose(cards[:, 0], one_hop_counts, atol=1e-8))
        self.assertTrue(torch.allclose(cards[:, 1], two_hop_counts, atol=1e-8))

    def test_refine_hll_count_estimate(self):
        eh = ElphHashes(self.args)
        n_links = 10
        estimates = torch.rand(n_links) + 5 * eh.m - 0.5  # make sure we have estimates above and below the threshold
        new_estimates = eh._refine_hll_count_estimate(estimates)
        self.assertTrue(new_estimates.shape == estimates.shape)
        idx = estimates > 5 * eh.m
        self.assertTrue(torch.allclose(new_estimates[idx], estimates[idx]))

    def test_jaccard(self):
        random.seed(0)
        self.args.max_hash_hops = 2
        eh = ElphHashes(self.args)
        hashes, cards = eh.build_hash_tables(self.n_nodes, self.edge_index)
        node1 = 0
        node2 = 1
        # get the neigbhours<node_id,num_hops>. The cards include the central nodes, so need to add them here too
        neighbors11 = neighbors([node1], self.A).union({node1})
        neighbors21 = neighbors([node2], self.A).union({node2})
        jaccard = len(neighbors11.intersection(neighbors21)) / len(neighbors11.union(neighbors21))
        jaccard_est = eh.jaccard(hashes[1]['minhash'][0], hashes[1]['minhash'][1])
        self.assertTrue(isclose(jaccard, jaccard_est, abs_tol=0.1))

    def test_intersections(self):
        setup_seed(0)
        self.args.max_hash_hops = 2
        eh = ElphHashes(self.args)
        hashes, cards = eh.build_hash_tables(self.n_nodes, self.edge_index)
        node1 = 0
        node2 = 1
        links = torch.tensor([[node1, node2], [node2, node1]])
        intersections = eh._get_intersections(links, hashes)
        self.assertTrue(len(intersections) == self.args.max_hash_hops ** 2)
        neighbors11 = neighbors([node1], self.A).union({node1})
        neighbors21 = neighbors([node2], self.A).union({node2})
        neighbors12 = neighbors(neighbors11, self.A).union(neighbors11)
        neighbors22 = neighbors(neighbors21, self.A).union(neighbors21)
        int11 = len(neighbors11.intersection(neighbors21))
        int22 = len(neighbors12.intersection(neighbors22))
        int21 = len(neighbors12.intersection(neighbors21))
        int12 = len(neighbors11.intersection(neighbors22))
        # (1,1)
        self.assertTrue(isclose(int11, intersections[(1, 1)][0], abs_tol=1))
        self.assertTrue(isclose(int11, intersections[(1, 1)][1], abs_tol=1))
        # (2,1)
        self.assertTrue(isclose(int21, intersections[(2, 1)][0], abs_tol=1))
        self.assertTrue(isclose(int21, intersections[(1, 2)][1], abs_tol=1))
        # (1,2)
        self.assertTrue(isclose(int12, intersections[(1, 2)][0], abs_tol=1))
        self.assertTrue(isclose(int12, intersections[(2, 1)][1], abs_tol=1))
        # (2,2)
        self.assertTrue(isclose(int22, intersections[(2, 2)][0], abs_tol=1))
        self.assertTrue(isclose(int22, intersections[(2, 2)][1], abs_tol=1))

    def test_ElphHashes(self):
        random.seed(0)
        self.args.max_hash_hops = 3
        eh = ElphHashes(self.args)
        hashes, cards = eh.build_hash_tables(self.n_nodes, self.edge_index)
        node1 = 0
        node2 = 1
        # get the neigbhours<node_id,num_hops>. The cards include the central nodes, so need to add them here too
        neighbors11 = neighbors([node1], self.A).union({node1})
        neighbors21 = neighbors([node2], self.A).union({node2})
        neighbors12 = neighbors(neighbors11, self.A).union(neighbors11)
        neighbors22 = neighbors(neighbors21, self.A).union(neighbors21)
        neighbors13 = neighbors(neighbors12, self.A).union(neighbors12)
        neighbors23 = neighbors(neighbors22, self.A).union(neighbors22)
        count11 = len(neighbors11)
        count21 = len(neighbors21)
        count12 = len(neighbors12)
        count22 = len(neighbors22)
        count13 = len(neighbors13)
        count23 = len(neighbors23)
        cards1 = cards[node1]
        cards2 = cards[node2]
        # cards counts the central node whereas neighbours does not, so need to add 1
        self.assertTrue(isclose(cards1[0].item(), count11, abs_tol=1))
        self.assertTrue(isclose(cards1[1].item(), count12, abs_tol=1.5))
        self.assertTrue(isclose(cards1[2].item(), count13, abs_tol=2))
        self.assertTrue(isclose(cards2[0].item(), count21, abs_tol=1))
        self.assertTrue(isclose(cards2[1].item(), count22, abs_tol=1.5))
        self.assertTrue(isclose(cards2[2].item(), count23, abs_tol=2))

    def test_neighbour_merge(self):
        max_hops = 2
        self.args.max_hash_hops = max_hops
        eh = ElphHashes(self.args)
        hashes, cards = eh.build_hash_tables(self.n_nodes, self.edge_index)
        node = 0
        neighbours = list(neighbors([node], self.A))
        root_reg = hashes[1]['hll'][node]
        root_hashvalues = hashes[1]['minhash'][node]
        regs = hashes[1]['hll'][neighbours]
        hashvalues = hashes[1]['minhash'][neighbours]
        two_hop_reg = eh.hll_neighbour_merge(root_reg, regs)
        two_hop_hashes = eh.minhash_neighbour_merge(root_hashvalues, hashvalues)
        true_reg = hashes[2]['hll'][node]
        true_hashvalues = hashes[2]['minhash'][node]
        self.assertTrue(torch.allclose(two_hop_reg, true_reg))
        self.assertTrue(torch.allclose(two_hop_hashes, true_hashvalues))

    def test_bit_length(self):
        eh = ElphHashes(self.args)
        arr = np.arange(1000)
        bit_lengths = eh._np_bit_length(arr)
        for bl, elem in zip(bit_lengths, arr):
            self.assertTrue(int(elem).bit_length() == bl)

    def test_initialise_hll(self):
        eh = ElphHashes(self.args)
        regs = eh.initialise_hll(self.n_nodes).numpy()
        self.assertTrue(regs.shape == (self.n_nodes, eh.m))
        self.assertTrue(np.array_equal(np.count_nonzero(regs, axis=1), np.ones(self.n_nodes)))
        self.assertTrue(np.amin(regs) >= 0)
        self.assertTrue(np.amax(regs) <= eh.max_rank)
        for node in range(self.n_nodes):
            self.assertTrue(np.isclose(eh.hll_count(torch.tensor(regs[node])).item(), 1, atol=0.1))

    def test_initialise_minhash(self):
        eh = ElphHashes(self.args)
        hashes = eh.initialise_minhash(self.n_nodes).numpy()
        self.assertTrue(hashes.shape == (self.n_nodes, eh.num_perm))
        self.assertTrue(np.amin(hashes) >= 0)
        self.assertTrue(np.amax(hashes) <= eh._max_minhash)

    def test_propagate_minhash(self):
        class MinPropagation(MessagePassing):
            def __init__(self):
                super().__init__(aggr='max')

            def forward(self, x, edge_index):
                out = self.propagate(edge_index, x=-x)
                return -out

        mp = MinPropagation()
        x = torch.tensor([[1, 2], [3, 1]])
        edge_index = torch.tensor([[0, 1, 0, 1], [0, 1, 1, 0]])
        out = mp(x, edge_index)
        self.assertTrue(torch.all(torch.eq(out, torch.tensor([[1, 1], [1, 1]]))))
        x = torch.tensor([[1, 2], [-3, 1]])
        out = mp(x, edge_index)
        self.assertTrue(torch.all(torch.eq(out, torch.tensor([[-3, 1], [-3, 1]]))))
        self.args.minhash_num_perm = 8
        self.args.hll_p = 4
        eh = ElphHashes(self.args)
        n_nodes = 2
        hashes = eh.initialise_minhash(n_nodes)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_index, _ = add_self_loops(edge_index, num_nodes=n_nodes)
        propagated_hashes = eh.minhash_prop(hashes, edge_index)
        truth = torch.min(hashes, dim=0)[0].repeat(n_nodes, 1)
        self.assertTrue(torch.all(torch.eq(propagated_hashes, truth)))
        hlls = eh.initialise_hll(n_nodes)
        propagated_hll = eh.hll_prop(hlls, edge_index)
        truth = torch.max(hlls, dim=0)[0].repeat(n_nodes, 1)
        self.assertTrue(torch.all(torch.eq(propagated_hll, truth)))

    def test_subgraph_features(self):
        setup_seed(0)
        n_links = 100
        self.args.max_hash_hops = 2
        self.args.floor_sf = True  # make subgraph features >= 0
        self.args.hll_p = 16  # choose a high value for more accuracy
        eh = ElphHashes(self.args)
        hashes, cards = eh.build_hash_tables(self.n_nodes, self.edge_index)
        links = torch.randint(self.n_nodes, (n_links, 2))
        sf = eh.get_subgraph_features(links, hashes, cards)
        self.assertTrue(sf.shape == (n_links, self.args.max_hash_hops * (self.args.max_hash_hops + 2)))
        self.args.max_hash_hops = 3
        eh = ElphHashes(self.args)
        hashes, cards = eh.build_hash_tables(self.n_nodes, self.edge_index)
        sf = eh.get_subgraph_features(links, hashes, cards)
        self.assertTrue(sf.shape == (n_links, self.args.max_hash_hops * (self.args.max_hash_hops + 2)))
        links = torch.tensor([[0, 1]])
        tmp_features = eh.get_subgraph_features(links, hashes, cards)
        node1 = 0
        node2 = 1
        # get the neigbhours<node_id,num_hops>. The cards include the central nodes, so need to add them here too
        neighbors11 = neighbors([node1], self.A).union({node1})
        neighbors21 = neighbors([node2], self.A).union({node2})
        neighbors12 = neighbors(neighbors11, self.A).union(neighbors11)
        neighbors22 = neighbors(neighbors21, self.A).union(neighbors21)
        neighbors13 = neighbors(neighbors12, self.A).union(neighbors12)
        neighbors23 = neighbors(neighbors22, self.A).union(neighbors22)

        features = {eh.label_lookup[idx]: val for idx, val in enumerate(tmp_features.flatten())}
        # test (1,1) features
        int11 = neighbors11.intersection(neighbors21)
        self.assertTrue(isclose(len(int11), features[(1, 1)], abs_tol=1))
        # test (2,1) features
        int21 = neighbors12.intersection(neighbors21)
        feat21 = int21.difference(int11)
        self.assertTrue(isclose(len(feat21), features[(2, 1)], abs_tol=1.5))
        # test (1,2) features
        int12 = neighbors11.intersection(neighbors22)
        feat12 = int12.difference(int11)
        self.assertTrue(isclose(len(feat12), features[(1, 2)], abs_tol=1))
        # test (2,2) features
        int22 = neighbors12.intersection(neighbors22)
        feat22 = int22.difference(feat12 | feat21 | int11)
        # for this test the two numbers come in around 12 and 16. looked, but can't explain the large error when
        # everything else is fine
        self.assertTrue(isclose(len(feat22), features[(2, 2)], abs_tol=4))
        # TEST ORDER 3 FROM HERE
        #  (3,1)
        int31 = neighbors13.intersection(neighbors21)
        feat31 = int31.difference(int11 | feat21)
        self.assertTrue(isclose(len(feat31), features[(3, 1)], abs_tol=1.5))
        # (1,3)
        int13 = neighbors11.intersection(neighbors23)
        feat13 = int13.difference(int11 | feat12)
        self.assertTrue(isclose(len(feat13), features[(1, 3)], abs_tol=1.5))
        # (3,2)
        int32 = neighbors13.intersection(neighbors22)
        feat32 = int32.difference(int11 | feat21 | feat12 | feat22 | feat31)
        self.assertTrue(isclose(len(feat32), features[(3, 2)], abs_tol=2))
        # (2, 3)
        int23 = neighbors12.intersection(neighbors23)
        feat23 = int23.difference(int11 | feat21 | feat12 | feat22 | feat13)
        self.assertTrue(isclose(len(feat23), features[(2, 3)], abs_tol=2))
        # (3, 3)
        int33 = neighbors13.intersection(neighbors23)
        feat33 = int33.difference(int11 | feat21 | feat12 | feat22 | feat31 | feat13 | feat23 | feat32)
        self.assertTrue(isclose(len(feat33), features[(3, 3)], abs_tol=2))
        # (0,1)
        feat01 = neighbors21.difference(int11 | feat21 | feat31)
        self.assertTrue(isclose(len(feat01), features[(0, 1)], abs_tol=2))
