"""
hashed based data sketching for graphs. Implemented in pytorch, but based on the datasketch library
"""
from time import time
import logging

from tqdm import tqdm
import torch
from torch import float
import numpy as np
from pandas.util import hash_array
from datasketch import HyperLogLogPlusPlus, hyperloglog_const
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# the LABEL_LOOKUP primary key is the max hops, secondary key is an index into the feature vector and values are hops
# from nodes (u,v)
LABEL_LOOKUP = {1: {0: (1, 1), 1: (0, 1), 2: (1, 0)},
                2: {0: (1, 1), 1: (2, 1), 2: (1, 2), 3: (2, 2), 4: (0, 1), 5: (1, 0), 6: (0, 2), 7: (2, 0)},
                3: {0: (1, 1), 1: (2, 1), 2: (1, 2), 3: (2, 2), 4: (3, 1), 5: (1, 3), 6: (3, 2), 7: (2, 3), 8: (3, 3),
                    9: (0, 1), 10: (1, 0), 11: (0, 2), 12: (2, 0), 13: (0, 3), 14: (3, 0)}}


class MinhashPropagation(MessagePassing):
    def __init__(self):
        super().__init__(aggr='max')

    @torch.no_grad()
    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=-x)
        return -out


class HllPropagation(MessagePassing):
    def __init__(self):
        super().__init__(aggr='max')

    @torch.no_grad()
    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return out


class ElphHashes(object):
    """
    class to store hashes and retrieve subgraph features
    """

    def __init__(self, args):
        assert args.max_hash_hops in {1, 2, 3}, f'hashing is not implemented for {args.max_hash_hops} hops'
        self.max_hops = args.max_hash_hops
        self.floor_sf = args.floor_sf  # if true set minimum sf to 0 (they're counts, so it should be)
        # minhash params
        self._mersenne_prime = np.uint64((1 << 61) - 1)
        self._max_minhash = np.uint64((1 << 32) - 1)
        self._minhash_range = (1 << 32)
        self.minhash_seed = 1
        self.num_perm = args.minhash_num_perm
        self.minhash_prop = MinhashPropagation()
        # hll params
        self.p = args.hll_p
        self.m = 1 << self.p  # the bitshift way of writing 2^p
        self.use_zero_one = args.use_zero_one
        self.label_lookup = LABEL_LOOKUP[self.max_hops]
        tmp = HyperLogLogPlusPlus(p=self.p)
        # store values that are shared and only depend on p
        self.hll_hashfunc = tmp.hashfunc
        self.alpha = tmp.alpha
        # the rank is the number of leading zeros. The max rank is the number of bits used in the hashes (64) minus p
        # as p bits are used to index the different counters
        self.max_rank = tmp.max_rank
        assert self.max_rank == 64 - self.p, 'not using 64 bits for hll++ hashing'
        self.hll_size = len(tmp.reg)
        self.hll_threshold = hyperloglog_const._thresholds[self.p - 4]
        self.bias_vector = torch.tensor(hyperloglog_const._bias[self.p - 4], dtype=torch.float)
        self.estimate_vector = torch.tensor(hyperloglog_const._raw_estimate[self.p - 4], dtype=torch.float)
        self.hll_prop = HllPropagation()

    def _np_bit_length(self, bits):
        """
        Get the number of bits required to represent each int in bits array
        @param bits: numpy [n_edges] array of ints
        @return:
        """
        return np.ceil(np.log2(bits + 1)).astype(int)

    def _get_hll_rank(self, bits):
        """
        get the number of leading zeros when each int in bits is represented as a self.max_rank-p bit array
        @param bits: a numpy array of ints
        @return:
        """
        # get the number of bits needed to represent each integer in bits
        bit_length = self._np_bit_length(bits)
        # the rank is the number of leading zeros, no idea about the +1 though
        rank = self.max_rank - bit_length + 1
        if min(rank) <= 0:
            raise ValueError("Hash value overflow, maximum size is %d\
                        bits" % self.max_rank)
        return rank

    def _init_permutations(self, num_perm):
        # Create parameters for a random bijective permutation function
        # that maps a 32-bit hash value to another 32-bit hash value.
        # http://en.wikipedia.org/wiki/Universal_hashing
        gen = np.random.RandomState(self.minhash_seed)
        return np.array([
            (gen.randint(1, self._mersenne_prime, dtype=np.uint64),
             gen.randint(0, self._mersenne_prime, dtype=np.uint64)) for
            _ in
            range(num_perm)
        ], dtype=np.uint64).T

    def initialise_minhash(self, n_nodes):
        init_hv = np.ones((n_nodes, self.num_perm), dtype=np.int64) * self._max_minhash
        a, b = self._init_permutations(self.num_perm)
        hv = hash_array(np.arange(1, n_nodes + 1))
        phv = np.bitwise_and((a * np.expand_dims(hv, 1) + b) % self._mersenne_prime, self._max_minhash)
        hv = np.minimum(phv, init_hv)
        return torch.tensor(hv, dtype=torch.int64)  # this conversion should be ok as self._max_minhash < max(int64)

    def initialise_hll(self, n_nodes):
        regs = np.zeros((n_nodes, self.m), dtype=np.int8)  # the registers to store binary values
        hv = hash_array(np.arange(1, n_nodes + 1))  # this function hashes 0 -> 0, so avoid
        # Get the index of the register using the first p bits of the hash
        # e.g. p=2, m=2^2=4, m-1=3=011 => only keep the right p bits
        reg_index = hv & (self.m - 1)
        # Get the rest of the hash. Python right shift drops the rightmost bits
        bits = hv >> self.p
        # Update the register
        ranks = self._get_hll_rank(bits)  # get the number of leading zeros in each element of bits
        regs[np.arange(n_nodes), reg_index] = np.maximum(regs[np.arange(n_nodes), reg_index], ranks)
        return torch.tensor(regs, dtype=torch.int8)  # int8 is fine as we're storing leading zeros in 64 bit numbers

    def build_hash_tables(self, num_nodes, edge_index):
        """
        Generate a hashing table that allows the size of the intersection of two nodes k-hop neighbours to be
        estimated in constant time
        @param num_nodes: The number of nodes in the graph
        @param adj: Int Tensor [2, edges] edges in the graph
        @return: hashes, cards. Hashes is a dictionary{dictionary}{tensor} with keys num_hops, 'hll' or 'minhash', cards
        is a tensor[n_nodes, max_hops-1]
        """
        hash_edge_index, _ = add_self_loops(edge_index)
        cards = torch.zeros((num_nodes, self.max_hops))
        node_hashings_table = {}
        for k in range(self.max_hops + 1):
            logger.info(f"Calculating hop {k} hashes")
            node_hashings_table[k] = {'hll': torch.zeros((num_nodes, self.hll_size), dtype=torch.int8),
                                      'minhash': torch.zeros((num_nodes, self.num_perm), dtype=torch.int64)}
            start = time()
            if k == 0:
                node_hashings_table[k]['minhash'] = self.initialise_minhash(num_nodes)
                node_hashings_table[k]['hll'] = self.initialise_hll(num_nodes)
            else:
                node_hashings_table[k]['hll'] = self.hll_prop(node_hashings_table[k - 1]['hll'], hash_edge_index)
                node_hashings_table[k]['minhash'] = self.minhash_prop(node_hashings_table[k - 1]['minhash'],
                                                                      hash_edge_index)
                cards[:, k - 1] = self.hll_count(node_hashings_table[k]['hll'])
            logger.info(f'{k} hop hash generation ran in {time() - start} s')
        return node_hashings_table, cards

    def _get_intersections(self, edge_list, hash_table):
        """
        extract set intersections as jaccard * union
        @param edge_list: [n_edges, 2] tensor to get intersections for
        @param hash_table:
        @param max_hops:
        @param p: hll precision parameter. hll uses 6 * 2^p bits
        @return:
        """
        intersections = {}
        # create features for each combination of hops.
        for k1 in range(1, self.max_hops + 1):
            for k2 in range(1, self.max_hops + 1):
                src_hll = hash_table[k1]['hll'][edge_list[:, 0]]
                src_minhash = hash_table[k1]['minhash'][edge_list[:, 0]]
                dst_hll = hash_table[k2]['hll'][edge_list[:, 1]]
                dst_minhash = hash_table[k2]['minhash'][edge_list[:, 1]]
                jaccard = self.jaccard(src_minhash, dst_minhash)
                unions = self._hll_merge(src_hll, dst_hll)
                union_size = self.hll_count(unions)
                intersection = jaccard * union_size
                intersections[(k1, k2)] = intersection
        return intersections

    def get_hashval(self, x):
        return x.hashvals

    def _linearcounting(self, num_zero):
        return self.m * torch.log(self.m / num_zero)

    def _estimate_bias(self, e):
        """
        Not exactly sure what this is doing or why exactly 6 nearest neighbours are used.
        @param e: torch tensor [n_links] of estimates
        @return:
        """
        nearest_neighbors = torch.argsort((e.unsqueeze(-1) - self.estimate_vector.to(e.device)) ** 2)[:, :6]
        return torch.mean(self.bias_vector.to(e.device)[nearest_neighbors], dim=1)

    def _refine_hll_count_estimate(self, estimate):
        idx = estimate <= 5 * self.m
        estimate_bias = self._estimate_bias(estimate)
        estimate[idx] = estimate[idx] - estimate_bias[idx]
        return estimate

    def hll_count(self, regs):
        """
        Estimate the size of set unions associated with regs
        @param regs: A tensor of registers [n_nodes, register_size]
        @return:
        """
        if regs.dim() == 1:
            regs = regs.unsqueeze(dim=0)
        retval = torch.ones(regs.shape[0], device=regs.device) * self.hll_threshold + 1
        num_zero = self.m - torch.count_nonzero(regs, dim=1)
        idx = num_zero > 0
        lc = self._linearcounting(num_zero[idx])
        retval[idx] = lc
        # we only keep lc values where lc <= self.hll_threshold, otherwise
        estimate_indices = retval > self.hll_threshold
        # Use HyperLogLog estimation function
        e = (self.alpha * self.m ** 2) / torch.sum(2.0 ** (-regs[estimate_indices]), dim=1)
        # for some reason there are two more cases
        e = self._refine_hll_count_estimate(e)
        retval[estimate_indices] = e
        return retval

    def _hll_merge(self, src, dst):
        if src.shape != dst.shape:
            raise ValueError('source and destination register shapes must be the same')
        return torch.maximum(src, dst)

    def hll_neighbour_merge(self, root, neighbours):
        all_regs = torch.cat([root.unsqueeze(dim=0), neighbours], dim=0)
        return torch.max(all_regs, dim=0)[0]

    def minhash_neighbour_merge(self, root, neighbours):
        all_regs = torch.cat([root.unsqueeze(dim=0), neighbours], dim=0)
        return torch.min(all_regs, dim=0)[0]

    def jaccard(self, src, dst):
        """
        get the minhash Jaccard estimate
        @param src: tensor [n_edges, num_perms] of hashvalues
        @param dst: tensor [n_edges, num_perms] of hashvalues
        @return: tensor [n_edges] jaccard estimates
        """
        if src.shape != dst.shape:
            raise ValueError('source and destination hash value shapes must be the same')
        return torch.count_nonzero(src == dst, dim=-1) / self.num_perm

    def get_subgraph_features(self, links, hash_table, cards, batch_size=11000000):
        """
        extracts the features that play a similar role to the labeling trick features. These can be thought of as approximations
        of path distances from the source and destination nodes. There are k+2+\sum_1^k 2k features
        @param links: tensor [n_edges, 2]
        @param hash_table: A Dict{Dict} of torch tensor [num_nodes, hash_size] keys are hop index and hash type (hyperlogloghash, minhash)
        @param cards: Tensor[n_nodes, max_hops] of hll neighbourhood cardinality estimates
        @param batch_size: batch size for computing intersections. 11m splits the large ogb datasets into 3.
        @return: Tensor[n_edges, max_hops(max_hops+2)]
        """
        if links.dim() == 1:
            links = links.unsqueeze(0)
        link_loader = DataLoader(range(links.size(0)), batch_size, shuffle=False, num_workers=0)
        all_features = []
        for batch in tqdm(link_loader):
            intersections = self._get_intersections(links[batch], hash_table)
            cards1, cards2 = cards.to(links.device)[links[batch, 0]], cards.to(links.device)[links[batch, 1]]
            features = torch.zeros((len(batch), self.max_hops * (self.max_hops + 2)), dtype=float, device=links.device)
            features[:, 0] = intersections[(1, 1)]
            if self.max_hops == 1:
                features[:, 1] = cards2[:, 0] - features[:, 0]
                features[:, 2] = cards1[:, 0] - features[:, 0]
            elif self.max_hops == 2:
                features[:, 1] = intersections[(2, 1)] - features[:, 0]  # (2,1)
                features[:, 2] = intersections[(1, 2)] - features[:, 0]  # (1,2)
                features[:, 3] = intersections[(2, 2)] - features[:, 0] - features[:, 1] - features[:, 2]  # (2,2)
                features[:, 4] = cards2[:, 0] - torch.sum(features[:, 0:2], dim=1)  # (0, 1)
                features[:, 5] = cards1[:, 0] - features[:, 0] - features[:, 2]  # (1, 0)
                features[:, 6] = cards2[:, 1] - torch.sum(features[:, 0:5], dim=1)  # (0, 2)
                features[:, 7] = cards1[:, 1] - features[:, 0] - torch.sum(features[:, 0:4], dim=1) - features[:,
                                                                                                      5]  # (2, 0)
            elif self.max_hops == 3:
                features[:, 1] = intersections[(2, 1)] - features[:, 0]  # (2,1)
                features[:, 2] = intersections[(1, 2)] - features[:, 0]  # (1,2)
                features[:, 3] = intersections[(2, 2)] - features[:, 0] - features[:, 1] - features[:, 2]  # (2,2)
                features[:, 4] = intersections[(3, 1)] - features[:, 0] - features[:, 1]  # (3,1)
                features[:, 5] = intersections[(1, 3)] - features[:, 0] - features[:, 2]  # (1, 3)
                features[:, 6] = intersections[(3, 2)] - torch.sum(features[:, 0:4], dim=1) - features[:, 4]  # (3,2)
                features[:, 7] = intersections[(2, 3)] - torch.sum(features[:, 0:4], dim=1) - features[:, 5]  # (2,3)
                features[:, 8] = intersections[(3, 3)] - torch.sum(features[:, 0:8], dim=1)  # (3,3)
                features[:, 9] = cards2[:, 0] - features[:, 0] - features[:, 1] - features[:, 4]  # (0, 1)
                features[:, 10] = cards1[:, 0] - features[:, 0] - features[:, 2] - features[:, 5]  # (1, 0)
                features[:, 11] = cards2[:, 1] - torch.sum(features[:, 0:5], dim=1) - features[:, 6] - features[:,
                                                                                                       9]  # (0, 2)
                features[:, 12] = cards1[:, 1] - torch.sum(features[:, 0:5], dim=1) - features[:, 7] - features[:,
                                                                                                       10]  # (2, 0)
                features[:, 13] = cards2[:, 2] - torch.sum(features[:, 0:9], dim=1) - features[:, 9] - features[:,
                                                                                                       11]  # (0, 3)
                features[:, 14] = cards1[:, 2] - torch.sum(features[:, 0:9], dim=1) - features[:, 10] - features[:,
                                                                                                        12]  # (3, 0)
            else:
                raise NotImplementedError("Only 1, 2 and 3 hop hashes are implemented")
            if not self.use_zero_one:
                if self.max_hops == 2:  # for two hops any positive edge that's dist 1 from u must be dist 2 from v etc.
                    features[:, 4] = 0
                    features[:, 5] = 0
                elif self.max_hops == 3:  # in addition for three hops 0,2 is impossible for positive edges
                    features[:, 4] = 0
                    features[:, 5] = 0
                    features[:, 11] = 0
                    features[:, 12] = 0
            if self.floor_sf:  # should be more accurate, but in practice makes no difference
                features[features < 0] = 0
            all_features.append(features)
        features = torch.cat(all_features, dim=0)
        return features
