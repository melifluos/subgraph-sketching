"""
constructing the hashed data objects used by elph and buddy
"""

import os
from time import time

import numpy as np
import pandas as pd
import scipy.sparse as ssp
import torch
import torch_sparse
# from embiggen.embedders.ensmallen_embedders.hyper_sketching import HyperSketching
# from grape import Graph
from torch_geometric.data import Dataset
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_undirected
from torch_sparse import coalesce

from src.hashing import ElphHashes
from src.heuristics import RA
from src.utils import (get_pos_neg_edges, get_same_source_negs,
                       get_src_dst_degree)


class HashDataset(Dataset):
    """
    A class that combines propagated node features x, and subgraph features that are encoded as sketches of
    nodes k-hop neighbors
    """

    def __init__(
            self, root, split, data, pos_edges, neg_edges, args, use_coalesce=False,
            directed=False, use_unbiased_feature=False, **kwargs):
        if args.model != 'ELPH':  # elph stores the hashes directly in the model class for message passing
            self.elph_hashes = ElphHashes(args)  # object for hash and subgraph feature operations
        self.split = split  # string: train, valid or test
        self.root = root
        self.use_grape = args.use_grape
        self.use_grape_exact = args.use_grape_exact
        self.pos_edges = pos_edges
        self.neg_edges = neg_edges
        self.use_coalesce = use_coalesce
        self.directed = directed
        self.args = args
        self.remove_edge_bias = bool(args.remove_edge_bias)
        self.normalise_grape = bool(args.normalise_grape)
        self.load_features = args.load_features
        self.use_unbiased_feature = use_unbiased_feature
        self.use_zero_one = args.use_zero_one
        self.cache_subgraph_features = args.cache_subgraph_features
        self.max_hash_hops = args.max_hash_hops
        self.self_loops = args.self_loops
        self.use_feature = args.use_feature
        self.use_RA = args.use_RA
        self.hll_p = args.hll_p
        self.subgraph_features = None
        self.hashes = None
        super(HashDataset, self).__init__(root)

        self.links = torch.cat([self.pos_edges, self.neg_edges], 0)  # [n_edges, 2]
        self.labels = [1] * self.pos_edges.size(0) + [0] * self.neg_edges.size(0)

        if self.use_coalesce:  # compress multi-edge into edge with weight
            data.edge_index, data.edge_weight = coalesce(
                data.edge_index, data.edge_weight,
                data.num_nodes, data.num_nodes)

        if 'edge_weight' in data:
            self.edge_weight = data.edge_weight.view(-1)
        else:
            self.edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.int)
        if self.directed:  # make undirected graphs like citation2 directed
            print(
                f'this is a directed graph. Making the adjacency matrix undirected to propagate features and calculate subgraph features')
            self.edge_index, self.edge_weight = to_undirected(data.edge_index, self.edge_weight)
        else:
            self.edge_index = data.edge_index
        self.A = ssp.csr_matrix(
            (self.edge_weight, (self.edge_index[0], self.edge_index[1])),
            shape=(data.num_nodes, data.num_nodes)
        )

        self.degrees = torch.tensor(self.A.sum(axis=0, dtype=float), dtype=torch.float).flatten()

        if self.use_RA:
            self.RA = RA(self.A, self.links, batch_size=2000000)[0]

        if args.model == 'ELPH':  # features propagated in the model instead of preprocessed
            self.x = data.x
        else:
            self.x = self._preprocess_node_features(data, self.edge_index, self.edge_weight)
            if self.use_unbiased_feature:
                self.unbiased_features = self._preprocess_unbiased_node_features(data, self.edge_index,
                                                                                 self.edge_weight)
            # ELPH does hashing and feature prop on the fly
            # either set self.hashes or self.subgraph_features depending on cmd args
            self._preprocess_subgraph_features(self.edge_index.device, data.num_nodes, args.num_negs)

    def _generate_sign_features(self, data, edge_index, edge_weight, sign_k):
        """
        Generate features by preprocessing using the Scalable Inception Graph Neural Networks (SIGN) method
         https://arxiv.org/abs/2004.11198
        @param data: A pyg data object
        @param sign_k: the maximum number of times to apply the propagation operator
        @return:
        """
        try:
            num_nodes = data.x.size(0)
        except AttributeError:
            num_nodes = data.num_nodes
        edge_index, edge_weight = gcn_norm(  # yapf: disable
            edge_index, edge_weight.float(), num_nodes)
        if sign_k == 0:
            # for most datasets it works best do one step of propagation instead of using the raw features
            xs = torch_sparse.spmm(edge_index, edge_weight, data.x.shape[0], data.x.shape[0], data.x)
        else:
            xs = [data.x]
            for _ in range(sign_k):
                x = torch_sparse.spmm(edge_index, edge_weight, data.x.shape[0], data.x.shape[0], data.x)
                xs.append(x)
            xs = torch.cat(xs, dim=-1)
        return xs

    def _preprocess_node_features(self, data, edge_index, edge_weight, verbose=False):
        """
        preprocess the node features by propagating them using the SIGN method
        @param data: pyg Data object
        @param edge_weight: pyg edge index Int Tensor [edges, 2]
        @param sign_k: the number of propagation steps used by SIGN
        @return: Float Tensor [num_nodes, hidden_dim]
        """
        sk = self.args.sign_k
        if sk == 0:
            feature_name = f'{self.root}_{self.split}_featurecache.pt'
        else:
            feature_name = f'{self.root}_{self.split}_k{sk}_featurecache.pt'
        if self.load_features and os.path.exists(feature_name):
            print('loading node features from disk')
            x = torch.load(feature_name).to(edge_index.device)
        else:
            if verbose:
                print('constructing node features')
                start_time = time()
            x = self._generate_sign_features(data, edge_index, edge_weight, sk)
            if verbose:
                print("Preprocessed features in: {:.2f} seconds".format(time() - start_time))
            if self.load_features:
                torch.save(x.cpu(), feature_name)
        return x

    def _preprocess_unbiased_node_features(self, data, edge_index, edge_weight):
        """
        Propagating the node features at training time over all edges in the graph introduces a bias because
        the message passing edges are the same as the supervision edges at training time. This method propagates the
        node features one edge at a time generating features for edge ij with edge ij omitted from the graph.
        This only happens for positive features as features for negative edges can be calculated in parallel with
        matrix multiplication as they all share the same edge index.
        @param edge_index:
        @param data: pyg Data object
        @param edge_weight: pyg edge index Int Tensor [edges]
        @param sign_k: the number of propagation steps used by SIGN
        @return: Float Tensor [num_nodes, hidden_dim]
        """
        # todo:
        # 1. test this method
        feature_name = f'{self.root}_{self.split}_k{self.args.sign_k}_unbiased_feature_cache.pt'
        try:
            unbiased_features = torch.load(feature_name).to(edge_index.device)
            print(f'unbiased features found at {feature_name}.')
            return unbiased_features
        except FileNotFoundError:
            print(f'no unbiased features found at {feature_name}. Generating unbiased features')
        pos_unbiased_features = []
        #  don't load features from disk as we are generating them with each positive edge omitted
        old_flag = self.load_features
        self.load_features = False
        for edge in self.pos_edges:
            u, v = edge[0], edge[1]
            mask = ~((edge_index[0] == u) & (edge_index[1] == v) |
                     (edge_index[0] == v) & (edge_index[1] == u))
            assert torch.sum(mask) == edge_index.shape[1] - 2, ('either the pos edges are not in the edge index or '
                                                                'the edge index contains duplicates')
            x = self._preprocess_node_features(data, edge_index[:, mask], edge_weight[mask])
            feat = x[u] * x[v]
            pos_unbiased_features.append(feat)
        pos_unbiased_features = torch.stack(pos_unbiased_features, dim=0)
        assert pos_unbiased_features.shape[0] == self.pos_edges.shape[0], ('pos unbiased features are inconsistent with '                                                                           'the link object.')
        neg_node_features = self.x[self.neg_edges]
        neg_unbiased_features = neg_node_features[:, 0, :] * neg_node_features[:, 1, :]
        unbiased_features = torch.cat([pos_unbiased_features, neg_unbiased_features], dim=0)
        assert unbiased_features.shape[0] == len(
            self.links), 'unbiased features are inconsistent with the link object. Delete unbiased features file and regenerate'
        self.load_features = old_flag
        torch.save(unbiased_features, feature_name)
        return unbiased_features

    def _read_subgraph_features(self, name: str, device: torch.device) -> bool:
        """
        return True if the subgraph features can be read off disk, otherwise returns False
        @param name:
        @param device:
        @return:
        """
        retval = False
        # look on disk
        if self.cache_subgraph_features and os.path.exists(name):
            print(f'looking for subgraph features in {name}')
            self.subgraph_features = torch.load(name).to(device)
            print(f"cached subgraph features found at: {name}")
            assert self.subgraph_features.shape[0] == len(
                self.links), 'subgraph features are inconsistent with the link object. Delete subgraph features file and regenerate'
            retval = True
        return retval

    def _generate_file_names(self, num_negs):
        """
        get the subgraph feature file name and the stubs needed to make a new one if necessary
        :param num_negs: Int negative samples / positive sample
        :return:
        """
        if self.max_hash_hops != 2:
            hop_str = f'{self.max_hash_hops}hop_'
        else:
            hop_str = ''
        end_str = f'_{hop_str}subgraph_featurecache.pt'
        if self.args.dataset_name == 'ogbl-collab' and self.args.year > 0:
            year_str = f'_year_{self.args.year}'
        else:
            year_str = ''
        hll_str = f'_hllp{self.hll_p}'
        if self.use_grape:
            grape_str = '_grape'
            if self.use_grape_exact:
                hll_str = ''
                if self.remove_edge_bias:
                    grape_str += '_exact_unbiased'
                else:
                    grape_str += '_exact_biased'
        else:
            grape_str = ''
        if num_negs == 1 or self.split != 'train':
            subgraph_cache_name = f'{self.root}{self.split}{grape_str}{year_str}'
        else:
            subgraph_cache_name = f'{self.root}{self.split}{grape_str}_negs{num_negs}{year_str}'
        if self.use_zero_one:
            subgraph_cache_name += '_masked_features'
        subgraph_cache_name += hll_str
        subgraph_cache_name += end_str
        return subgraph_cache_name, year_str, hop_str

    def construct_grape_features(self, num_nodes: int, links: torch.Tensor):
        """
        Construct the edge_df using the grape hashing library instead of datasketch. This is much faster and only uses
        a fraction of the memory of datasketch by not generating minhashes
        @param num_nodes: number of nodes in the graph being sketched
        @param links: Tensor [n_edges, 2] of edge indices. Passed even though it is a class attribute because we don't always
        want to use all the edges
        @return:
        """
        # sketching = HyperSketching(
        #     number_of_hops=self.max_hash_hops,
        #     precision=self.args.hll_p,
        #     bits=6,
        #     include_selfloops=self.self_loops,
        #     normalize=self.normalise_grape,
        #     unbiased=self.remove_edge_bias,
        #     exact=self.use_grape_exact,
        #     zero_out_differences_cardinalities=not self.use_zero_one,
        #     include_node_types=False,
        #     include_edge_types=False,
        # )
        # node_df = pd.DataFrame({'name': np.arange(num_nodes)})
        # graph = Graph.from_pd(edges_df=pd.DataFrame(
        #     {'src': self.edge_index[0].cpu().numpy(), 'dst': self.edge_index[1].cpu().numpy()}),
        #     edge_src_column='src', edge_dst_column='dst', directed=False, nodes_df=node_df)
        #
        # # if not self.use_grape_exact:
        # sketching.fit(graph)
        # print(f'generated sketches for {graph.get_name()}. Graph has diameter {graph.get_diameter()}')
        # # edge_df is a dict with keys {"overlap": np.array,
        # # "left_difference": np.array, "right_difference:" np.array}
        # # double check cpu-> numpy offloading
        # edge_df = sketching.get_edge_feature_from_edge_node_ids(graph,
        #                                                         links[:, 0].cpu().numpy().astype(
        #                                                             np.uint32),
        #                                                         links[:, 1].cpu().numpy().astype(
        #                                                             np.uint32))

        # return torch.tensor(edge_df['edge_features'])
        return None

    def _preprocess_subgraph_features(self, device, num_nodes, num_negs=1):
        """
        Handles caching of hashes and subgraph features where each edge is fully hydrated as a preprocessing step
        Sets self.subgraph_features
        @return:
        """
        subgraph_cache_name, year_str, hop_str = self._generate_file_names(num_negs)
        # if the subgraph features are already on disk, read them into self.subgraph_features
        found_subgraph_features: bool = self._read_subgraph_features(subgraph_cache_name, device)
        if not found_subgraph_features:
            print(f'no subgraph features found at {subgraph_cache_name}. Generating subgraph features')
            start_time = time()
            if False:
                self.subgraph_features = self.construct_grape_features(num_nodes, self.links)
            else:
                hashes, cards = self.elph_hashes.build_hash_tables(num_nodes, self.edge_index)
                print(f'Preprocessed hashes in: {time() - start_time:.2f} seconds. Constructing subgraph features')
                self.subgraph_features = self.elph_hashes.get_subgraph_features(self.links, hashes, cards,
                                                                                self.args.subgraph_feature_batch_size)

            print("Preprocessed subgraph features in: {:.2f} seconds".format(time() - start_time))
            assert self.subgraph_features.shape[0] == len(
                self.links), 'subgraph features are a different shape link object. Delete subgraph features file and regenerate'
            if self.cache_subgraph_features:
                torch.save(self.subgraph_features, subgraph_cache_name)
        if self.args.floor_sf and self.subgraph_features is not None:
            self.subgraph_features[self.subgraph_features < 0] = 0
            print(
                f'setting {torch.sum(self.subgraph_features[self.subgraph_features < 0]).item()} negative values to zero')

    def len(self):
        return len(self.links)

    def get(self, idx):
        src, dst = self.links[idx]
        if self.args.use_struct_feature:
            subgraph_features = self.subgraph_features[idx]
        else:
            subgraph_features = torch.zeros(self.max_hash_hops * (2 + self.max_hash_hops))

        y = self.labels[idx]
        if self.use_RA:
            RA = self.A[src].dot(self.A_RA[dst].T)[0, 0]
            RA = torch.tensor([RA], dtype=torch.float)
        else:
            RA = -1
        src_degree, dst_degree = get_src_dst_degree(src, dst, self.A, None)
        node_features = torch.cat([self.x[src].unsqueeze(dim=0), self.x[dst].unsqueeze(dim=0)], dim=0)
        return subgraph_features, node_features, src_degree, dst_degree, RA, y


def get_hashed_train_val_test_datasets(dataset, train_data, val_data, test_data, args, directed=False):
    root = f'{dataset.root}/elph_'
    print(f'data path: {root}')
    use_coalesce = True if args.dataset_name == 'ogbl-collab' else False
    pos_train_edge, neg_train_edge = get_pos_neg_edges(train_data)
    pos_val_edge, neg_val_edge = get_pos_neg_edges(val_data)
    pos_test_edge, neg_test_edge = get_pos_neg_edges(test_data)
    print(
        f'before sampling, considering a superset of {pos_train_edge.shape[0]} pos, {neg_train_edge.shape[0]} neg train edges '
        f'{pos_val_edge.shape[0]} pos, {neg_val_edge.shape[0]} neg val edges '
        f'and {pos_test_edge.shape[0]} pos, {neg_test_edge.shape[0]} neg test edges for supervision')
    print('constructing training dataset object')
    train_dataset = HashDataset(root, 'train', train_data, pos_train_edge, neg_train_edge, args,
                                use_coalesce=use_coalesce, directed=directed,
                                use_unbiased_feature=args.use_unbiased_feature)
    print('constructing validation dataset object')
    val_dataset = HashDataset(root, 'valid', val_data, pos_val_edge, neg_val_edge, args,
                              use_coalesce=use_coalesce, directed=directed)
    print('constructing test dataset object')
    test_dataset = HashDataset(root, 'test', test_data, pos_test_edge, neg_test_edge, args,
                               use_coalesce=use_coalesce, directed=directed)
    return train_dataset, val_dataset, test_dataset


class HashedTrainEvalDataset(Dataset):
    """
    Subset of the full training dataset used to get unbiased estimate of training performance for large datasets
    where otherwise training eval is a significant % of runtime
    """

    def __init__(
            self, links, labels, subgraph_features, RA, dataset):
        super(HashedTrainEvalDataset, self).__init__()
        self.links = links
        self.labels = labels
        self.edge_index = dataset.edge_index
        self.subgraph_features = subgraph_features
        self.x = dataset.x
        self.degrees = dataset.degrees
        self.RA = RA

    def len(self):
        return len(self.links)

    def get(self, idx):
        return self.links[idx]


def make_train_eval_data(train_dataset, num_nodes, n_pos_samples=5000, n_negs_per_pos=None):
    """
    A much smaller subset of the training data to get a comparable (with test and val) measure of training performance
    to diagnose overfitting
    @param args: Namespace object of cmd args
    @param train_dataset: pyG Dataset object
    @param n_pos_samples: The number of positive samples to evaluate the training set on
    @return: HashedTrainEvalDataset
    """
    # ideally the negatives and the subgraph features are cached and just read from disk
    # need to save train_eval_negs_5000 and train_eval_subgraph_features_5000 files
    # and ensure that the order is always the same just as with the other datasets
    print('constructing dataset to evaluate training performance')
    n_pos_edges = len(train_dataset.pos_edges)
    n_neg_edges = len(train_dataset.neg_edges)
    negs_per_pos = int(n_neg_edges / n_pos_edges)
    if n_negs_per_pos is not None and n_negs_per_pos > negs_per_pos:
        # we don't have enough negatives in the training set already, so need to generate them.
        neg_sample = get_same_source_negs(num_nodes, train_dataset.pos_edges, n_negs_per_pos)
    else:
        neg_sample = train_dataset.neg_edges[:n_pos_samples * negs_per_pos]  # [num_neg_edges, 2]
    print(f'constructing dataset with {negs_per_pos} negative edges for each positive edge')
    pos_sample = train_dataset.pos_edges[:n_pos_samples]  # [num_edges, 2]

    assert torch.all(torch.eq(pos_sample[:, 0].repeat_interleave(negs_per_pos), neg_sample[:,
                                                                                0])), 'negatives have different source nodes to positives. Delete train_eval_negative_samples_* and subgraph features and regenerate'
    links = torch.cat([pos_sample, neg_sample], 0)  # [n_edges, 2]
    labels = [1] * pos_sample.size(0) + [0] * neg_sample.size(0)
    if train_dataset.use_RA:
        pos_RA = train_dataset.RA[:n_pos_samples]
        neg_RA = RA(train_dataset.A, neg_sample, batch_size=2000000)[0]
        RA_links = torch.cat([pos_RA, neg_RA], dim=0)
    else:
        RA_links = None
    pos_sf = train_dataset.subgraph_features[:n_pos_samples]
    if n_negs_per_pos is not None and n_negs_per_pos > negs_per_pos:
        start_time = time()
        if train_dataset.use_grape:
            # use grape Tensor[n_edges, max_hops(max_hops+2)]
            overlap, left, right = train_dataset.construct_grape_features(num_nodes, neg_sample)
            # use grape Tensor[n_edges, max_hops(max_hops+2)]
            neg_sf = torch.cat([overlap, left, right], dim=1)
        else:
            hashes, cards = train_dataset.elph_hashes.build_hash_tables(num_nodes, train_dataset.edge_index)
            print(f'Preprocessed hashes in: {time() - start_time:.2f} seconds. Constructing subgraph features')
            neg_sf = train_dataset.elph_hashes.get_subgraph_features(neg_sample, hashes, cards)
    else:
        neg_sf = train_dataset.subgraph_features[n_pos_edges: n_pos_edges + len(neg_sample)]
    # check these indices are all negative samples
    assert sum(train_dataset.labels[n_pos_edges: n_pos_edges + len(neg_sample)]) == 0
    subgraph_features = torch.cat([pos_sf, neg_sf], dim=0)
    train_eval_dataset = HashedTrainEvalDataset(links, labels, subgraph_features, RA_links, train_dataset)
    return train_eval_dataset
