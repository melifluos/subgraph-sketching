"""
constructing the hashed data objects used by elph and buddy
"""

import os
from time import time
from math import inf

import torch
from torch_geometric.data import Dataset
from torch_geometric.utils import to_undirected
from torch_sparse import coalesce
import scipy.sparse as ssp
import torch_sparse
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from heuristics import RA
from utils import ROOT_DIR, get_src_dst_degree, get_pos_neg_edges, get_same_source_negs
from hashing import ElphHashes


class HashedDynamicDataset(Dataset):
    """
    A class that combines propagated node features x, and graph substructure features that are encoded as sketches of
    nodes k-hop neighbors
    """

    def __init__(
            self, root, split, data, pos_edges, neg_edges, args, cache_structure_features=False, use_coalesce=False,
            directed=False, **kwargs):
        if args.model != 'ELPH':  # elph stores the hashes directly in the model class for message passing
            self.elph_hashes = ElphHashes(args)  # object for hash and subgraph feature operations
        self.split = split  # string: train, valid or test
        self.root = root
        self.pos_edges = pos_edges
        self.neg_edges = neg_edges
        self.use_coalesce = use_coalesce
        self.directed = directed
        self.args = args
        self.load_features = args.load_features
        self.load_hashes = args.load_hashes
        self.use_zero_one = args.use_zero_one
        self.cache_structure_features = cache_structure_features
        self.max_hash_hops = args.max_hash_hops
        self.use_feature = args.use_feature
        self.use_RA = args.use_RA
        self.hll_p = args.hll_p
        self.structure_features = None
        self.hashes = None
        super(HashedDynamicDataset, self).__init__(root)

        self.links = torch.cat([self.pos_edges, self.neg_edges], 0)  # [n_edges, 2]
        self.labels = [1] * self.pos_edges.size(0) + [0] * self.neg_edges.size(0)

        if self.use_coalesce:  # compress multi-edge into edge with weight
            data.edge_index, data.edge_weight = coalesce(
                data.edge_index, data.edge_weight,
                data.num_nodes, data.num_nodes)

        if 'edge_weight' in data:
            self.edge_weight = data.edge_weight.view(-1)
        else:
            self.edge_weight = torch.ones(data.edge_index.size(1), dtype=int)
        if self.directed:  # make undirected graphs like citation2 directed
            print(
                f'this is a directed graph. Making the adjacency matrix undirected to propagate features and calculate structure features')
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
            self.x = self._preprocess_node_features(data, self.edge_index, self.edge_weight, args.sign_k)
        if args.model != 'ELPH':  # ELPH does hashing and feature prop on the fly
            # either set self.hashes or self.structure_features depending on cmd args
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
            # for most datasets it works best do one step of propagation
            xs = torch_sparse.spmm(edge_index, edge_weight, data.x.shape[0], data.x.shape[0], data.x)
        else:
            xs = [data.x]
            for _ in range(sign_k):
                x = torch_sparse.spmm(edge_index, edge_weight, data.x.shape[0], data.x.shape[0], data.x)
                xs.append(x)
            xs = torch.cat(xs, dim=-1)
        return xs

    def _preprocess_node_features(self, data, edge_index, edge_weight, sign_k=0):
        """
        preprocess the node features
        @param data: pyg Data object
        @param edge_weight: pyg edge index Int Tensor [edges, 2]
        @param sign_k: the number of propagation steps used by SIGN
        @return: Float Tensor [num_nodes, hidden_dim]
        """
        if sign_k == 0:
            feature_name = f'{self.root}_{self.split}_featurecache.pt'
        else:
            feature_name = f'{self.root}_{self.split}_k{sign_k}_featurecache.pt'
        if self.load_features and os.path.exists(feature_name):
            print('loading node features from disk')
            x = torch.load(feature_name).to(edge_index.device)
        else:
            print('constructing node features')
            start_time = time()
            x = self._generate_sign_features(data, edge_index, edge_weight, sign_k)
            print("Preprocessed features in: {:.2f} seconds".format(time() - start_time))
            if self.load_features:
                torch.save(x.cpu(), feature_name)
        return x

    def _read_subgraph_features(self, name, device):
        """
        return True if the subgraph features can be read off disk, otherwise returns False
        @param name:
        @param device:
        @return:
        """
        retval = False
        # look on disk
        if self.cache_structure_features and os.path.exists(name):
            print(f'looking for subgraph features in {name}')
            self.structure_features = torch.load(name).to(device)
            print(f"cached subgraph features found at: {name}")
            assert self.structure_features.shape[0] == len(
                self.links), 'subgraph features are inconsistent with the link object. Delete structure features file and regenerate'
            retval = True
        return retval

    def get_structure_feature_cache_name(self):
        if self.split == 'test':
            if self.args.test_samples != inf:
                retval = f'{self.root}_{self.split}_{self.args.test_samples}_structure_featurecache.pt'
            else:
                retval = f'{self.root}_{self.split}_structure_featurecache.pt'
        elif self.split == 'valid':
            if self.args.val_samples != inf:
                retval = f'{self.root}_{self.split}_{self.args.val_samples}_structure_featurecache.pt'
            else:
                retval = f'{self.root}_{self.split}_structure_featurecache.pt'
        elif self.split == 'train':
            if self.args.train_samples != inf:
                retval = f'{self.root}_{self.split}_{self.args.train_samples}_structure_featurecache.pt'
            else:
                retval = f'{self.root}_{self.split}_structure_featurecache.pt'
        return retval

    def _preprocess_subgraph_features(self, device, num_nodes, num_negs=1):
        """
        Handles caching of hashes and or full structure features where each edge is fully hydrated as a preprocessing step
        Will either set self.hashes, self.structure_features depending on cmd args
        @return:
        """
        # If cache structure features was set to True then look for them
        # If we find them, we don't need the hashes
        if self.max_hash_hops != 2:
            hop_str = f'{self.max_hash_hops}hop_'
        else:
            hop_str = ''
        end_str = f'_{hop_str}structure_featurecache.pt'
        if self.args.dataset_name == 'ogbl-collab' and self.args.year > 0:
            year_str = f'year_{self.args.year}'
        else:
            year_str = ''
        if num_negs == 1 or self.split != 'train':
            structure_cache_name = f'{self.root}_{self.split}{year_str}{end_str}'
        else:
            structure_cache_name = f'{self.root}_{self.split}_negs{num_negs}{year_str}{end_str}'
        found_subgraph_feautures = self._read_subgraph_features(structure_cache_name, device)
        if not found_subgraph_feautures:
            print(f'no structure features found at {structure_cache_name}. Need to generate them')
            hash_name = f'{self.root}{self.split}{year_str}_{hop_str}hashcache.pt'
            cards_name = f'{self.root}{self.split}{year_str}_{hop_str}cardcache.pt'
            if self.load_hashes and os.path.exists(hash_name):
                print('loading hashes from disk')
                hashes = torch.load(hash_name)
                if os.path.exists(cards_name):
                    print('loading cards from disk')
                    cards = torch.load(cards_name)
                else:
                    print(f'hashes found at {hash_name}, but cards not found. Delete hashes and run again')
            else:
                print('constructing hashes')
                start_time = time()
                hashes, cards = self.elph_hashes.build_hash_tables(num_nodes, self.edge_index)
                print("Preprocessed hashes in: {:.2f} seconds".format(time() - start_time))
                if self.load_hashes:
                    torch.save(cards, cards_name)
                    torch.save(hashes, hash_name)

            if self.cache_structure_features and self.structure_features is None:
                # now we have the hashes we can make the structure features
                print('constructing structure features')
                start_time = time()
                self.structure_features = self.elph_hashes.get_subgraph_features(self.links, hashes, cards)
                print("Preprocessed structure features in: {:.2f} seconds".format(time() - start_time))
                assert self.structure_features.shape[0] == len(
                    self.links), 'structure features are a different shape link object. Delete structure features file and regenerate'
                torch.save(self.structure_features, structure_cache_name)
            else:  # we need the hashes if we're not storing the structure features or using the database
                self.hashes = hashes
                self.cards = cards
        if self.args.floor_sf and self.structure_features is not None:
            self.structure_features[self.structure_features < 0] = 0
            print(
                f'setting {torch.sum(self.structure_features[self.structure_features < 0]).item()} negative values to zero')
        if not self.use_zero_one and self.structure_features is not None:  # knock out the zero_one features (0,1) and (1,0)
            if self.max_hash_hops > 1:
                self.structure_features[:, [4, 5]] = 0
            if self.max_hash_hops == 3:
                self.structure_features[:, [11, 12]] = 0  # also need to get rid of (0, 2) and (2, 0)

    def len(self):
        return len(self.links)

    def get(self, idx):
        src, dst = self.links[idx]
        if self.args.use_struct_feature:
            if self.cache_structure_features:
                try:
                    structure_features = self.structure_features[idx]
                except TypeError:  # structure features are only cached in te
                    structure_features = self.elph_hashes.get_subgraph_features(self.links[idx], self.hashes,
                                                                                self.cards)
            else:
                structure_features = self.elph_hashes.get_subgraph_features(self.links[idx], self.hashes,
                                                                            self.cards)
        else:
            structure_features = torch.zeros(8)

        y = self.labels[idx]
        if self.use_RA:
            RA = self.A[src].dot(self.A_RA[dst].T)[0, 0]
            RA = torch.tensor([RA], dtype=torch.float)
        else:
            RA = -1
        src_degree, dst_degree = get_src_dst_degree(src, dst, self.A, None)
        node_features = torch.cat([self.x[src].unsqueeze(dim=0), self.x[dst].unsqueeze(dim=0)], dim=0)
        return structure_features, node_features, src_degree, dst_degree, RA, y


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
    train_dataset = HashedDynamicDataset(root, 'train', train_data, pos_train_edge, neg_train_edge, args,
                                         args.cache_train_structure_features,
                                         use_coalesce=use_coalesce, directed=directed)
    print('constructing validation dataset object')
    val_dataset = HashedDynamicDataset(root, 'valid', val_data, pos_val_edge, neg_val_edge, args,
                                       args.cache_val_structure_features,
                                       use_coalesce=use_coalesce, directed=directed)
    print('constructing test dataset object')
    test_dataset = HashedDynamicDataset(root, 'test', test_data, pos_test_edge, neg_test_edge, args,
                                        args.cache_test_structure_features,
                                        use_coalesce=use_coalesce, directed=directed)
    return train_dataset, val_dataset, test_dataset


class HashedTrainEvalDataset(Dataset):
    """
    Subset of the full training dataset used to get unbiased estimate of training performance for large datasets
    where otherwise training eval is a significant % of runtime
    """

    def __init__(
            self, links, labels, structure_features, RA, dataset):
        self.links = links
        self.labels = labels
        self.edge_index = dataset.edge_index
        self.structure_features = structure_features
        self.x = dataset.x
        self.degrees = dataset.degrees
        self.RA = RA

    def len(self):
        return len(self.links)

    def get(self, idx):
        return self.links[idx]


def make_train_eval_data(args, train_dataset, num_nodes, hll_p=8, n_pos_samples=5000, negs_per_pos=1000):
    """
    A much smaller subset of the training data to get a comparable (with test and val) measure of training performance
    to diagnose overfitting
    @param train_dataset: pyG Dataset object
    @param n_pos_samples: The number of positive samples to evaluate the training set on
    @return: HashedTrainEvalDataset
    """
    # ideally the negatives and the structure features are cached and just read from disk
    # need to save train_eval_negs_5000 and train_eval_structure_features_5000 files
    # and ensure that the order is always the same just as with the other datasets
    print('constructing dataset to evaluate training performance')
    dataset_name = args.dataset_name
    pos_sample = train_dataset.pos_edges[:n_pos_samples]  # [num_edges, 2]
    negs_name = f'{ROOT_DIR}/dataset/{dataset_name}/train_eval_negative_samples_{negs_per_pos}.pt'
    print(f'looking for negative edges at {negs_name}')
    if os.path.exists(negs_name):
        print('loading negatives from disk')
        neg_sample = torch.load(negs_name)
    else:
        print('negatives not found on disk. Generating negatives')
        neg_sample = get_same_source_negs(num_nodes, negs_per_pos, pos_sample.t()).t()  # [num_neg_edges, 2]
        torch.save(neg_sample, negs_name)
    # make sure these are the correct negative samples with source nodes corresponding to the positive samples
    assert torch.all(torch.eq(pos_sample[:, 0].repeat_interleave(negs_per_pos), neg_sample[:,
                                                                                0])), 'negatives have different source nodes to positives. Delete train_eval_negative_samples_* and structure features and regenerate'
    links = torch.cat([pos_sample, neg_sample], 0)  # [n_edges, 2]
    labels = [1] * pos_sample.size(0) + [0] * neg_sample.size(0)
    if train_dataset.use_RA:
        pos_RA = train_dataset.RA[:n_pos_samples]
        neg_RA = RA(train_dataset.A, neg_sample, batch_size=2000000)[0]
        RA_links = torch.cat([pos_RA, neg_RA], dim=0)
    else:
        RA_links = None
    pos_sf = train_dataset.structure_features[:n_pos_samples]
    # try to read negative structure features from disk or generate them
    structure_cache_name = f'{ROOT_DIR}/dataset/{dataset_name}/train_eval_negative_samples_{negs_per_pos}_structure_featurecache.pt'
    print(f'looking for structure features at {structure_cache_name}')
    if os.path.exists(structure_cache_name):
        neg_sf = torch.load(structure_cache_name).to(pos_sf.device)
        print(f"cached structure features found at: {structure_cache_name}")
        assert neg_sf.shape[0] == len(
            neg_sample * negs_per_pos), 'structure features are a different shape link object. Delete structure features file and regenerate'
    else:  # generate negative structure features
        #  we're going to need the hashes
        file_stub = dataset_name.replace('-', '_')  # pyg likes to add -
        if args.max_hash_hops == 3:
            hash_name = f'{ROOT_DIR}/dataset/{dataset_name}/{file_stub}_elph__train_3hop_hashcache.pt'
        else:
            hash_name = f'{ROOT_DIR}/dataset/{dataset_name}/{file_stub}_elph__train_hashcache.pt'
        print(f'looking for hashes at {hash_name}')
        eh = ElphHashes(args)
        if os.path.exists(hash_name):
            hashes = torch.load(hash_name)
            print(f"cached hashes found at: {hash_name}")
        else:  # need to generate the hashes, but this is a corner case as they should have been generated to make the training dataset
            hashes, cards = eh.build_hash_tables(num_nodes, train_dataset.edge_index)
            torch.save(hashes, hash_name)
        print('caching structure features for negative samples to evaluate training performance')
        neg_sf = eh.get_subgraph_features(neg_sample, hashes, cards)
        torch.save(neg_sf, structure_cache_name)
    structure_features = torch.cat([pos_sf, neg_sf], dim=0)
    train_eval_dataset = HashedTrainEvalDataset(links, labels, structure_features, RA_links, train_dataset)
    return train_eval_dataset
