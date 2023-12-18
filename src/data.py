"""
Read and split ogb and planetoid datasets
"""

import os
import time
from typing import Optional, Tuple, Union


import torch
from torch.utils.data import DataLoader
from torch import Tensor
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import (add_self_loops, negative_sampling,
                                   to_undirected, to_networkx)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.loader import DataLoader as pygDataLoader
import wandb
import networkx as nx

from src.utils import ROOT_DIR, get_same_source_negs
from src.lcc import get_largest_connected_component, remap_edges, get_node_mapper
from src.datasets.seal import get_train_val_test_datasets
from src.datasets.elph import get_hashed_train_val_test_datasets, make_train_eval_data
from src.splits import get_splits

def get_loaders(args, dataset, splits: dict, directed: bool) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    train_data, val_data, test_data = splits['train'], splits['valid'], splits['test']
    if args.model in {'ELPH', 'BUDDY'}:
        train_dataset, val_dataset, test_dataset = get_hashed_train_val_test_datasets(dataset, train_data, val_data,
                                                                                      test_data, args, directed)
    else:
        t0 = time.time()
        train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(dataset, train_data, val_data, test_data,
                                                                               args)
        print(f'SEAL preprocessing ran in {time.time() - t0} s')
        if args.wandb:
            wandb.log({"seal_preprocessing_time": time.time() - t0})

    dl = DataLoader if args.model in {'ELPH', 'BUDDY'} else pygDataLoader
    train_loader = dl(train_dataset, batch_size=args.batch_size,
                      shuffle=True, num_workers=args.num_workers)
    # as the val and test edges are often sampled they also need to be shuffled
    # the citation2 dataset has specific negatives for each positive and so can't be shuffled
    shuffle_val = False if args.dataset_name.startswith('ogbl-citation') else True
    val_loader = dl(val_dataset, batch_size=args.batch_size, shuffle=shuffle_val,
                    num_workers=args.num_workers)
    shuffle_test = False if args.dataset_name.startswith('ogbl-citation') else True
    test_loader = dl(test_dataset, batch_size=args.batch_size, shuffle=shuffle_test,
                     num_workers=args.num_workers)
    if (args.dataset_name == 'ogbl-citation2') and (args.model in {'ELPH', 'BUDDY'}):
        train_eval_loader = dl(
            make_train_eval_data(train_dataset, dataset.num_nodes, n_pos_samples=5000), batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers)
    else:
        # todo change this so that eval doesn't have to use the full training set
        train_eval_loader = train_loader

    return train_loader, train_eval_loader, val_loader, test_loader


def get_data(args) -> Tuple[Union[Planetoid,PygLinkPropPredDataset], dict, bool, str]:
    """
    Read the dataset and generate train, val and test splits.
    For GNN link prediction edges play 2 roles 1/ message passing edges 2/ supervision edges
    - train message passing edges = train supervision edges
    - val message passing edges = train supervision edges
    val supervision edges are disjoint from the training edges
    - test message passing edges = val supervision + train message passing (= val message passing)
    test supervision edges are disjoint from both val and train supervision edges
    :param args: arguments Namespace object
    :return: dataset, dic splits, bool directed, str eval_metric
    """
    include_negatives = True
    dataset_name = args.dataset_name
    val_pct = args.val_pct
    test_pct = args.test_pct
    use_lcc_flag = True
    directed = False
    eval_metric = 'hits'
    path = os.path.join(ROOT_DIR, 'dataset', dataset_name)
    print(f'reading data from: {path}')
    if dataset_name.startswith('ogbl'):
        use_lcc_flag = False
        dataset = PygLinkPropPredDataset(name=dataset_name, root=path)
        if dataset_name == 'ogbl-ddi':
            dataset.data.x = torch.ones((dataset.data.num_nodes, 1))
            dataset.data.edge_weight = torch.ones(dataset.data.edge_index.size(1), dtype=int)
    else:
        dataset = Planetoid(path, dataset_name)

    # set the metric
    if dataset_name.startswith('ogbl-citation'):
        eval_metric = 'mrr'
        directed = True

    if use_lcc_flag:
        dataset = use_lcc(dataset)

    undirected = not directed

    if dataset_name.startswith('ogbl'):  # use the built in splits
        data = dataset[0]
        split_edge = dataset.get_edge_split()
        if dataset_name == 'ogbl-collab' and args.year > 0:  # filter out training edges before args.year
            data, split_edge = filter_by_year(data, split_edge, args.year)
        splits = get_ogb_data(data, split_edge, dataset_name, args.num_negs)
    else:  # make random splits
        if args.connected_holdout:
            # grape splits ensure that the train graph has no more connected components that the original graph
            splits = get_splits(dataset_name)
        else:
            transform = RandomLinkSplit(is_undirected=undirected, num_val=val_pct, num_test=test_pct,
                                        add_negative_train_samples=include_negatives)
            train_data, val_data, test_data = transform(dataset.data)
            nccs = check_ncc(train_data)  # val and test will have the same of less connected components
            print(f'train data has {nccs} connected components')
            splits = {'train': train_data, 'valid': val_data, 'test': test_data}

    return dataset, splits, directed, eval_metric


def filter_by_year(data, split_edge, year):
    """
    remove edges before year from data and split edge
    @param data: pyg Data, pyg SplitEdge
    @param split_edges:
    @param year: int first year to use
    @return: pyg Data, pyg SplitEdge
    """
    selected_year_index = torch.reshape(
        (split_edge['train']['year'] >= year).nonzero(as_tuple=False), (-1,))
    split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
    split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
    split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
    train_edge_index = split_edge['train']['edge'].t()
    # create adjacency matrix
    new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
    new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
    data.edge_index = new_edge_index
    data.edge_weight = new_edge_weight.unsqueeze(-1)
    return data, split_edge


def get_ogb_data(data, split_edge, dataset_name, num_negs=1):
    """
    ogb datasets come with fixed train-val-test splits and a fixed set of negatives against which to evaluate the test set
    The dataset.data object contains all of the nodes, but only the training edges
    @param dataset:
    @param use_valedges_as_input:
    @return:
    """
    if num_negs == 1:
        negs_name = f'{ROOT_DIR}/dataset/{dataset_name}/negative_samples.pt'
    else:
        negs_name = f'{ROOT_DIR}/dataset/{dataset_name}/negative_samples_{num_negs}.pt'
    print(f'looking for negative edges at {negs_name}')
    if os.path.exists(negs_name):
        print('loading negatives from disk')
        train_negs = torch.load(negs_name)
    else:
        print('negatives not found on disk. Generating negatives')
        train_negs = get_ogb_train_negs(split_edge, data.edge_index, data.num_nodes, num_negs, dataset_name)
        torch.save(train_negs, negs_name)
    # else:
    #     train_negs = get_ogb_train_negs(split_edge, data.edge_index, data.num_nodes, num_negs, dataset_name)
    splits = {}
    for key in split_edge.keys():
        # the ogb datasets come with test and valid negatives, but you have to cook your own train negs
        neg_edges = train_negs if key == 'train' else None
        edge_label, edge_label_index = make_obg_supervision_edges(split_edge, key, neg_edges)
        # use the validation edges for message passing at test time
        # according to the rules https://ogb.stanford.edu/docs/leader_rules/ only collab can use val edges at test time
        if key == 'test' and dataset_name == 'ogbl-collab':
            vei, vw = to_undirected(split_edge['valid']['edge'].t(), split_edge['valid']['weight'])
            edge_index = torch.cat([data.edge_index, vei], dim=1)
            edge_weight = torch.cat([data.edge_weight, vw.unsqueeze(-1)], dim=0)
        else:
            edge_index = data.edge_index
            if hasattr(data, "edge_weight"):
                edge_weight = data.edge_weight
            else:
                edge_weight = torch.ones(data.edge_index.shape[1])
        splits[key] = Data(x=data.x, edge_index=edge_index, edge_weight=edge_weight, edge_label=edge_label,
                           edge_label_index=edge_label_index)
    return splits


def get_ogb_pos_edges(split_edge, split):
    if 'edge' in split_edge[split]:
        pos_edge = split_edge[split]['edge']
    elif 'source_node' in split_edge[split]:
        pos_edge = torch.stack([split_edge[split]['source_node'], split_edge[split]['target_node']],
                               dim=1)
    else:
        raise NotImplementedError
    return pos_edge


def get_ogb_train_negs(split_edge, edge_index, num_nodes, num_negs=1, dataset_name=None):
    """
    for some inexplicable reason ogb datasets split_edge object stores edge indices as (n_edges, 2) tensors
    @param split_edge:
    @param edge_index: A [2, num_edges] tensor
    @param num_nodes:
    @param num_negs: the number of negatives to sample for each positive
    @return: A [num_edges * num_negs, 2] tensor of negative edges
    """
    pos_edge = get_ogb_pos_edges(split_edge, 'train').t()
    if dataset_name is not None and dataset_name.startswith('ogbl-citation'):
        neg_edge = get_same_source_negs(num_nodes, num_negs, pos_edge)
    else:  # any source is fine
        new_edge_index, _ = add_self_loops(edge_index)
        neg_edge = negative_sampling(
            new_edge_index, num_nodes=num_nodes,
            num_neg_samples=pos_edge.size(1) * num_negs)
    return neg_edge.t()


def make_obg_supervision_edges(split_edge, split, neg_edges=None):
    if neg_edges is not None:
        neg_edges = neg_edges
    else:
        if 'edge_neg' in split_edge[split]:
            neg_edges = split_edge[split]['edge_neg']
        elif 'target_node_neg' in split_edge[split]:
            n_neg_nodes = split_edge[split]['target_node_neg'].shape[1]
            neg_edges = torch.stack([split_edge[split]['source_node'].unsqueeze(1).repeat(1, n_neg_nodes).ravel(),
                                     split_edge[split]['target_node_neg'].ravel()
                                     ]).t()
        else:
            raise NotImplementedError

    pos_edges = get_ogb_pos_edges(split_edge, split)
    n_pos, n_neg = pos_edges.shape[0], neg_edges.shape[0]
    edge_label = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)], dim=0)
    edge_label_index = torch.cat([pos_edges, neg_edges], dim=0).t()
    return edge_label, edge_label_index


def check_ncc(data):
    """
    check the number of connected component
    @param dataset:
    @return:
    """
    graph = to_networkx(data, to_undirected=True)
    return nx.number_connected_components(graph)

def use_lcc(dataset):
    lcc = get_largest_connected_component(dataset)

    x_new = dataset.data.x[lcc]
    y_new = dataset.data.y[lcc]

    row, col = dataset.data.edge_index.numpy()
    edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
    edges = remap_edges(edges, get_node_mapper(lcc))

    data = Data(
        x=x_new,
        edge_index=torch.LongTensor(edges),
        y=y_new,
        train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
    )
    dataset.data = data
    return dataset

def sample_hard_negatives(edge_index: Tensor,
                      num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
                      num_neg_samples: Optional[int] = None)-> Tensor:
    """
    Sample hard negatives for each edge in edge_index
    @param edge_index:
    @return:
    """
    reg_negs = negative_sampling(edge_index, num_nodes, num_neg_samples)

    if num_nodes is None:
        num_nodes = maybe_num_nodes(edge_index)
