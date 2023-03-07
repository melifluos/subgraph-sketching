import copy
import os
import pickle

import numpy as np
import pandas as pd
import torch
from ogb.linkproppred import PygLinkPropPredDataset
import torch_sparse
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import (add_self_loops, negative_sampling,
                                   to_undirected)
from utils import ROOT_DIR
from lcc import get_largest_connected_component, remap_edges, get_node_mapper


def get_data(args):
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
        splits = get_ogb_data(data, split_edge, dataset_name, args.bulk_train, args.num_negs)
    else:  # make random splits
        transform = RandomLinkSplit(is_undirected=undirected, num_val=val_pct, num_test=test_pct,
                                    add_negative_train_samples=include_negatives)
        train_data, val_data, test_data = transform(dataset.data)
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


def get_ogb_data(data, split_edge, dataset_name, bulk_train, num_negs=1):
    """
    ogb datasets come with fixed train-val-test splits and a fixed set of negatives against which to evaluate the test set
    The dataset.data object contains all of the nodes, but only the training edges
    @param dataset:
    @param use_valedges_as_input:
    @return:
    """
    if bulk_train:
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
    else:
        train_negs = get_ogb_train_negs(split_edge, data.edge_index, data.num_nodes, num_negs, dataset_name)
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


def get_same_source_negs(num_nodes, num_negs_per_pos, pos_edge):
    """
    The ogb-citation datasets uses negatives with the same src, but different dst to the positives
    :param num_nodes:
    :param num_negs_per_pos:
    :param pos_edge:
    :return:
    """
    print(f'generating {num_negs_per_pos} single source negatives for each positive source node')
    dst_neg = torch.randint(0, num_nodes, (1, pos_edge.size(1) * num_negs_per_pos), dtype=torch.long)
    src_neg = pos_edge[0].repeat_interleave(num_negs_per_pos)
    return torch.cat([src_neg.unsqueeze(0), dst_neg], dim=0)
