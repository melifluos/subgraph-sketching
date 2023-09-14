"""
training functions
"""
import time
from math import inf

import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
import wandb
import numpy as np

from src.utils import get_num_samples


def get_train_func(args):
    if args.model == 'ELPH':
        return train_elph
    elif args.model == 'BUDDY':
        train_func = train_buddy
    else:
        train_func = train
    return train_func


def train_buddy(model, optimizer, train_loader, args, device, emb=None):
    print('starting training')
    t0 = time.time()
    model.train()
    total_loss = 0
    data = train_loader.dataset
    # hydrate edges
    links = data.links
    labels = torch.tensor(data.labels)
    # sampling
    train_samples = get_num_samples(args.train_samples, len(labels))
    sample_indices = torch.randperm(len(labels))[:train_samples]
    links = links[sample_indices]
    labels = labels[sample_indices]

    if args.wandb:
        wandb.log({"train_total_batches": len(train_loader)})
    batch_processing_times = []
    loader = DataLoader(range(len(links)), args.batch_size, shuffle=True)
    for batch_count, indices in enumerate(tqdm(loader)):
        # do node level things
        if model.node_embedding is not None:
            if args.propagate_embeddings:
                emb = model.propagate_embeddings_func(data.edge_index.to(device))
            else:
                emb = model.node_embedding.weight
        else:
            emb = None
        curr_links = links[indices]
        batch_emb = None if emb is None else emb[curr_links].to(device)

        if args.use_struct_feature:
            sf_indices = sample_indices[indices]  # need the original link indices as these correspond to sf
            subgraph_features = data.subgraph_features[sf_indices].to(device)
        else:
            subgraph_features = torch.zeros(data.subgraph_features[indices].shape).to(device)
        node_features = data.x[curr_links].to(device)
        degrees = data.degrees[curr_links].to(device)
        if args.use_RA:
            ra_indices = sample_indices[indices]
            RA = data.RA[ra_indices].to(device)
        else:
            RA = None
        start_time = time.time()
        optimizer.zero_grad()
        logits = model(subgraph_features, node_features, degrees[:, 0], degrees[:, 1], RA, batch_emb)
        loss = get_loss(args.loss)(logits, labels[indices].squeeze(0).to(device))

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * args.batch_size
        batch_processing_times.append(time.time() - start_time)

    if args.wandb:
        wandb.log({"train_batch_time": np.mean(batch_processing_times)})
        wandb.log({"train_epoch_time": time.time() - t0})

    print(f'training ran in {time.time() - t0}')

    if args.log_features:
        model.log_wandb()

    return total_loss / len(train_loader.dataset)


def train(model, optimizer, train_loader, args, device, emb=None):
    """
    Adapted version of the SEAL training function
    :param model:
    :param optimizer:
    :param train_loader:
    :param args:
    :param device:
    :param emb:
    :return:
    """

    print('starting training')
    t0 = time.time()
    model.train()
    if args.dynamic_train:
        train_samples = get_num_samples(args.train_samples, len(train_loader.dataset))
    else:
        train_samples = inf
    total_loss = 0
    pbar = tqdm(train_loader, ncols=70)
    if args.wandb:
        wandb.log({"train_total_batches": len(train_loader)})
    batch_processing_times = []
    for batch_count, data in enumerate(pbar):
        start_time = time.time()
        optimizer.zero_grad()
        # todo this loop should no longer be hit as this function isn't called for BUDDY
        if args.model == 'BUDDY':
            data_dev = [elem.squeeze().to(device) for elem in data]
            logits = model(*data_dev[:-1])  # everything but the labels
            loss = get_loss(args.loss)(logits, data[-1].squeeze(0).to(device))
        else:
            data = data.to(device)
            x = data.x if args.use_feature else None
            edge_weight = data.edge_weight if args.use_edge_weight else None
            node_id = data.node_id if emb else None
            logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id, data.src_degree,
                           data.dst_degree)
            loss = get_loss(args.loss)(logits, data.y)
        if args.l1 > 0:
            l1_reg = torch.tensor(0, dtype=torch.float)
            lin_params = torch.cat([x.view(-1) for x in model.lin.parameters()])
            for param in lin_params:
                l1_reg += torch.norm(param, 1) ** 2
            loss = loss + args.l1 * l1_reg
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * args.batch_size
        del data
        torch.cuda.empty_cache()
        batch_processing_times.append(time.time() - start_time)
        if (batch_count + 1) * args.batch_size > train_samples:
            break
    if args.wandb:
        wandb.log({"train_batch_time": np.mean(batch_processing_times)})
        wandb.log({"train_epoch_time": time.time() - t0})

    print(f'training ran in {time.time() - t0}')
    if args.model in {'linear', 'pmi', 'ra', 'aa', 'one_layer'}:
        model.print_params()

    if args.log_features:
        model.log_wandb()

    return total_loss / len(train_loader.dataset)


def train_elph(model, optimizer, train_loader, args, device):
    """
    train a GNN that calculates hashes using message passing
    @param model:
    @param optimizer:
    @param train_loader:
    @param args:
    @param device:
    @return:
    """
    print('starting training')
    t0 = time.time()
    model.train()
    total_loss = 0
    data = train_loader.dataset
    # hydrate edges
    links = data.links
    labels = torch.tensor(data.labels)
    # sampling
    train_samples = get_num_samples(args.train_samples, len(labels))
    sample_indices = torch.randperm(len(labels))[:train_samples]
    links = links[sample_indices]
    labels = labels[sample_indices]

    if args.wandb:
        wandb.log({"train_total_batches": len(train_loader)})
    batch_processing_times = []
    loader = DataLoader(range(len(links)), args.batch_size, shuffle=True)
    for batch_count, indices in enumerate(tqdm(loader)):
        # do node level things
        if model.node_embedding is not None:
            if args.propagate_embeddings:
                emb = model.propagate_embeddings_func(data.edge_index.to(device))
            else:
                emb = model.node_embedding.weight
        else:
            emb = None
        # get node features
        # TODO replace x with word_embedding generated from LM
        node_features, hashes, cards = model(data.x.to(device), data.edge_index.to(device))
        curr_links = links[indices].to(device)
        batch_node_features = None if node_features is None else node_features[curr_links]
        batch_emb = None if emb is None else emb[curr_links].to(device)
        # hydrate link features
        if args.use_struct_feature:
            subgraph_features = model.elph_hashes.get_subgraph_features(curr_links, hashes, cards).to(device)
        else:  # todo fix this
            subgraph_features = torch.zeros(data.subgraph_features[indices].shape).to(device)
        start_time = time.time()
        optimizer.zero_grad()
        logits = model.predictor(subgraph_features, batch_node_features, batch_emb)
        loss = get_loss(args.loss)(logits, labels[indices].squeeze(0).to(device))

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * args.batch_size
        batch_processing_times.append(time.time() - start_time)

    if args.wandb:
        wandb.log({"train_batch_time": np.mean(batch_processing_times)})
        wandb.log({"train_epoch_time": time.time() - t0})

    print(f'training ran in {time.time() - t0}')
    if args.model in {'linear', 'pmi', 'ra', 'aa', 'one_layer'}:
        model.print_params()

    if args.log_features:
        model.log_wandb()

    return total_loss / len(train_loader.dataset)


def auc_loss(logits, y, num_neg=1):
    pos_out = logits[y == 1]
    neg_out = logits[y == 0]
    # hack, should really pair negative and positives in the training set
    if len(neg_out) <= len(pos_out):
        pos_out = pos_out[:len(neg_out)]
    else:
        neg_out = neg_out[:len(pos_out)]
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return torch.square(1 - (pos_out - neg_out)).sum()


def bce_loss(logits, y, num_neg=1):
    return BCEWithLogitsLoss()(logits.view(-1), y.to(torch.float))


def get_loss(loss_str):
    if loss_str == 'bce':
        loss = bce_loss
    elif loss_str == 'auc':
        loss = auc_loss
    else:
        raise NotImplementedError
    return loss
