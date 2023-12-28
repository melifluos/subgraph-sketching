"""
runners for basic GNN models
"""
import argparse
import time
import random
from math import inf
import sys

sys.path.insert(0, '..')

import torch
from tqdm import tqdm
import wandb
import numpy as np
from ogb.linkproppred import Evaluator
from src.models.gnn import GCN, SAGE, LinkPredictor
from torch_geometric.loader import DataLoader

from src.data import get_data, get_loaders
from src.evaluation import evaluate_hits, evaluate_mrr
from src.utils import select_embedding, print_model_params, str2bool
from src.wandb_setup import initialise_wandb
from src.runners.run import set_seed
from src.runners.train import bce_loss
from src.runners.inference import test
from src.graph_rewiring import EdgeBuilder


def select_model(args, num_features: int, device: torch.device):
    """
    select a GNN and configure an optimizer
    :param args: cmd line args
    :param train_dataset: pyG dataset object
    :return: a PyG model, torch optimizer
    """
    if args.model.lower() == 'sage':
        model = SAGE(num_features, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    elif args.model.lower() == 'gcn':
        model = GCN(num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
    else:
        raise NotImplementedError

    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=args.lr)
    total_params = sum(p.numel() for param in parameters for p in param)
    print(f'Total number of parameters is {total_params}')
    return model.to(device), optimizer


def get_preds(model, edges, batch_size, predictor, h, args):
    pred_list = []
    for perm in DataLoader(range(edges.size(0)), batch_size):
        edge = edges[perm].t()
        pred_list += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    preds = torch.cat(pred_list, dim=0)
    return preds


def get_eval_edges(train, val, test, sample_eval_edges=True):
    # todo refactor into tuples and test
    pos_test_edge, neg_test_edge = get_pos_neg_edges(test)
    pos_train_edge, neg_train_edge = get_pos_neg_edges(train)
    pos_val_edge, neg_val_edge = get_pos_neg_edges(val)
    if sample_eval_edges:
        n_test = pos_test_edge.shape[0]
        n_val = pos_val_edge.shape[0]
        n_train = pos_train_edge.shape[0]
        if n_val > n_test:
            indices = random.sample(range(n_val), n_test)
            pos_val_edge = pos_val_edge[indices]
            neg_val_edge = neg_val_edge[indices]
        if n_train > n_test:
            indices = random.sample(range(n_train), n_test)
            pos_train_edge = pos_train_edge[indices]
            neg_train_edge = neg_train_edge[indices]
    return pos_train_edge, neg_train_edge, pos_val_edge, neg_val_edge, pos_test_edge, neg_test_edge


def train_with_rewiring(model, optimizer, train_loader, args, device):
    """
    A train function that uses a different graph at each batch for convolutions and possibly also structure features
    @return:
    """
    edge_builder = EdgeBuilder(args)
    print('starting training')
    t0 = time.time()
    model.train()
    total_loss = 0
    data = train_loader.dataset
    # hydrate edges
    links = data.links
    labels = torch.tensor(data.labels)
    loader = DataLoader(range(len(links)), args.batch_size, shuffle=True)
    for batch_count, indices in enumerate(tqdm(loader)):
        curr_links = links[indices].to(device)
        curr_labels = labels[indices].to(device)
        # get node features
        edge_index, _ = edge_builder.rewire_graph(curr_links, curr_labels, data.edge_index, data.edge_weight)
        node_features = model(data.x.to(device), edge_index.to(device))
        # make edgewise predictions
        batch_node_features = None if node_features is None else node_features[curr_links]
        optimizer.zero_grad()
        logits = model.predictor(batch_node_features)
        loss = bce_loss(logits, curr_labels.squeeze(0).to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * args.batch_size
    if args.wandb:
        wandb.log({"train_epoch_time": time.time() - t0})
    print(f'training ran in {time.time() - t0}')
    return total_loss / len(train_loader.dataset)


def run(args):
    args = initialise_wandb(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"executing on {device}")
    results_list = []
    for rep in range(args.reps):
        set_seed(rep)
        dataset, splits, directed, eval_metric = get_data(args)
        train_loader, train_eval_loader, val_loader, test_loader = get_loaders(args, dataset, splits, directed)
        # GNNs require an additional predictor NN to map from embeddings to edge probabilities
        predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                                  args.num_layers, args.dropout).to(device)
        if args.dataset_name.startswith('ogbl'):
            evaluator = Evaluator(name=args.dataset_name)
        else:
            evaluator = Evaluator(name='ogbl-ppa')  # this says use HR@100 as the metric

        model, optimizer = select_model(args, dataset.num_features, device)
        train_res = val_res = test_res = best_epoch = 0
        print(f'running repetition {rep}')
        if rep == 0:
            print_model_params(model)
        for epoch in range(args.epochs):
            t0 = time.time()
            loss = train_with_rewiring(model, optimizer, train_loader, args, device)
            if epoch % args.eval_steps == 0:
                results = test(model, evaluator, train_eval_loader, val_loader, test_loader, args, device)
                for key, result in results.items():
                    train_res, tmp_val_res, tmp_test_res = result
                    if tmp_val_res > val_res:
                        val_res = tmp_val_res
                        test_res = tmp_test_res
                        best_epoch = epoch
                    res_dic = {f'rep{rep}_loss': loss, f'rep{rep}_Train' + key: 100 * train_res,
                               f'rep{rep}_Val' + key: 100 * val_res, f'rep{rep}_tmp_val' + key: 100 * tmp_val_res,
                               f'rep{rep}_tmp_test' + key: 100 * tmp_test_res,
                               f'rep{rep}_Test' + key: 100 * test_res, f'rep{rep}_best_epoch': best_epoch,
                               f'rep{rep}_epoch_time': time.time() - t0, 'epoch_step': epoch}
                    if args.wandb:
                        wandb.log(res_dic)
                    to_print = f'Epoch: {epoch:02d}, Best epoch: {best_epoch}, Loss: {loss:.4f}, Train: {100 * train_res:.2f}%, Valid: ' \
                               f'{100 * val_res:.2f}%, Test: {100 * test_res:.2f}%, epoch time: {time.time() - t0:.1f}'
                    print(key)
                    print(to_print)
        if args.reps > 1:
            results_list.append([test_res, val_res, train_res])
    if args.reps > 1:
        test_acc_mean, val_acc_mean, train_acc_mean = np.mean(results_list, axis=0) * 100
        test_acc_std = np.sqrt(np.var(results_list, axis=0)[0]) * 100
        wandb_results = {'test_mean': test_acc_mean, 'val_mean': val_acc_mean, 'train_mean': train_acc_mean,
                         'test_acc_std': test_acc_std}
        print(wandb_results)
        if args.wandb:
            wandb.log(wandb_results)
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    # data settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Cora',
                        choices=['Cora', 'producer', 'Citeseer', 'Pubmed', 'ogbl-ppa', 'ogbl-collab', 'ogbl-ddi',
                                 'ogbl-citation2'])
    parser.add_argument('--train_samples', type=float, default=inf, help='the number of training edges or % if < 1')
    parser.add_argument('--val_samples', type=float, default=inf, help='the number of val edges or % if < 1')
    parser.add_argument('--test_samples', type=float, default=inf, help='the number of test edges or % if < 1')
    parser.add_argument('--val_pct', type=float, default=0.1,
                        help='the percentage of supervision edges to be used for validation. These edges will not appear'
                             ' in the training set and will only be used as message passing edges in the test set')
    parser.add_argument('--test_pct', type=float, default=0.2,
                        help='the percentage of supervision edges to be used for test. These edges will not appear'
                             ' in the training or validation sets for either supervision or message passing')
    parser.add_argument('--pos_enc_dim', type=int, default=2, help='add a positional encoding')
    parser.add_argument('--add_pos_enc', action='store_true', help='add a positional encoding')
    parser.add_argument('--connected_holdout', action='store_true',
                        help='ensure a single connected component after splits')
    parser.add_argument('--sign_k', type=int, default=0)
    # GNN settings
    parser.add_argument('--model', type=str, default='GCN',
                        choices=('GCN', 'SAGE'))
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--use_feature', action='store_true',
                        help="whether to use raw node features as GNN input")
    # Training settings
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--edge_dropout', type=float, default=0)
    parser.add_argument('--remove_target_links', type=str2bool, default=1)
    parser.add_argument('--add_negative_links', type=str2bool, default=1)
    parser.add_argument('--drop_message_passing_links', type=str2bool, default=1)
    # test settings
    parser.add_argument('--K', type=int, default=100, help='the @k to calculate hitrate and ndcg')
    parser.add_argument('--reps', type=int, default=1, help='the number of repetition of the experiment to run')
    parser.add_argument('--eval_batch_size', type=int, default=1000000)
    # Compute settings
    parser.add_argument('--num_workers', type=int, default=1)
    # wandb settings
    parser.add_argument('--wandb', action='store_true', help="flag if logging to wandb")
    parser.add_argument('-wandb_offline', dest='use_wandb_offline',
                        action='store_true')  # https://docs.wandb.ai/guides/technical-faq
    parser.add_argument('--wandb_sweep', action='store_true',
                        help="flag if sweeping")  # if not it picks up params in greed_params
    parser.add_argument('--wandb_watch_grad', action='store_true', help='allows gradient tracking in train function')
    parser.add_argument('--wandb_track_grad_flow', action='store_true')

    parser.add_argument('--wandb_entity', default="link-prediction", type=str)
    parser.add_argument('--wandb_project', default="link-prediction", type=str)
    parser.add_argument('--wandb_group', default="testing", type=str, help="testing,tuning,eval")
    parser.add_argument('--wandb_run_name', default=None, type=str)
    parser.add_argument('--wandb_output_dir', default='./wandb_output',
                        help='folder to output results, images and model checkpoints')
    parser.add_argument('--wandb_log_freq', type=int, default=1, help='Frequency to log metrics.')
    parser.add_argument('--wandb_epoch_list', nargs='+', default=[0, 1, 2, 4, 8, 16],
                        help='list of epochs to log gradient flow')

    args = parser.parse_args()
    print(args)
    run(args)
