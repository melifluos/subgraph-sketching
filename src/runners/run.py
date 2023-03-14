"""
main module
"""
import argparse
import time
import warnings
from math import inf

import numpy as np
import torch
from ogb.linkproppred import Evaluator
from torch_geometric.loader import DataLoader as pygDataLoader
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor

torch.set_printoptions(precision=4)
import wandb
# when generating subgraphs the supervision edge is deleted, which triggers a SparseEfficiencyWarning, but this is
# not a performance bottleneck, so suppress for now
from scipy.sparse import SparseEfficiencyWarning
from tqdm import tqdm

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

from data import get_data, pre_propagate_features, get_loaders
from evaluation import evaluate_auc, evaluate_hits, evaluate_mrr
from datasets.seal import get_train_val_test_datasets
from datasets.elph import get_hashed_train_val_test_datasets, make_train_eval_data
from models.elph import ELPH, ELP, ELPHGNN
from models.seal import SEALDGCNN, SEALGCN, SEALGIN, SEALSAGE
from utils import ROOT_DIR, print_model_params, select_embedding, str2bool
from wandb_setup import initialise_wandb
from train import get_train_func
from inference import test

def run(args):
    args = initialise_wandb(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # todo move the next 3 lines somewhere less obnoxious
    if (args.max_hash_hops == 1) and (not args.use_zero_one):
        print("WARNING: (0,1) feature knock out is not supported for 1 hop. Running with all features")
    print(f"executing on {device}")
    results_list = []
    train_func = get_train_func(args)
    for rep in range(args.reps):
        dataset, splits, directed, eval_metric = get_data(args)
        train_loader, train_eval_loader, val_loader, test_loader = get_loaders(args, dataset, splits, directed)
        train_data, val_data, test_data = splits['train'], splits['valid'], splits['test']
        if args.dataset_name.startswith('ogbl'):
            evaluator = Evaluator(name=args.dataset_name)
        else:
            evaluator = Evaluator(name='ogbl-ppa')  # this sets HR@100 as the metric
        emb = select_embedding(args, train_data.num_nodes, device)
        model, optimizer = select_model(args, dataset, args.max_z, emb, device)
        val_res = test_res = best_epoch = 0
        print(f'running repetition {rep}')
        if rep == 0:
            print_model_params(model)
        for epoch in range(args.epochs):
            t0 = time.time()
            loss = train_func(model, optimizer, train_loader, args, device)
            if (epoch + 1) % args.eval_steps == 0:
                results = test(model, evaluator, train_eval_loader, val_loader, test_loader, args, device,
                               eval_metric=eval_metric)
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
                    to_print = f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * train_res:.2f}%, Valid: ' \
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
        if args.save_model:
            path = f'{ROOT_DIR}/saved_models/{args.dataset_name}'
            torch.save(model.state_dict(), path)


def select_model(args, dataset, max_z, emb, device):
    if args.model == 'SEALDGCNN':
        model = SEALDGCNN(args.hidden_channels, args.num_layers, max_z, args.sortpool_k,
                          dataset, args.dynamic_train, use_feature=args.use_feature,
                          node_embedding=emb).to(device)
    elif args.model == 'SEALSAGE':
        model = SEALSAGE(args.hidden_channels, args.num_layers, max_z, dataset.num_features,
                         args.use_feature, node_embedding=emb, dropout=args.dropout).to(device)
    elif args.model == 'SEALGCN':
        model = SEALGCN(args.hidden_channels, args.num_layers, max_z, dataset.num_features,
                        args.use_feature, node_embedding=emb, dropout=args.dropout, pooling=args.seal_pooling).to(
            device)
    elif args.model == 'SEALGIN':
        model = SEALGIN(args.hidden_channels, args.num_layers, max_z, dataset.num_features,
                        args.use_feature, node_embedding=emb, dropout=args.dropout).to(device)
    elif args.model == 'BUDDY':
        model = ELPH(args, dataset.num_features, node_embedding=emb).to(device)
    elif args.model == 'ELPH':
        model = ELPHGNN(args, dataset.num_features, node_embedding=emb).to(device)
    else:
        raise NotImplementedError
    parameters = list(model.parameters())
    if args.train_node_embedding:
        torch.nn.init.xavier_uniform_(emb.weight)
        parameters += list(emb.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=args.weight_decay)
    total_params = sum(p.numel() for param in parameters for p in param)
    print(f'Total number of parameters is {total_params}')
    if args.model == 'DGCNN':
        print(f'SortPooling k is set to {model.k}')
    return model, optimizer
