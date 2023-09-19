"""
main module
"""
import argparse
import time
import warnings
from math import inf
import random
import numpy as np
import torch
from ogb.linkproppred import Evaluator
torch.set_printoptions(precision=4)
import wandb
# when generating subgraphs the supervision edge is deleted, which triggers a SparseEfficiencyWarning, but this is
# not a performance bottleneck, so suppress for now
from scipy.sparse import SparseEfficiencyWarning
import sys, os
sys.path.insert(0, '..')

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

from src.data import get_data, get_loaders
from src.models.elph import ELPH, BUDDY
from src.models.seal import SEALDGCNN, SEALGCN, SEALGIN, SEALSAGE
from src.utils import ROOT_DIR, print_model_params, select_embedding, str2bool
from src.wandb_setup import initialise_wandb
from src.runners.train import get_train_func
from src.runners.inference import test

def print_results_list(results_list):
    for idx, res in enumerate(results_list):
        print(f'repetition {idx}: test {res[0]:.2f}, val {res[1]:.2f}, train {res[2]:.2f}')

def set_seed(seed):
    """
    setting a random seed for reproducibility and in accordance with OGB rules
    @param seed: an integer seed
    @return: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run(args):
    args = initialise_wandb(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"executing on {device}")
    results_list = []
    train_func = get_train_func(args)
    for rep in range(args.reps):
        set_seed(rep)
        dataset, splits, directed, eval_metric = get_data(args)
        train_loader, train_eval_loader, val_loader, test_loader = get_loaders(args, dataset, splits, directed)
        if args.dataset_name.startswith('ogbl'):  # then this is one of the ogb link prediction datasets
            evaluator = Evaluator(name=args.dataset_name)
        else:
            evaluator = Evaluator(name='ogbl-ppa')  # this sets HR@100 as the metric
        
        emb = select_embedding(args, dataset.data.num_nodes, device)
        model, optimizer = select_model(args, dataset, emb, device)
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
                    to_print = f'Epoch: {epoch:02d}, Best epoch: {best_epoch}, Loss: {loss:.4f}, Train: {100 * train_res:.2f}%, Valid: ' \
                               f'{100 * val_res:.2f}%, Test: {100 * test_res:.2f}%, epoch time: {time.time() - t0:.1f}'
                    print(key)
                    print(to_print)
        if args.reps > 1:
            results_list.append([test_res, val_res, train_res])
            print_results_list(results_list)
    if args.reps > 1:
        test_acc_mean, val_acc_mean, train_acc_mean = np.mean(results_list, axis=0) * 100
        test_acc_std = np.sqrt(np.var(results_list, axis=0)[0]) * 100
        val_acc_std = np.sqrt(np.var(results_list, axis=0)[1]) * 100

        wandb_results = {'test_mean': test_acc_mean, 'val_mean': val_acc_mean, 'train_mean': train_acc_mean,
                         'test_acc_std': test_acc_std, 'val_acc_std': val_acc_std}
        print(wandb_results)
        if args.wandb:
            wandb.log(wandb_results)
    if args.wandb:
        wandb.finish()
    if args.save_model:
        path = f'{ROOT_DIR}/saved_models/{args.dataset_name}'
        torch.save(model.state_dict(), path)


def select_model(args, dataset, emb, device):
    if args.model == 'SEALDGCNN':
        model = SEALDGCNN(args.hidden_channels, args.num_seal_layers, args.max_z, args.sortpool_k,
                          dataset, args.dynamic_train, use_feature=args.use_feature,
                          node_embedding=emb).to(device)
    elif args.model == 'SEALSAGE':
        model = SEALSAGE(args.hidden_channels, args.num_seal_layers, args.max_z, dataset.num_features,
                         args.use_feature, node_embedding=emb, dropout=args.dropout).to(device)
    elif args.model == 'SEALGCN':
        model = SEALGCN(args.hidden_channels, args.num_seal_layers, args.max_z, dataset.num_features,
                        args.use_feature, node_embedding=emb, dropout=args.dropout, pooling=args.seal_pooling).to(
            device)
    elif args.model == 'SEALGIN':
        model = SEALGIN(args.hidden_channels, args.num_seal_layers, args.max_z, dataset.num_features,
                        args.use_feature, node_embedding=emb, dropout=args.dropout).to(device)
    elif args.model == 'BUDDY':
        model = BUDDY(args, dataset.num_features, node_embedding=emb).to(device)
    elif args.model == 'ELPH':
        model = ELPH(args, dataset.num_features, node_embedding=emb).to(device)
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


if __name__ == '__main__':
    # Data settings
    parser = argparse.ArgumentParser(description='Efficient Link Prediction with Hashes (ELPH)')
    parser.add_argument('--dataset_name', type=str, default='Cora',
                        choices=['cora', 'Citeseer', 'pubmed', 'ogbl-ppa', 'ogbl-collab', 'ogbl-ddi',
                                 'ogbl-citation2', 'ogbn-arxiv'])
    parser.add_argument('--use_text', type=bool, default=False,
                        help='whether to use text features')
    
    parser.add_argument('--val_pct', type=float, default=0.1,
                        help='the percentage of supervision edges to be used for validation. These edges will not appear'
                             ' in the training set and will only be used as message passing edges in the test set')
    parser.add_argument('--test_pct', type=float, default=0.2,
                        help='the percentage of supervision edges to be used for test. These edges will not appear'
                             ' in the training or validation sets for either supervision or message passing')
    parser.add_argument('--train_samples', type=float, default=inf, help='the number of training edges or % if < 1')
    parser.add_argument('--val_samples', type=float, default=inf, help='the number of val edges or % if < 1')
    parser.add_argument('--test_samples', type=float, default=inf, help='the number of test edges or % if < 1')
    parser.add_argument('--preprocessing', type=str, default=None)
    parser.add_argument('--sign_k', type=int, default=0)
    parser.add_argument('--load_features', action='store_true', help='load node features from disk')
    parser.add_argument('--load_hashes', action='store_true', help='load hashes from disk')
    parser.add_argument('--cache_subgraph_features', action='store_true',
                        help='write / read subgraph features from disk')
    parser.add_argument('--train_cache_size', type=int, default=inf, help='the number of training edges to cache')
    parser.add_argument('--year', type=int, default=0, help='filter training data from before this year')
    # GNN settings
    parser.add_argument('--model', type=str, default='BUDDY')
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eval_batch_size', type=int, default=1000000,
                        help='eval batch size should be largest the GPU memory can take - the same is not necessarily true at training time')
    parser.add_argument('--label_dropout', type=float, default=0.5)
    parser.add_argument('--feature_dropout', type=float, default=0.5)
    parser.add_argument('--sign_dropout', type=float, default=0.5)
    parser.add_argument('--save_model', action='store_true', help='save the model to use later for inference')
    parser.add_argument('--feature_prop', type=str, default='gcn',
                        help='how to propagate ELPH node features. Values are gcn, residual (resGCN) or cat (jumping knowledge networks)')
    # SEAL settings
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_seal_layers', type=int, default=3)
    parser.add_argument('--sortpool_k', type=float, default=0.6)
    parser.add_argument('--label_pooling', type=str, default='add', help='add or mean')
    parser.add_argument('--seal_pooling', type=str, default='edge', help='how SEAL pools features in the subgraph')
    # Subgraph settings
    parser.add_argument('--num_hops', type=int, default=1)
    parser.add_argument('--ratio_per_hop', type=float, default=1.0)
    parser.add_argument('--max_nodes_per_hop', type=int, default=None)
    parser.add_argument('--node_label', type=str, default='drnl')
    parser.add_argument('--max_dist', type=int, default=4)
    parser.add_argument('--max_z', type=int, default=1000,
                        help='the size of the label embedding table. ie. the maximum number of labels possible')
    parser.add_argument('--use_feature', type=str2bool, default=True,
                        help="whether to use raw node features as GNN input")
    parser.add_argument('--use_struct_feature', type=str2bool, default=True,
                        help="whether to use structural graph features as GNN input")
    parser.add_argument('--use_edge_weight', action='store_true',
                        help="whether to consider edge weight in GNN")
    # Training settings
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimization')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_negs', type=int, default=1, help='number of negatives for each positive')
    parser.add_argument('--train_node_embedding', action='store_true',
                        help="also train free-parameter node embeddings together with GNN")
    parser.add_argument('--propagate_embeddings', action='store_true',
                        help='propagate the node embeddings using the GCN diffusion operator')
    parser.add_argument('--loss', default='bce', type=str, help='bce or auc')
    parser.add_argument('--add_normed_features', dest='add_normed_features', type=str2bool,
                        help='Adds a set of features that are normalsied by sqrt(d_i*d_j) to calculate cosine sim')
    parser.add_argument('--use_RA', type=str2bool, default=False, help='whether to add resource allocation features')
    # SEAL specific args
    parser.add_argument('--dynamic_train', action='store_true',
                        help="dynamically extract enclosing subgraphs on the fly")
    parser.add_argument('--dynamic_val', action='store_true')
    parser.add_argument('--dynamic_test', action='store_true')
    parser.add_argument('--pretrained_node_embedding', type=str, default=None,
                        help="load pretrained node embeddings as additional node features")
    # Testing settings
    parser.add_argument('--reps', type=int, default=1, help='the number of repetition of the experiment to run')
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--eval_metric', type=str, default='hits',
                        choices=('hits', 'mrr', 'auc'))
    parser.add_argument('--K', type=int, default=100, help='the hit rate @K')
    # hash settings
    parser.add_argument('--use_zero_one', type=str2bool, default=0,
                        help="whether to use the counts of (0,1) and (1,0) neighbors")
    parser.add_argument('--floor_sf', type=str2bool, default=0,
                        help='the subgraph features represent counts, so should not be negative. If --floor_sf the min is set to 0')
    parser.add_argument('--hll_p', type=int, default=8, help='the hyperloglog p parameter')
    parser.add_argument('--minhash_num_perm', type=int, default=128, help='the number of minhash perms')
    parser.add_argument('--max_hash_hops', type=int, default=2, help='the maximum number of hops to hash')
    parser.add_argument('--subgraph_feature_batch_size', type=int, default=11000000,
                        help='the number of edges to use in each batch when calculating subgraph features. '
                             'Reduce or this or increase system RAM if seeing killed messages for large graphs')
    # wandb settings
    parser.add_argument('--wandb', action='store_true', help="flag if logging to wandb")
    parser.add_argument('--wandb_offline', dest='use_wandb_offline',
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
    parser.add_argument('--log_features', action='store_true', help="log feature importance")
    args = parser.parse_args()
    if (args.max_hash_hops == 1) and (not args.use_zero_one):
        print("WARNING: (0,1) feature knock out is not supported for 1 hop. Running with all features")
    if args.dataset_name == 'ogbl-ddi':
        args.use_feature = 0  # dataset has no features
        assert args.sign_k > 0, '--sign_k must be set to > 0 i.e. 1,2 or 3 for ogbl-ddi'
    print(args)
    run(args)
