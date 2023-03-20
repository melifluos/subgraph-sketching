"""
testing data reader and preprocessing

Store the global parameter dictionary to be imported and modified by each test
"""
import random
import torch
import numpy as np
from math import inf

OPT = {'sample_size': None, 'dataset_name': 'Cora', 'num_hops': 2, 'max_dist': 10, 'max_nodes_per_hop': 10,
       'data_appendix': None, 'val_pct': 0.1, 'test_pct': 0.2, 'train_sample': 1, 'dynamic_train': True,
       'dynamic_val': True, 'model': 'BUDDY', 'sign_k': 2, 'loss': 'bce', 'log_features': False,
       'dynamic_test': True, 'node_label': 'drnl', 'ratio_per_hop': 1, 'use_feature': True, 'dropout': 0,
       'label_dropout': 0, 'feature_dropout': 0,
       'add_normed_features': False, 'use_RA': False, 'hidden_channels': 32, 'load_features': True,
       'load_hashes': True, 'use_zero_one': True, 'wandb': False, 'batch_size': 32, 'num_workers': 1,
       'cache_train_structure_features': False, 'cache_val_structure_features': False,
       'cache_test_structure_features': False, 'lr': 0.1, 'weight_decay': 0, 'eval_batch_size': 100,
       'propagate_embeddings': False, 'num_negs': 1,
       'sign_dropout': 0.5, 'sign_k': 2, 'use_struct_feature': True, 'max_hash_hops': 2, 'hll_p': 8,
       'minhash_num_perm': 128, 'floor_sf': False, 'year': 0, 'feature_prop': 'gcn', 'train_node_embeddings': False,
       'train_samples': inf, 'val_samples': inf, 'test_samples': inf, 'reps': 1, 'train_node_embedding': False,
       'pretrained_node_embedding': False, 'max_z': 1000, 'eval_steps': 1, 'K': 100, 'save_model': False}


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
