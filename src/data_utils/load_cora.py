import numpy as np
import torch
import random
import os, sys 
sys.path.insert(0, os.getcwd()+'/src')
import pandas as pd 
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from pdb import set_trace as bp
# param
from configs.config_load import update_cfg, cfg

# return cora dataset as pytorch geometric Data object together with 60/20/20 split, and list of cora IDs


def get_cora_casestudy(SEED=0):
    data_X, data_Y, data_citeid, data_edges = parse_cora(cfg)
    # data_X = sklearn.preprocessing.normalize(data_X, norm="l1")
    bp()
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    # load data
    data_name = 'cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('dataset', data_name,
                        transform=T.NormalizeFeatures())
    data = dataset[0]

    data.x = torch.tensor(data_X).float()
    data.edge_index = torch.tensor(data_edges).long()
    data.y = torch.tensor(data_Y).long()
    data.num_nodes = len(data_Y)

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])

    return data, data_citeid


# credit: https://github.com/tkipf/pygcn/issues/27, xuhaiyun


def parse_cora(cfg):
    path = cfg.dataset.cora.original
    idx_features_labels = np.genfromtxt(
        "{}.content".format(path), dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                             'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning',
                                             'Theory'])}
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        "{}.cites".format(path), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    del path, idx_features_labels, labels, edges_unordered, edges
    return data_X, data_Y, data_citeid, np.unique(data_edges, axis=0).transpose()


def get_raw_text_cora_from_rpo(use_text=False, seed=0):
    """adopted from repo: 
    https://github.com/XiaoxinHe/TAPE/blob/241c93b735dcebbe2853414395c1559d5c2ce202/core/data_utils/load_cora.py
    """
    data, data_citeid = get_cora_casestudy(seed)
    if not use_text:
        return data, None

    with open('dataset/cora_orig/mccallum/cora/papers') as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split('\t')[0]
        fn = line.split('\t')[1]
        pid_filename[pid] = fn

    path = 'dataset/cora_andrew_mccallum/extractions/'
    text = []
    for pid in data_citeid:
        fn = pid_filename[pid]
        with open(path + fn) as f:
            lines = f.read().splitlines()

        for line in lines:
            if 'Title:' in line:
                ti = line
            if 'Abstract:' in line:
                ab = line
        text.append(ti + '\n' + ab)
    return data, text


def get_raw_text_cora(cfg, use_text=False, seed=0):
    path_papers = cfg.dataset.cora.papers
    andrew_maccallum_path = cfg.dataset.cora.extractions 
    # path = 'dataset/cora_orig/mccallum/cora/extractions/'
    data, data_citeid = get_cora_casestudy(seed)
    if not use_text:
        return data, None

    with open(path_papers) as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split('\t')[0]
        fn = line.split('\t')[1]
        pid_filename[pid] = fn

    text = []
    whole, founded = len(data_citeid), 0
    no_ab_or_ti = 0
    for pid in data_citeid:
        fn = pid_filename[pid]
        ti, ab = load_ab_ti(andrew_maccallum_path, fn)
        founded += 1
        text.append(ti + '\n' + ab)

        if ti == '' or ab == '':
            # print(f"no title {ti}, no abstract {ab}")
            no_ab_or_ti += 1
    print(f"found {founded}/{whole} papers, {no_ab_or_ti} no ab or ti.")
    return data, text


def load_ab_ti(path, fn):
    ti, ab = '', ''
    with open(path + fn) as f:
        lines = f.read().splitlines()
    for line in lines:
        if line.split(':')[0] == 'Title':
            ti = line
        elif line.split(':')[0] == 'Abstract':
            ab = line
    return ti, ab




def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    TRAINER = DGLGNNTrainer if cfg.gnn.train.use_dgl else GNNTrainer
    all_acc = []
    for seed in seeds:
        cfg.seed = seed
        trainer = TRAINER(cfg, cfg.gnn.train.feature_type)
        trainer.train()
        _, acc = trainer.eval_and_save()
        all_acc.append(acc)

    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        print(f"[{cfg.gnn.train.feature_type}] ValACC: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}, TestAcc: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}")


if __name__ == '__main__':
    cfg = update_cfg(cfg)
    data, data_citeid = get_cora_casestudy(cfg.seed)
    data, text = get_raw_text_cora(cfg, use_text=True)
    print(data)
    print(data_citeid)
    print(text)




