# subgraph-sketching 

Many Graph Neural Networks (GNNs) perform poorly compared to simple heuristics on Link Prediction (LP) tasks. This is due to limitations in expressive power such as the inability to count triangles (the backbone of most LP heuristics) and because they can not distinguish automorphic nodes (those having identical structural roles). Both expressiveness issues can be alleviated by learning link (rather than node) representations and incorporating structural features such as triangle counts. Since explicit link representations are often prohibitively expensive, recent
works resorted to subgraph-based methods, which have achieved state-of-the-art performance for LP, but suffer from poor efficiency due to high levels of redundancy between subgraphs. We analyze the components of subgraph GNN (SGNN) methods for link prediction. Based on our analysis, we propose a novel full-graph GNN called ELPH (Efficient Link Prediction with Hashing) that passes subgraph
sketches as messages to approximate the key components of SGNNs without explicit subgraph construction. ELPH is provably more expressive than Message Passing GNNs (MPNNs). It outperforms existing SGNN models on many standard LP benchmarks while being orders of magnitude faster. However, it shares the common GNN limitation that it is only efficient when the dataset fits in GPU memory. Accordingly, we develop a highly scalable model, called BUDDY, which uses feature precomputation to circumvent this limitation without sacrificing predictive performance. Our experiments show that BUDDY also outperforms SGNNs on standard LP benchmarks while being highly scalable and faster than ELPH.

## Introduction

This is a reimplementation of the code used for "Graph Neural Networks for Link Prediction with Subgraph Sketching" https://openreview.net/pdf?id=m1oqEOAozQU which was accepted for oral presentation (top 5% of accepted papers) at ICLR 2023.

The high level structure of the code will not change, but some details such as default parameter settings remain work in progress.

## Dataset and Preprocessing

Create a root level folder
```
./dataset
``` 
Datasets will automatically be downloaded to this folder provided you are connected to the internet.

## Running experiments

### Requirements
Dependencies (with python >= 3.9):
Main dependencies are

pytorch==1.13

torch_geometric==2.2.0

torch-scatter==2.1.1+pt113cpu

torch-sparse==0.6.17+pt113cpu

torch-spline-conv==1.2.2+pt113cpu


Example commands to install the dependencies in a new conda environment (tested on a Linux machine without GPU).
```
conda create --name ss python=3.9
conda activate ss
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
pip install torch_geometric
pip install fast-pagerank wandb datasketch ogb
```


For GPU installation (assuming CUDA 11.8): 
```
conda create --name ss python=3.9
conda activate ss
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pytorch-sparse -c pyg
conda install pyg -c pyg
pip install fast-pagerank wandb datasketch ogb
```


if you are unfamiliar with wandb, see
[wandb quickstart](https://docs.wandb.ai/quickstart)


### Experiments
To run experiments
```
cd subgraph-sketching/src
conda activate ss
python runners/run.py --dataset Cora --model ELPH
python runners/run.py --dataset Cora --model BUDDY
python runners/run.py --dataset Citeseer --model ELPH
python runners/run.py --dataset Citeseer --model BUDDY
python runners/run.py --dataset Pubmed --max_hash_hops 3 --feature_dropout 0.2 --model ELPH
python runners/run.py --dataset Pubmed --max_hash_hops 3 --feature_dropout 0.2 --model BUDDY
python runners/run.py --dataset ogbl-collab --K 50 --lr 0.01 --feature_dropout 0.05 --add_normed_features 1 --label_dropout 0.1 --batch_size 2048 --year 2007 --model ELPH
python runners/run.py --dataset ogbl-collab --K 50 --lr 0.02 --feature_dropout 0.05 --add_normed_features 1 --cache_subgraph_features --label_dropout 0.1 --year 2007 --model BUDDY
python runners/run.py --dataset ogbl-ppa --label_dropout 0.1 --use_feature 0 --use_RA 1 --lr 0.03 --epochs 100 --hidden_channels 256 --cache_subgraph_features --add_normed_features 1 --model BUDDY
python runners/run.py --dataset ogbl-ddi --K 20 --train_node_embedding --propagate_embeddings --label_dropout 0.25 --epochs 150 --hidden_channels 256 --lr 0.0015 --num_negs 6 --use_feature 0 --sign_k 2 --batch_size 131072 --model ELPH
python runners/run.py --dataset ogbl-ddi --K 20 --train_node_embedding --propagate_embeddings --label_dropout 0.25 --epochs 150 --hidden_channels 256 --lr 0.0015 --num_negs 6 --use_feature 0 --sign_k 2 --batch_size 131072 --model BUDDY
python runners/run.py --dataset ogbl-citation2 --hidden_channels 128 --num_negs 5 --sign_dropout 0.2 --sign_k 3 --cache_subgraph_features --model BUDDY
```
You may need to adjust 
```
--batch_size 
--num_workers
```
and 
```
--eval_batch_size
```

based on available (GPU) memory and CPU cores.

Most of the runtime of BUDDY is building hashes and subgraph features. If you intend to run BUDDY more than once, then set the flag
```
--cache_subgraph_features
```
to store subgraph features on disk and read them if previously cached.

To reproduce the results submited to the OGB leaderboards https://ogb.stanford.edu/docs/leader_linkprop/ add
```
--reps 10
```
to the list of command line arguments


## Cite us
If you found this work useful, please cite our paper
```
@inproceedings
{chamberlain2023graph,
  title={Graph Neural Networks for Link Prediction with Subgraph Sketching},
  author={Chamberlain, Benjamin Paul and Shirobokov, Sergey and Rossi, Emanuele and Frasca, Fabrizio and Markovich, Thomas and Hammerla, Nils and     Bronstein, Michael M and Hansmire, Max},
  booktitle={ICLR}
  year={2023}
}
```
