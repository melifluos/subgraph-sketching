# subgraph-sketching (Work in Progress)

## Introduction

This is a reimplementation of the code used for "Graph Neural Networks for Link Prediction with Subgraph Sketching" https://openreview.net/pdf?id=m1oqEOAozQU which was accepted for oral presentation (top 5% of accepted papers) at ICLR 2023.

The high level structure of the code will not change, but some details such as default parameter setting remain work in progress

## Running experiments

### Requirements
Dependencies (with python >= 3.9):
Main dependencies are

pytorch==1.13

torch_geometric==2.2.0

torch-scatter==2.1.1+pt113cpu

torch-sparse==0.6.17+pt113cpu

torch-spline-conv==1.2.2+pt113cpu


Example commands to install the dependencies in a new conda environment (tested on Linux machine without GPU).
```
conda create --name ss python=3.9
conda activate ss
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
pip install torch_geometric
pip install fast-pagerank wandb datasketch ogb
```


For GPU installation: 
```
conda create --name ss python=3.9
conda activate ss
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install torch_geometric
pip install fast-pagerank wandb datasketch ogb
```


if you are unfamiliar with wandb, quickstart instructions are
[pip install wandb](https://docs.wandb.ai/quickstart)


### Experiments
To run experiments
```
cd src
python runners/run.py --dataset Cora --model ELPH
python runners/run.py --dataset Citeseer --model ELPH
python runners/run.py --dataset Cora --model BUDDY
python runners/run.py --dataset Citeseer --model BUDDY
```

### Dataset and Preprocessing

Create a root level folder
```
./dataset
``` 

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
