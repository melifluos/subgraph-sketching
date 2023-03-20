# subgraph-sketching (Work in Progress)

## Introduction

This is a reimplementation of the code used for "Graph Neural Networks for Link Prediction with Subgraph Sketching" https://openreview.net/pdf?id=m1oqEOAozQU which was accepted for oral presentation (top 5% of accepted papers) at ICLR 2023.

The high level structure of the code will not change, but some details such as default parameter setting remain work in progress

## Running experiments

### Requirements
Dependencies (with python >= 3.10):
Main dependencies are

pytorch==1.13

pyg==2.2

wandb==0.13.9 (for logging and tuning)

datasketch==1.5.9

fast-pagerank==0.0.4 (only to run PPR baselines)

scipy==1.10.0

Example commands to install the dependencies in a new conda environment
```
conda create --name ss python=3.10
conda activate ss
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pyg -c pyg
conda install -c anaconda scipy
pip install fast-pagerank
pip install wandb
pip install datasketch
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
