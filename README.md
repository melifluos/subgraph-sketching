# subgraph-sketching

## Introduction

Code for Graph Neural Networks for Link Prediction with Subgraph Sketching https://openreview.net/pdf?id=m1oqEOAozQU

## Running experiments

### Requirements
Dependencies (with python >= 3.10):
Main dependencies are

pytorch==1.13

pyg==2.2

Commands to install all the dependencies in a new conda environment
```
conda create --name ss python=3.7
conda activate ss
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pyg -c pyg
```


### Experiments
For example to run for Cora with random splits:
```
cd src
python run.py --dataset Cora 
```

### Dataset and Preprocessing

Create a root level 
```
./dataset folder
``` 

## Cite us
If you found this work useful, please consider citing our papers
```
@inproceedings
{chamberlain2023graph,
  title={Graph Neural Networks for Link Prediction with Subgraph Sketching},
  author={Chamberlain, Benjamin Paul and Shirobokov, Sergey and Rossi, Emanuele and Frasca, Fabrizio and Markovich, Thomas and Hammerla, Nils and     Bronstein, Michael M and Hansmire, Max},
  booktitle={ICLR}
  year={2023}
}
```
