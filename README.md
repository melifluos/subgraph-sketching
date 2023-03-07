# subgraph-sketching

## Introduction

Code for Graph Neural Networks for Link Prediction with Subgraph Sketching https://openreview.net/pdf?id=m1oqEOAozQU

## Running experiments

### Requirements
Dependencies (with python >= 3.7):
Main dependencies are
torch==1.8.1
torch-cluster==1.5.9
torch-geometric==1.7.0
torch-scatter==2.0.6
torch-sparse==0.6.9
torch-spline-conv==1.2.1
Commands to install all the dependencies in a new conda environment
```
conda create --name ss python=3.7
conda activate ss
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
./data folder
``` 

## Cite us
If you found this work useful, please consider citing our papers
```
@inproceedings
{chamberlain2021grand,
  title={Graph Neural Networks for Link Prediction with Subgraph Sketching},
  author={Chamberlain, Benjamin Paul and Shirobokov, Sergey and Rossi, Emanuele and Frasca, Fabrizio and Markovich, Thomas and Hammerla, Nils and     Bronstein, Michael M and Hansmire, Max},
  booktitle={ICLR}
  year={2023}
}
```
