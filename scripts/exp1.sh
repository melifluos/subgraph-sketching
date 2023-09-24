# Description

# Experiment 1
# head 1 
python runners/run.py --dataset_name pubmed --model ELPH --use_text False --wandb_run_name pubmed_elph
python runners/run.py --dataset_name cora --model ELPH --use_text False --wandb_run_name cora_elph
## not work 
python runners/run.py --dataset_name ogbn-arxiv --model ELPH --use_text False --wandb_run_name ogbn-arxiv_elph
python runners/run.py --dataset_name pubmed --model BUDDY --use_text False --wandb_run_name pubmed_elph
python runners/run.py --dataset_name cora --model BUDDY --use_text False --wandb_run_name cora_elph
## not work 
python runners/run.py --dataset_name ogbn-arxiv --model BUDDY --use_text False --wandb_run_name ogbn-arxiv_elph

# head 2 
python runners/run.py --dataset_name pubmed --model ELPH --use_text True --wandb_run_name pubmed_elph_text
python runners/run.py --dataset_name cora --model ELPH --use_text True  --wandb_run_name cora_elph_text
## not work 
python runners/run.py --dataset_name ogbn-arxiv --model ELPH --use_text True  --wandb_run_name ogbn-arxiv_elph_text
python runners/run.py --dataset_name pubmed --model BUDDY --use_text True --wandb_run_name pubmed_elph_text
python runners/run.py --dataset_name cora --model BUDDY --use_text True  --wandb_run_name cora_elph_text
## not work 
python runners/run.py --dataset_name ogbn-arxiv --model BUDDY --use_text True  --wandb_run_name ogbn-arxiv_elph_text

