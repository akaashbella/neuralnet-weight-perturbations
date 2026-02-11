#!/bin/bash
#SBATCH --job-name=wn_resnet20
#SBATCH --partition=dgxh
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bellaak@oregonstate.edu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd ~/hpc-share/research/neuralnet-weight-perturbations
source venv/bin/activate

python -u run_experiments.py --dataset cifar10 resnet20
