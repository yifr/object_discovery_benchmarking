#!/bin/bash
#SBATCH --job-name simone_movi_D
#SBATCH --mail-type=END          # Mail events (NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-user=yyf@mit.edu   # Where to send mail)
#SBATCH -t 78:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=48GB
#SBATCH -p tenenbaum
#SBATCH --mem=48G
#SBATCH --out=simone_movi_d.out

python train_simone.py
