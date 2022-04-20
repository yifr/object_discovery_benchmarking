#!/bin/bash
#SBATCH --job-name slots_rgb
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-user=yyf@mit.edu   # Where to send mail)
#SBATCH -t 144:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=24GB
#SBATCH -p tenenbaum
#SBATCH --mem=48G
#SBATCH --out=recon=rgb_all_k=5.out
#SBATCH --exclude=node097,node098

run_name="recon=rgb_all_k=7"
python train.py --batch_size 16 --log_every 500 --run_name ${run_name} --recon_regime rgb --output_channels 4 --num_slots 5
