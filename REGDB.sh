#!/bin/bash
#SBATCH -J pytorch 
#SBATCH --wckey=t2024035
#SBATCH -p gpu   
#SBATCH -N 1  
#SBATCH -n 8  
#SBATCH --gres=gpu:1  
source /share/apps/anaconda3/etc/profile.d/conda.sh 
conda activate EOT
python train.py --config_file config/RegDB.yml