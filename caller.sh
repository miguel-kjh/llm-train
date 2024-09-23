#!/bin/sh
#SBATCH --account=bsc81
#SBATCH --job-name=test
#SBATCH -q acc_bscls
#SBATCH -N 1
#SBATCH -c 80
#SBATCH --gres=gpu:1
#SBATCH -t 00-48:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=miguel.medina@bsc.es
#SBATCH --exclusive

srun ./run.sh