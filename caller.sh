#!/bin/sh
#SBATCH --account=bsc81
#SBATCH --job-name=test
#SBATCH -q acc_bscls
#SBATCH -c 80
#SBATCH --gres=gpu:4
#SBATCH -N 8
#SBATCH -t 00-35:00:00
#SBATCH --exclusive
#SBATCH --mail-type=all
#SBATCH --mail-user=miguel.medina@bsc.es
#SBATCH --exclusive

srun ./run.sh