#!/bin/sh
#SBATCH --account=bsc81
#SBATCH --job-name=test
#SBATCH -q acc_bscls
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 02-00:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=miguel.medina@bsc.es
#SBATCH --exclusive

srun -n 1 -c 1 ./run.sh