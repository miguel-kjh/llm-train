#!/bin/sh
#SBATCH --account=bsc81
#SBATCH --job-name=test
#SBATCH -q acc_bscls
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=miguel.medina@bsc.es
#SBATCH --exclusive

srun ./run.sh