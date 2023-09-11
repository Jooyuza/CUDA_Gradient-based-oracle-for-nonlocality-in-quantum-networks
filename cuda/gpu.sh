#!/bin/bash
#SBATCH --job-name=g_0.3621_50k_0.001
#SBATCH --output=g_0.3621_100k_0.001.txt
#SBATCH --error=g_0.3621.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuze.zhu@inria.fr
#SBATCH --partition=gpu
##SBATCH --nodelist=margpu008
#SBATCH --gres=gpu:2
#SBATCH --exclusive

module load conda
source activate new
srun python ~/cuda/qu4_cuda.py
