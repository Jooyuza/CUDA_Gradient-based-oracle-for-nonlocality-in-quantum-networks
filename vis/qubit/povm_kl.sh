#!/bin/bash
#SBATCH --job-name=v0.1_kl
#SBATCH --output=v0.1_kl.txt
#SBATCH --error=kl.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuze.zhu@inria.fr
#SBATCH --partition=gpu
#SBATCH --nodelist=margpu008
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=60
##SBATCH --exclusive

module load conda
source activate new
srun python ~/vis/qubit/vis_kl.py
