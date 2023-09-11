#!/bin/bash
#SBATCH --job-name=1cuda_try
#SBATCH --output=try.txt
#SBATCH --error=try.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuzhu@inria.fr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load conda
source activate new
srun python ~/cuda/qu4_cuda.py
