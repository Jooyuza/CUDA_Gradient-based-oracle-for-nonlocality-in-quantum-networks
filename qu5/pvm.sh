#!/bin/bash
#SBATCH --job-name=0.363_r12
#SBATCH --output=pvm_0.363_r12_0.01.txt
#SBATCH --error=pvm_r12_0.363.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuze.zhu@inria.fr
#SBATCH --partition=gpu
#SBATCH --nodelist=margpu008
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=60
##SBATCH --exclusive

module load conda
source activate new
srun python ~/qu5/qu5_pvm.py
