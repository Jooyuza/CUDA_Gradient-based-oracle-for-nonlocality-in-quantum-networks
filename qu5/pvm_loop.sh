#!/bin/bash
#SBATCH --job-name=Qu5_loop
#SBATCH --output=pvm_loop_0.3621_0.001.txt
#SBATCH --error=loop.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuze.zhu@inria.fr
#SBATCH --partition=gpu
#SBATCH --nodelist=margpu006
#SBATCH --gres=gpu:1
##SBATCH --exclusive

module load conda
source activate new
srun python ~/qu5/qu5_pvm_loop.py
