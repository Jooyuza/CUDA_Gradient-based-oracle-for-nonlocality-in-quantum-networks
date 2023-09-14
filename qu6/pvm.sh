#!/bin/bash
#SBATCH --job-name=0.363_r18
#SBATCH --output=pvm_0.363_r18_0.01.txt
#SBATCH --error=pvm_r18_0.363.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuze.zhu@inria.fr
#SBATCH --partition=gpu
#SBATCH --nodelist=margpu009
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=80
##SBATCH --exclusive

module load conda
source activate new
srun python ~/qu6/qu6_pvm.py
