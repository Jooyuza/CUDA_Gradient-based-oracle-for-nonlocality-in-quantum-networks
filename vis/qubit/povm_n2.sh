#!/bin/bash
#SBATCH --job-name=v0.15_n2
#SBATCH --output=v0.15_n2.txt
#SBATCH --error=error_n2.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuze.zhu@inria.fr
#SBATCH --partition=gpu
#SBATCH --nodelist=margpu010
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=50
##SBATCH --mem=40G
#SBATCH --exclusive

module load conda
source activate new
srun python ~/vis/qubit/vis_n2.py
