#!/bin/bash
#SBATCH --job-name=0.37_20k_0.01
#SBATCH --output=re_cpu_0.37_20k_0.01.txt
#SBATCH --error=re_cpu_0.37.err
#SBATCH --exclusive
#SBATCH --cpus-per-task=40


module load conda
source activate env_pytorch
srun python ~/cuda/qu4_cpu.py

