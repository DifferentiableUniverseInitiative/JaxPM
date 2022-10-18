#!/bin/bash
#SBATCH -A m1727
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:05:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

module load python cudnn/8.2.0 nccl/2.11.4 cudatoolkit
export SLURM_CPU_BIND="cores"
srun python test_pfft.py
