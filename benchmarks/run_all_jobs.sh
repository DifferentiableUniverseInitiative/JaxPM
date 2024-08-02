#!/bin/bash
# Run all slurms jobs
nodes_v100=(1 2 4 8 16)
nodes_a100=(1 2 4 8 16)


for n in ${nodes_v100[@]}; do
    sbatch --nodes=$n  --job-name=v100_$n-JAXPM particle_mesh_v100.slurm
done

for n in ${nodes_a100[@]}; do
    sbatch --nodes=$n --job-name=a100_$n-JAXPM particle_mesh_a100.slurm
done

# single GPUs
sbatch --job-name=JAXPM-1GPU-V100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 particle_mesh_v100.slurm
sbatch --job-name=JAXPM-1GPU-A100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 particle_mesh_a100.slurm
sbatch --job-name=PMWD-v100 pmwd_v100.slurm
sbatch --job-name=PMWD-a100 pmwd_a100.slurm
