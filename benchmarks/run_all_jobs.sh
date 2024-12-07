#!/bin/bash
# Run all slurms jobs
nodes_v100=(1 2 4 8 16 32)
nodes_a100=(1 2 4 8 16 32)


for n in ${nodes_v100[@]}; do
    sbatch --account=tkc@v100 --nodes=$n --gres=gpu:4 --tasks-per-node=4 -C v100-32g --job-name=JAXPM-$n-N-v100 particle_mesh.slurm
done

for n in ${nodes_a100[@]}; do
    sbatch --account=tkc@a100 --nodes=$n --gres=gpu:4 --tasks-per-node=4 -C a100     --job-name=JAXPM-$n-N-a100 particle_mesh.slurm
done

# single GPUs
sbatch --account=tkc@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100     --job-name=JAXPM-1GPU-V100 particle_mesh.slurm
sbatch --account=tkc@v100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C v100-32g --job-name=JAXPM-1GPU-A100 particle_mesh.slurm
sbatch --account=tkc@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100     --job-name=PMWD-1GPU-v100 pmwd_pm.slurm
sbatch --account=tkc@v100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C v100-32g --job-name=PMWD-1GPU-a100 pmwd_pm.slurm
