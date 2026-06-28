#!/bin/bash
#
# Weak-scaling submitter for the JaxPM MG-vs-FFT force benchmark (H100, 4 GPUs/node).
#
# Weak scaling: the per-GPU problem size (--local_shape) is held FIXED while the GPU
# count grows from 4 to 512, so the global mesh = local_shape * pdims on the x/y axes.
# Ideal weak scaling => flat wall-clock as GPUs increase.
#
# Mirrors jaxDecomp/benchmarks/slurm.sh: it loops a config grid and submits one sbatch
# job per (config, solver) via the external $SLURM_SCRIPT launcher (the sbatch header +
# srun wrapper for your cluster), exactly as jaxDecomp does:
#
#     export SLURM_SCRIPT=/path/to/your/h100_launcher.slurm
#     ./bench/slurm_weak_scaling.sh
#
# $SLURM_SCRIPT is invoked as:  sbatch <args> $SLURM_SCRIPT FORCES_WEAK python bench_forces.py ...

set -euo pipefail

# Configuration
ACCOUNT="${ACCOUNT:-XXX@h100}"
CONSTRAINT="${CONSTRAINT:-h100}"
OUTPUT_DIR="${OUTPUT_DIR:-results/weak_scaling}"
PRECISION="${PRECISION:-float32}"
ITERATIONS="${ITERATIONS:-5}"
GPUS_PER_NODE=4
# Halo is auto-sized per run: the smallest fraction of the local slab (HALO_FRACTIONS in the
# bench) that covers the rms displacement sigma for this box + mesh. Its share of the shrinking
# local slab grows with the GPU count. Only the box enters here.
BOX_SIZE="${BOX_SIZE:-2000.0}"     # Mpc/h

# Fixed per-GPU mesh (weak scaling). Conservative power-of-two size; the force solver
# (particles + density + 3 force meshes + MG hierarchy + potential) uses several times
# the memory of a bare FFT, so reduce this if you hit OOM (e.g. "128 128 128").
LOCAL_SHAPE="${LOCAL_SHAPE:-256 256 256}"

# Force solvers to benchmark, one sbatch job each.
SOLVERS=(fft mg mgwarm)

if [ -z "${SLURM_SCRIPT:-}" ]; then
    echo "Error: SLURM_SCRIPT is not set. Point it at your H100 sbatch launcher." >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

# Common SBATCH arguments base
BASE_SBATCH_ARGS="--account=$ACCOUNT -C $CONSTRAINT --time=01:00:00 --exclusive"

# Node counts (4 GPUs/node), 4 -> 512 GPUs. The run is always a 1D slab: pdims = n x 1
# where n = total GPUs.
NODE_COUNTS=(1 2 4 8 16 32 64 128)   # -> 4 8 16 32 64 128 256 512 GPUs

for NODES in "${NODE_COUNTS[@]}"; do
    GPN=$GPUS_PER_NODE
    TOTAL_GPUS=$((NODES * GPN))       # slab width n

    for SOLVER in "${SOLVERS[@]}"; do
        JOB_NAME="weak_${SOLVER}_N${NODES}_${TOTAL_GPUS}x1"
        echo "Submitting $JOB_NAME (Nodes:$NODES slab:${TOTAL_GPUS}x1 solver:$SOLVER local:$LOCAL_SHAPE)"
        sbatch $BASE_SBATCH_ARGS \
            --nodes="$NODES" \
            --gres=gpu:"$GPN" \
            --tasks-per-node="$GPN" \
            --gpu-bind=none \
            --job-name="$JOB_NAME" \
            "$SLURM_SCRIPT" FORCES_WEAK python bench_forces.py \
            --pdims "$TOTAL_GPUS" \
            --local_shape $LOCAL_SHAPE \
            -b "$SOLVER" \
            -n "$NODES" \
            --gpus-per-node "$GPN" \
            --box-size "$BOX_SIZE" \
            -o "$OUTPUT_DIR" \
            -pr "$PRECISION" \
            -i "$ITERATIONS"
    done
done
