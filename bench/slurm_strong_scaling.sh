#!/bin/bash
#
# Strong-scaling submitter for the JaxPM MG-vs-FFT force benchmark (H100, 4 GPUs/node).
#
# Strong scaling: the GLOBAL mesh size (--global_shape) is held FIXED while the GPU count
# grows from 4 to 512. Ideal strong scaling => wall-clock halves each time the GPU count
# doubles. The same global problem runs on every GPU count.
#
# Mirrors jaxDecomp/benchmarks/slurm_strong_scaling.sh: loops (config x shape x solver)
# and submits one sbatch job each via the external $SLURM_SCRIPT launcher:
#
#     export SLURM_SCRIPT=/path/to/your/h100_launcher.slurm
#     ./bench/slurm_strong_scaling.sh
#
# $SLURM_SCRIPT is invoked as:  sbatch <args> $SLURM_SCRIPT FORCES_STRONG python bench_forces.py ...

set -euo pipefail

# Configuration
ACCOUNT="${ACCOUNT:-XXX@h100}"
CONSTRAINT="${CONSTRAINT:-h100}"
OUTPUT_DIR="${OUTPUT_DIR:-results/strong_scaling}"
PRECISION="${PRECISION:-float32}"
ITERATIONS="${ITERATIONS:-5}"
GPUS_PER_NODE=4
# Halo is auto-sized per run: the smallest fraction of the local slab (HALO_FRACTIONS in the
# bench) that covers the rms displacement sigma for this box + mesh. Its share of the shrinking
# local slab grows with the GPU count. Only the box enters here.
BOX_SIZE="${BOX_SIZE:-2000.0}"     # Mpc/h

# Force solvers to benchmark, one sbatch job each.
SOLVERS=(fft mg mgwarm)

# Fixed global meshes (strong scaling). Powers of two for MG coarsening. NOTE: the force
# solver's footprint is several times a bare FFT's, so the larger shapes only fit at higher
# GPU counts -- big-shape / low-GPU-count jobs will OOM; prune them or start the sweep at a
# config large enough to hold the shape. (jaxDecomp's 4096^3 is intentionally dropped.)
SHAPES=(
    "512 512 512"
    "1024 1024 1024"
    "2048 2048 2048"
)

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

    for SHAPE in "${SHAPES[@]}"; do
        for SOLVER in "${SOLVERS[@]}"; do
            SHAPE_NAME=${SHAPE// /x}
            JOB_NAME="strong_${SOLVER}_N${NODES}_${TOTAL_GPUS}x1_${SHAPE_NAME}"
            echo "Submitting $JOB_NAME (Nodes:$NODES slab:${TOTAL_GPUS}x1 solver:$SOLVER global:$SHAPE)"
            sbatch $BASE_SBATCH_ARGS \
                --nodes="$NODES" \
                --gres=gpu:"$GPN" \
                --tasks-per-node="$GPN" \
                --gpu-bind=none \
                --job-name="$JOB_NAME" \
                "$SLURM_SCRIPT" FORCES_STRONG python bench_forces.py \
                --pdims "$TOTAL_GPUS" \
                --global_shape $SHAPE \
                -b "$SOLVER" \
                -n "$NODES" \
                --gpus-per-node "$GPN" \
                --box-size "$BOX_SIZE" \
                -o "$OUTPUT_DIR" \
                -pr "$PRECISION" \
                -i "$ITERATIONS"
        done
    done
done
