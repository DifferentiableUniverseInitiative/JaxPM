#!/bin/bash
##############################################################################################################################
# USAGE:sbatch --account=tkc@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100  benchmarks/particle_mesh_a100.slurm 
##############################################################################################################################
#SBATCH --job-name=Particle-Mesh   # nom du job
#SBATCH --cpus-per-task=8           # nombre de CPU par tache pour gpu_p5 (1/8 du noeud 8-GPU)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=04:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%N_a100.out # nom du fichier de sortie
#SBATCH --error=%x_%N_a100.err  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --exclusive                  # ressources dediees
##SBATCH --qos=qos_gpu-dev
# Nettoyage des modules charges en interactif et herites par defaut
num_nodes=$SLURM_JOB_NUM_NODES
num_gpu_per_node=$SLURM_NTASKS_PER_NODE
OUTPUT_FOLDER_ARGS=1
# Calculate the number of GPUs
nb_gpus=$(( num_nodes * num_gpu_per_node))

module purge

echo "Job partition: $SLURM_JOB_PARTITION"
# Decommenter la commande module suivante si vous utilisez la partition "gpu_p5"
# pour avoir acces aux modules compatibles avec cette partition

if [[ "$SLURM_JOB_PARTITION" == "gpu_p5" ]]; then
    module load cpuarch/amd
    source /gpfsdswork/projects/rech/tkc/commun/venv/a100/bin/activate
    gpu_name=a100
else
    source /gpfsdswork/projects/rech/tkc/commun/venv/v100/bin/activate
    gpu_name=v100
fi

# Chargement des modules
module load nvidia-compilers/23.9 cuda/12.2.0 cudnn/8.9.7.29-cuda  openmpi/4.1.5-cuda nccl/2.18.5-1-cuda cmake
module load nvidia-nsight-systems/2024.1.1.59


echo "The number of nodes allocated for this job is: $num_nodes"
echo "The number of GPUs allocated for this job is: $nb_gpus"

export EQX_ON_ERROR=nan
export ENABLE_PERFO_STEP=NVTX
export MPI4JAX_USE_CUDA_MPI=1

function profile_python() {
    if [ $# -lt 1 ]; then
        echo "Usage: profile_python <python_script> [arguments for the script]"
        return 1
    fi

    local script_name=$(basename "$1" .py)
    local output_dir="prof_traces/$gpu_name/$nb_gpus/$script_name"
    local report_dir="out_prof/$gpu_name/$nb_gpus/$script_name"

    if [ $OUTPUT_FOLDER_ARGS -eq 1 ]; then
        local args=$(echo "${@:2}" | tr ' ' '_')
        # Remove characters '/' and '-' from folder name
        args=$(echo "$args" | tr -d '/-')
        output_dir="prof_traces/$gpu_name/$nb_gpus/$script_name/$args"
        report_dir="out_prof/$gpu_name/$nb_gpus/$script_name/$args"
    fi

    mkdir -p "$output_dir"
    mkdir -p "$report_dir"

    srun timeout 10m nsys profile -t cuda,nvtx,osrt,mpi -o "$report_dir/report_rank%q{SLURM_PROCID}" python "$@" > "$output_dir/$script_name.out" 2> "$output_dir/$script_name.err" || true
}

function run_python() {
    if [ $# -lt 1 ]; then
        echo "Usage: run_python <python_script> [arguments for the script]"
        return 1
    fi

    local script_name=$(basename "$1" .py)
    local output_dir="traces/$gpu_name/$nb_gpus/$script_name"

    if [ $OUTPUT_FOLDER_ARGS -eq 1 ]; then
        local args=$(echo "${@:2}" | tr ' ' '_')
        # Remove characters '/' and '-' from folder name
        args=$(echo "$args" | tr -d '/-')
        output_dir="traces/$gpu_name/$nb_gpus/$script_name/$args"
    fi

    mkdir -p "$output_dir"

    srun timeout 10m python "$@" > "$output_dir/$script_name.out" 2> "$output_dir/$script_name.err" || true
}


# run or profile

function slaunch() {
    run_python "$@"
}

function plaunch() {
    profile_python "$@"
}

# Echo des commandes lancees
set -x

# Pour ne pas utiliser le /tmp
export TMPDIR=$JOBSCRATCH
# Pour contourner un bogue dans les versions actuelles de Nsight Systems
# il est également nécessaire de créer un lien symbolique permettant de
# faire pointer le répertoire /tmp/nvidia vers TMPDIR
ln -s $JOBSCRATCH /tmp/nvidia

declare -A pdims_table
# Define the table
pdims_table[1]="1x1"
pdims_table[4]="2x2 1x4 4x1"
pdims_table[8]="2x4 1x8 8x1 4x2"
pdims_table[16]="4x4 1x16 16x1"
pdims_table[32]="4x8 8x4 1x32 32x1"
pdims_table[64]="8x8 16x4 1x64 64x1"
pdims_table[128]="8x16 16x8 1x128 128x1"
pdims_table[256]="16x16 1x256 256x1"

# mpch=(128 256 512 1024 2048 4096)
grid=(256 512 1024 2048 4096 8192)
precisions=(float32 float64)
pdim="${pdims_table[$nb_gpus]}"
solvers=(lpt lfm)
echo "pdims: $pdim"

# Check if pdims is not empty
if [ -z "$pdim" ]; then
    echo "pdims is empty"
    echo "Number of gpus has to be 8, 16, 32, 64, 128 or 160"
    echo "Number of nodes selected: $num_nodes"
    echo "Number of gpus per node: $num_gpu_per_node"
    exit 1
fi

# GPU name is a100 if num_gpu_per_node is 8, otherwise it is v100
out_dir="pm_prof/$gpu_name/$nb_gpus"
trace_dir="traces/$gpu_name/$nb_gpus/bench_pm"
echo "Output dir is  : $out_dir" 
echo "Trace dir is  : $trace_dir"

for pr in "${precisions[@]}"; do
    for g in "${grid[@]}"; do
        for solver in "${solvers[@]}"; do
            for p in $pdim; do
                halo_size=$((g / 4))
                slaunch bench_pm.py -m $g -b $g -p $p -hs $halo_size -pr $pr -s $solver -i 4 -o $out_dir -f -n $num_nodes
            done
        done
        # delete crash core dump files
        rm -f core.python.*
    done
done

# # zip the output files and traces
# tar -czvf $out_dir.tar.gz $out_dir
# tar -czvf $trace_dir.tar.gz $trace_dir
# # remove the output files and traces
# rm -rf $out_dir $trace_dir
