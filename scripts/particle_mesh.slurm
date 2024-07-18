#!/bin/bash
########################################## 
## SELECT EITHER tkc@a100 OR tkc@v100 ## 
########################################## 
#SBATCH --account tkc@a100
########################################## 
#SBATCH --job-name=Particle-Mesh   # nom du job
# Il est possible d'utiliser une autre partition que celle par default
# en activant l'une des 5 directives suivantes :
########################################## 
## SELECT EITHER a100  or v100-32g   ## 
########################################## 
#SBATCH -C a100
########################################## 
#******************************************
########################################## 
## SELECT Number of nodes and GPUs per node
## For A100 ntasks-per-node and gres=gpu should be 8
## For V100 ntasks-per-node and gres=gpu should be 4
########################################## 
#SBATCH --nodes=1                    # nombre de noeud
#SBATCH --ntasks-per-node=8          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:8                # nombre de GPU par nÅ“ud (max 8 avec gpu_p2, gpu_p5)
##########################################
## Le nombre de CPU par tache doit etre adapte en fonction de la partition utilisee. Sachant
## qu'ici on ne reserve qu'un seul GPU par tache (soit 1/4 ou 1/8 des GPU du noeud suivant
## la partition), l'ideal est de reserver 1/4 ou 1/8 des CPU du noeud pour chaque tache:
##########################################
#SBATCH --cpus-per-task=8           # nombre de CPU par tache pour gpu_p5 (1/8 du noeud 8-GPU)
##########################################
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=04:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%N_a100.out # nom du fichier de sortie
#SBATCH --error=%x_%N_a100.out  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --qos=qos_gpu-dev
#SBATCH --exclusive                  # ressources dediees

# Nettoyage des modules charges en interactif et herites par defaut
num_nodes=$SLURM_JOB_NUM_NODES
num_gpu_per_node=$SLURM_NTASKS_PER_NODE
OUTPUT_FOLDER_ARGS=1
# Calculate the number of GPUs
nb_gpus=$(( num_nodes * num_gpu_per_node))

module purge

# Decommenter la commande module suivante si vous utilisez la partition "gpu_p5"
# pour avoir acces aux modules compatibles avec cette partition

if [ $num_gpu_per_node -eq 8 ]; then
    module load cpuarch/amd
    source /gpfsdswork/projects/rech/tkc/commun/venv/a100/bin/activate
else
    source /gpfsdswork/projects/rech/tkc/commun/venv/v100/bin/activate
fi

# Chargement des modules
module load nvidia-compilers/23.9 cuda/12.2.0 cudnn/8.9.7.29-cuda  openmpi/4.1.5-cuda nccl/2.18.5-1-cuda cmake
module load nvidia-nsight-systems/2024.1.1.59 

echo "The number of nodes allocated for this job is: $num_nodes"
echo "The number of GPUs allocated for this job is: $nb_gpus"

export EQX_ON_ERROR=nan
export CUDA_ALLOC=1

function profile_python() {
    if [ $# -lt 1 ]; then
        echo "Usage: profile_python <python_script> [arguments for the script]"
        return 1
    fi

    local script_name=$(basename "$1" .py)
    local output_dir="prof_traces/$script_name"
    local report_dir="out_prof/$gpu_name/$nb_gpus/$script_name"

    if [ $OUTPUT_FOLDER_ARGS -eq 1 ]; then
        local args=$(echo "${@:2}" | tr ' ' '_')
        # Remove characters '/' and '-' from folder name
        args=$(echo "$args" | tr -d '/-')
        output_dir="prof_traces/$script_name/$args"
        report_dir="out_prof/$gpu_name/$nb_gpus/$script_name/$args"
    fi

    mkdir -p "$output_dir"
    mkdir -p "$report_dir"

    srun nsys profile -t cuda,nvtx,osrt,mpi -o "$report_dir/report_rank%q{SLURM_PROCID}" python "$@" > "$output_dir/$script_name.out" 2> "$output_dir/$script_name.err" || true
}

function run_python() {
    if [ $# -lt 1 ]; then
        echo "Usage: run_python <python_script> [arguments for the script]"
        return 1
    fi

    local script_name=$(basename "$1" .py)
    local output_dir="traces/$script_name"

    if [ $OUTPUT_FOLDER_ARGS -eq 1 ]; then
        local args=$(echo "${@:2}" | tr ' ' '_')
        # Remove characters '/' and '-' from folder name
        args=$(echo "$args" | tr -d '/-')
        output_dir="traces/$script_name/$args"
    fi

    mkdir -p "$output_dir"

    srun python "$@" > "$output_dir/$script_name.out" 2> "$output_dir/$script_name.err" || true
}


# Echo des commandes lancees
set -x

# Pour la partition "gpu_p5", le code doit etre compile avec les modules compatibles
# Execution du code avec binding via bind_gpu.sh : 1 GPU par tache
 



declare -A pdims_table
# Define the table
pdims_table[4]="2x2 1x4"
pdims_table[8]="2x4 1x8"
pdims_table[16]="2x8 1x16"
pdims_table[32]="4x8 1x32"
pdims_table[64]="4x16 1x64"
pdims_table[128]="8x16 16x8 4x32 32x4 1x128 128x1 2x64 64x2"
pdims_table[160]="8x20 20x8 16x10 10x16 5x32 32x5 1x160 160x1 2x80 80x2 4x40 40x4"


#mpch=(128 256 512 1024 2048 4096)
grid=(1024 2048 4096)

pdim="${pdims_table[$nb_gpus]}"
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

if [ $num_gpu_per_node -eq 8 ]; then
    gpu_name="a100"
else
    gpu_name="v100"
fi

out_dir="out/$gpu_name/$nb_gpus"

echo "Output dir is  : $out_dir"

for g in ${grid[@]}; do
    for p in ${pdim[@]}; do
        # halo is 1/4 of the grid size
        halo_size=$((g / 4))
        slaunch scripts/fastpm_jaxdecomp.py -m $g -b $g -p $p -hs $halo_size -ode diffrax -o $out_dir
    done
done
