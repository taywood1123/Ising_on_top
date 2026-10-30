#!/bin/bash
# V3 γ sweep over the W bank (ising_core_V3.jl: |ΔE| < 1e-12 treated as a tie).
# Each array task handles W_PER_TASK consecutive Ws, multithreaded over all
# (W, sign, γ, GRN, sim) tasks. 100 tasks × 10 W = W1..W1000.
#
# Results: one file per W (and sign) in $ISING_OUT_DIR, all γ in the same file.
# Resubmitting (after a timeout, or with new γ values) only runs what is missing
# and merges it into the existing V3 files.
#
# Needs in $SCRIPT_DIR: run_ising_Wbank_V3.jl and ising_core_V3.jl
# Needs to exist before submitting: /home/twwoodw/ondemand/script/logs/

#SBATCH --job-name=ising_Wbank_V3
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=0-02:00:00
#SBATCH --array=1-100
#SBATCH --output=/home/twwoodw/ondemand/script/logs/%x_%A_%a.out
#SBATCH --error=/home/twwoodw/ondemand/script/logs/%x_%A_%a.err

module purge
module load Julia/1.12.6-linux-x86_64

export OPENBLAS_NUM_THREADS=1
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export ISING_OUT_DIR=/data/scratch/projects/punim0638/woodward/nonrcep_casestudy/gamma_data_V3


SCRIPT_DIR=/home/twwoodw/ondemand/script
W_BANK=/home/twwoodw/my_folder/compute/data/W_bank.jld2
W_PER_TASK=10

W_START=$(( (SLURM_ARRAY_TASK_ID - 1) * W_PER_TASK + 1 ))
W_END=$(( SLURM_ARRAY_TASK_ID * W_PER_TASK ))

N_CELLS=400
N_GENES=9
N_SIMS=10
MAX_SWEEPS=10000
BETA=Inf
BETA_NAME=betaInf
SIGMA_NAME=σrand
GAMMAS="0.25,0.22,0.20,0.18,0.16,0.14,0.12,0.125"
GAMMA_NAMES="g025,g022,g02,g018,g016,g014,g012,g0125"
W_SIGNS="1,-1"
SEED=1234

julia -t $SLURM_CPUS_PER_TASK $SCRIPT_DIR/run_ising_Wbank_V3.jl \
    $N_CELLS $N_GENES $N_SIMS $MAX_SWEEPS $BETA $BETA_NAME $SIGMA_NAME \
    "$GAMMAS" "$GAMMA_NAMES" \
    $W_BANK $W_START $W_END $W_SIGNS $SEED
