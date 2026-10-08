#!/bin/bash
# hpc_dataset_baseline.sh
# Slurm script for running dataset baseline evaluator on HPC

#SBATCH --job-name=eval_dataset
#SBATCH --output=baseline_eval_%j.out
#SBATCH --error=baseline_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -e

# Preflight check
if [ ! -f "requirements.txt" ]; then
    echo "requirements.txt not found!"
    exit 1
fi

DATASET_PATH=${1:-"T1DSim_population_dataset.mat"}

if [ ! -f "$DATASET_PATH" ]; then
    echo "Dataset file not found: $DATASET_PATH"
    exit 1
fi

echo "Evaluating dataset: $DATASET_PATH"
python scripts/dataset_baseline.py --dataset "$DATASET_PATH" --cohort all

echo "Statistical fingerprinting dataset: $DATASET_PATH"
python dataset_fingerprint.py --dataset_path "$DATASET_PATH"

echo "Done."
