#!/bin/bash
#SBATCH --partition=gpu5
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=baseline_eval_%j.out
#SBATCH --error=baseline_eval_%j.err

set -euo pipefail

cd "/nfs/slurm/cugp012/T1D_VPP-master"

echo "Starting dataset evaluation baseline"

# Test the script first
/nfs/slurm/cugp012/envs/t1d/bin/python -u dataset_baseline.py --test

# Run the actual evaluation
/nfs/slurm/cugp012/envs/t1d/bin/python -u dataset_baseline.py \
  --dataset /tmp/cugp012/population_development_dataset_merged.mat \
  --cohort all \
  --output-json dataset_baseline_report.json

echo "Evaluation finished successfully."
