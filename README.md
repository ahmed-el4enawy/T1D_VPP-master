# T1D Virtual Patient Population Dataset Reconstruction

## Overview

This repository contains the reconstructed dataset generator and validation suite for the paper:
*A Physiologically-Constrained Neural Network Digital Twin Framework for Replicating Glucose Dynamics in Type 1 Diabetes* (arXiv:2508.05705v1).

The core of this repository reconstructs the 10-state Hovorka-based ODE simulator (`T1DSim_ODE`) used by the authors to generate their population dataset (`D^P`).

## Repository Status

- **Dataset reconstruction is still under validation.** This is an independent reproduction effort.
- **This is not claimed to be an exact author dataset.** We have rigorously reverse-engineered the reported parameters and assumptions, but some exact seed states and bounds remain unpublished.
- **Unresolved assumptions are documented** in `docs/REPRODUCTION_ASSUMPTIONS.md`.

## Repository Layout

- `matlab/` - The active dataset generator MATLAB scripts.
- `scripts/` - Python data-building and dataset evaluation scripts.
- `data/inputs/` - The preprocessed T1DEXI day-scenario and midnight-CGM source data inputs.
- `tests/` - Python unit tests and the MATLAB 50-scenario smoke test.
- `reference/` - Unmodified legacy reference code (Resalat et al. / Hovorka) provided as scientific historical evidence.
- `docs/` - Assumptions and reference-code audit history.

## Reproduction Workflow

1. **Build source CSVs:** Parse the original clinical data (already provided in `data/inputs/`).
2. **Run small MATLAB smoke test:** Test the ODE simulator generator on 50 scenarios before deploying.
3. **Scientific pilot:** Generate a 500-1000 scenario dataset to verify output distributions.
4. **Evaluate ODE baseline:** Run `dataset_baseline.py` to compare standard glycemic risk metrics against the paper's reported values.
5. **Inspect fingerprint:** Compare structural IQR metrics using `dataset_fingerprint.py`.
6. **Full dataset generation:** Only execute the full 46,200 trace generation across HPC nodes once the pilot validates properly.
7. **Model Training:** Occurs in a separate repository containing the Neural Network implementation.

## Requirements

- Python 3.9+ (numpy, scipy, pandas, h5py)
- MATLAB (Required for dataset generation, but *not* strictly required for the Python evaluator if given a generated `.mat` file).

*Note: MATLAB is not required to run the Python validation suite.*

## Quick Start

### 1. Run the MATLAB Smoke Test
```bash
matlab -batch "run('tests/matlab/test_sample.m')"
```

### 2. Evaluate Glycemic Baselines
```bash
python scripts/dataset_baseline.py --dataset test_sample_dataset.mat
```

## Scientific Assumptions
For a full list of confirmed vs. unresolved scientific parameters, please see:
[REPRODUCTION_ASSUMPTIONS.md](docs/REPRODUCTION_ASSUMPTIONS.md)

## Reference Code
The original simulator implementations from Resalat et al. and the dual/single hormone VPPs are preserved without modification in the `reference/` directory for provenance and code-auditing purposes. See [REFERENCE_CODE_AUDIT.md](docs/REFERENCE_CODE_AUDIT.md) for known legacy bugs and notes.

## Validation
To validate the evaluation suite itself:
```bash
python -m unittest discover tests
```

## Known Limitations
- The exact authors' base-scenario count pre-augmentation was not published.
- The exact deterministic thresholds for bolus perturbations are undocumented.
- Full production out-of-core generation via HDF5 block-streaming is not yet fully integrated; the current implementation generates temporary MAT chunks which are then stitched in memory (requiring >16GB RAM for the full 46,200 cohort).
