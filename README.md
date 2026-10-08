# T1D Virtual Patient Population (T1D_VPP) Dataset Generator

This repository reconstructs the **Population Development Dataset** for the paper:
*"A Physiologically-Constrained Neural Network Digital Twin Framework for Replicating Glucose Dynamics in Type 1 Diabetes"* (arXiv:2508.05705v1).

## Objective
To create an auditable, deterministic, scientifically defensible dataset-generation pipeline consistent with:
1. The paper's methodology.
2. The released T1DSim_AI model artifacts.
3. The legacy Resalat/OHSU reference model code (read-only evidence).

## Workflow

1.  **Raw T1DEXI Extraction:** Extract raw data using `build_day_scenario_library_v4.py` and `build_midnight_cgm_stats_v4.py`.
2.  **Validation & Provenance:** Verify the `day_scenario_library.csv` and `midnight_cgm_stats.csv` against their JSON manifests.
3.  **Generator & Splits:** Run `generate_population_dataset.m` (MATLAB) to assemble the dataset. The generator guarantees deterministic, leakage-safe Train/Val/Test splits by construction.
4.  **Pilot Generation:** Run a small scale generation (e.g. `num_scenarios=50`) and confirm output dimensions using `test_sample.m`.
5.  **ODE Baseline Evaluation:** Analyze the generated test set with `dataset_baseline.py` using 5-hour Kovatchev windows.
6.  **Statistical Fingerprint:** Use `dataset_fingerprint.py` to compare dataset scale properties.
7.  **Candidate Full Generation:** Once approved, deploy full generation on HPC.
8.  **NN Retraining:** Proceed to train the neural network solely upon a successfully validated full dataset.

## Repository Structure

*   `generate_population_dataset.m`: The primary MATLAB generator script.
*   `load_day_scenario_library.m` & `load_midnight_cgm_stats.m`: Strongly typed CSV loaders.
*   `dataset_baseline.py`: Python evaluator for computing TIR, TAR, TBR, LBGI, HBGI, and MG on the final dataset traces.
*   `dataset_fingerprint.py`: Script to generate distribution statistics before NN training.
*   `REPRODUCTION_ASSUMPTIONS.md`: Explicit registry of scientifically unresolved choices and parameters.
*   `REFERENCE_CODE_AUDIT.md`: Historical bugs and anomalies in the author's legacy reference code.
*   `Common/`, `Single Hormone Population/`, `Dual Hormone Population/`: Legacy reference directories (Read-Only).

> **Note on Documentation:** Older documentation (such as `generate_population_dataset_explained.pdf`) may describe historical implementation details. Refer to the active scripts and Markdown files for the current true behavior.
