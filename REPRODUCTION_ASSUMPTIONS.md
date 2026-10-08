# Scientific Assumptions Registry

This document records the exact status of decisions, parameters, and assumptions used in reconstructing the `T1D_VPP-master` population generator based on the paper *A Physiologically-Constrained Neural Network Digital Twin Framework for Replicating Glucose Dynamics in Type 1 Diabetes* (arXiv:2508.05705v1) and related reference code.

## Classification System

- **[PAPER-EXPLICIT]**: Explicitly stated in the primary manuscript or supplementary materials.
- **[REFERENCE-CODE-DERIVED]**: Not fully explained in the paper, but discovered in the authors' legacy/reference implementation in this repository.
- **[CONFIRMED-ENGINEERING-FIX]**: A deviation or repair made strictly to ensure reproducible behavior, prevent leakage, or fix software faults.
- **[RECONSTRUCTION-ASSUMPTION]**: An interpretation or reasonable guess used to bridge a gap between the paper and a working simulation.
- **[UNRESOLVED]**: A detail that cannot currently be confirmed and requires further investigation or author clarification.

## Documented Assumptions

### P. Paper-Explicit
* **P1.** Seven-day simulations for each state trajectory.
* **P2.** Five initial glucose variants drawn per 7-day meal scenario.
* **P3.** The initial-glucose distribution is based on the midnight CGM distribution from the specific participant corresponding to the meal scenario.
* **P4.** The pooled midnight CGM across the dataset is reported as approximately 156 ± 45 mg/dL.
* **P5.** Bolus delays range from 5 through 45 minutes as described in the manuscript.
* **P6.** Rare-event supplementation thresholds: TBR > 20% OR percentage time >250 mg/dL > 40%.
* **P7.** Final development dataset size: 323,400 simulated days, which equates to exactly 46,200 final seven-day traces.
* **P8.** The final evaluation splits are approximately 60% Train, 20% Validation, and 20% Test.
* **P9.** No daily-scenario leakage: The same daily meal scenario must occur in no more than one subset.

### R. Reference-Code-Derived
* **R1.** `ICR = 1700 / TDIR / 3`: The `/3` divisor is not described in the new paper, but appears explicitly in the Resalat reference code (`Test_Population.m`).
* **R2.** `TDIR_Basal_Rate = 1.78`: This factor is used in the reference code for converting Basal to Total Daily Insulin Requirement (TDIR), whereas our generator currently uses `2.0`.
* **R3.** `Avg_Wgt = 76.3 kg`: The reference population code uses an average weight of 76.3 kg, whereas the current generator hardcodes 70 kg.

### U. Unresolved
* **U1.** The exact number of pre-augmentation base scenarios generated before the 5 variants and rare-event supplementation.
* **U2.** Whether the rare supplementation traces are included *inside* the final reported 46,200 count, or added to it, and exactly how the final traces were pruned if necessary.
* **U3.** The exact magnitude and distribution of under-/over-dosing applied to the boluses.
* **U4.** Initial glucose clipping bounds (currently clamped to [70, 260] mg/dL in our implementation).
* **U5.** The exact numerical and deterministic recipe for rare-event perturbation (how much to scale insulin, carbs, basal, and delays).
* **U6.** Whether the 7 sampled daily scenarios making up a 7-day trace must belong strictly to the *same* single participant.
* **U7.** Whether sampling daily scenarios is done with or without replacement.
* **U8.** The exact numerical integration method (e.g., explicit Euler, sub-stepping) used for the final dataset generation.
* **U9.** The exact midnight-CGM extraction window (e.g., what hours constitute "midnight").
* **U10.** The minimum number of midnight CGM readings required per participant to be included in the stats.
* **U11.** The exact final split indices and the random seed used to generate the published T1DSim_AI model.
* **U12.** The exact construction mechanism for the approximately 1M training sequences from the traces.
* **U13.** The precise population weight distribution used by the paper's ODE dataset generation.
* **U14.** The exact basal-to-TDD conversion equation used by the paper.
* **U15.** Whether the first stored sample in the generated dataset represents `t=0` (the initial condition) or the state *after* the first integration interval.
