# Reference Code Audit

This document records historical bugs, discrepancies, and architecture details found in the legacy reference code included in this repository (`Common/`, `Single Hormone Population/`, `Dual Hormone Population/`).

These directories contain the foundational models originally used by the authors (e.g., Resalat et al.) and are preserved as **read-only reference evidence**. Do not alter these files as part of the new dataset generation effort, as they serve as the "ground truth" for deciphering undocumented paper behaviors.

## Known Discrepancies and Findings

### 1. Population Weight
*   **Current Generator:** `Weight = 70.0 kg` (hardcoded).
*   **Reference Code:** Uses an average weight of `76.3 kg`.
*   **Status:** [UNRESOLVED] The exact weight for the paper dataset is unknown. Left as 70.0 for now, but reference code suggests 76.3.

### 2. Basal-to-TDD Conversion
*   **Current Generator:** `tdd_est = basal_Uhr * 24.0 * 2.0`
*   **Reference Code:** `TDIR_Basal_Rate = 1.78`
*   **Status:** [UNRESOLVED] The `1.78` multiplier is explicitly stated in the reference codebase. Recommending switching to `1.78`, but leaving as `2.0` pending scientific review.

### 3. ICR Calculation
*   **Current Generator:** `icr_base = 1700.0 / (tdd_est * 3.0)`
*   **Reference Code:** `Test_Population.m` also uses `1700 / TDIR / 3`.
*   **Status:** [REFERENCE-CODE-DERIVED] The `/3` divisor is confirmed to exist in the original codebase.

### 4. Steady-State Solvers
*   **Current Generator:** Analytical solver (`solve_steady_state`).
*   **Reference Code:** Employs an `fsolve`-based numerical steady-state solver.
*   **Status:** [UNRESOLVED] Residuals should be tested, but analytical is mathematically sound if structurally equivalent.

### 5. Integration Methods
*   **Current Generator:** 1-minute substeps inside a 5-minute frame (`simulate_7day_ode_fast`).
*   **Reference Code:** Uses various forms of 5-minute forward-Euler and other discretizations.
*   **Status:** [UNRESOLVED] Must verify what the new paper relies on.

### 6. Legacy Bugs in Reference Code
*   **`CreateVirtualPatientPopulation.m`:** References variables like `Ins_red_fac`, `Population_No`, and `Avg_Wgt` which are not passed into its function signature (likely relying on base workspace variables).
*   **`PrunePopulationDH.m`:** Defines its internal function name as `PrunePopulation` rather than `PrunePopulationDH`, which causes a MATLAB filename/function-name warning or execution inconsistency.
*   **Obsolete Functions:** Reliance on legacy control functions like `c2dm`.
*   **Rescue Carbohydrate Logic:** Uses poorly bounded workspace propagation that can cause silent state errors if not handled correctly.
