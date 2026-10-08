import os
import sys
import h5py
import numpy as np
import argparse
import json
import time

def window_outcomes_batch(bg):
    """
    Exact mathematical port of the clinical outcomes from evaluate_population_model.py.
    """
    TIR_LOW, TIR_HIGH = 70.0, 180.0
    tir = 100.0 * np.mean((bg >= TIR_LOW) & (bg <= TIR_HIGH), axis=1)
    tar = 100.0 * np.mean(bg > TIR_HIGH, axis=1)
    tbr = 100.0 * np.mean(bg < TIR_LOW, axis=1)

    bgc = np.clip(bg, 1.0, None)
    f_bg = 1.509 * (np.log(bgc) ** 1.084 - 5.381)
    r_bg = 10.0 * f_bg ** 2
    neg = f_bg < 0
    pos = ~neg
    n = bg.shape[1]
    lbgi = (r_bg * neg).sum(axis=1) / n
    hbgi = (r_bg * pos).sum(axis=1) / n

    return {"TIR": tir, "TAR": tar, "TBR": tbr, "LBGI": lbgi, "HBGI": hbgi, "MG": bg.mean(axis=1)}


class WelfordAccumulator:
    """ Numerically stable streaming mean and sample standard deviation. """
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values):
        for val in values:
            self.count += 1
            delta = val - self.mean
            self.mean += delta / self.count
            delta2 = val - self.mean
            self.m2 += delta * delta2
            
    def get_mean(self):
        return self.mean if self.count > 0 else float('nan')
        
    def get_std(self):
        if self.count < 2:
            return float('nan')
        return np.sqrt(self.m2 / (self.count - 1))  # ddof=1

def get_splits_logic(split_idx, day_ids_all, target_count, seed):
    """ Ported exactly from train_population_model.py (perf/paper-scale-72h) """
    split_idx = np.asarray(split_idx, dtype=np.int64)
    if len(split_idx) <= target_count:
        return np.sort(split_idx)

    keys = np.sort(day_ids_all[split_idx], axis=1)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)

    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse)
    starts = np.concatenate(([0], np.cumsum(counts)))

    rng = np.random.RandomState(seed)
    group_order = rng.permutation(len(counts))

    selected_groups = []
    selected_count = 0
    for g in group_order:
        c = int(counts[g])
        if selected_count + c <= target_count:
            selected_groups.append(int(g))
            selected_count += c
            if selected_count == target_count:
                break

    if selected_count == 0:
        raise RuntimeError("Paper-scale group selection produced no traces.")

    selected_parts = []
    for g in selected_groups:
        lo, hi = int(starts[g]), int(starts[g + 1])
        selected_parts.append(split_idx[order[lo:hi]])

    chosen = np.sort(np.concatenate(selected_parts))
    return chosen

def evaluate_cohort(f, trace_indices, orientation, name, is_rare_flags=None, chunk_size=1000):
    window_size = 61
    hop = 60
    
    tir_acc = WelfordAccumulator()
    tar_acc = WelfordAccumulator()
    tbr_acc = WelfordAccumulator()
    mg_acc = WelfordAccumulator()
    lbgi_acc = WelfordAccumulator()
    hbgi_acc = WelfordAccumulator()
    
    num_traces = len(trace_indices)
    total_windows = 0
    skipped_windows = 0
    
    rare_count = 0
    non_rare_count = 0
    rare_mg_acc = WelfordAccumulator()
    rare_tir_acc = WelfordAccumulator()
    rare_tar_acc = WelfordAccumulator()
    rare_tbr_acc = WelfordAccumulator()
    
    # Sort indices for contiguous HDF5 reading
    trace_indices = np.sort(trace_indices)
    
    for start_i in range(0, num_traces, chunk_size):
        end_i = min(start_i + chunk_size, num_traces)
        chunk_idx = trace_indices[start_i:end_i]
        
        # Extract chunk data efficiently based on detected orientation
        if orientation == 'time_first':
            cgm_chunk = f['dataset_glucose'][:, chunk_idx]
            cgm_chunk = cgm_chunk.T # Convert to (trace, time)
        else:
            cgm_chunk = f['dataset_glucose'][chunk_idx, :]
            
        rare_chunk = None
        if is_rare_flags is not None:
            rare_chunk = is_rare_flags[chunk_idx]
            
        N_steps = cgm_chunk.shape[1]
        
        # Verify number of generated windows mathematically
        # Example: for 2016 steps and hop=60 -> floor(2015 / 60) = 33
        num_windows_per_trace = (N_steps - 1) // hop
        
        for i, trace in enumerate(cgm_chunk):
            is_rare = False
            if rare_chunk is not None and rare_chunk[i]:
                is_rare = True
                rare_count += 1
            else:
                non_rare_count += 1
                
            for w in range(num_windows_per_trace):
                start_idx = w * hop
                end_idx = start_idx + window_size
                if end_idx > N_steps:
                    break
                win = trace[start_idx:end_idx]
                
                if np.isnan(win).any() or np.isinf(win).any():
                    skipped_windows += 1
                    continue
                
                total_windows += 1
                
                win_2d = win[None, :]
                out = window_outcomes_batch(win_2d)
                
                tir = out["TIR"][0]
                tar = out["TAR"][0]
                tbr = out["TBR"][0]
                lbgi = out["LBGI"][0]
                hbgi = out["HBGI"][0]
                mg = out["MG"][0]
                
                # Sanity Check 1
                assert np.isclose(tir + tar + tbr, 100.0), f"TIR+TAR+TBR={tir+tar+tbr}"
                
                tir_acc.update([tir])
                tar_acc.update([tar])
                tbr_acc.update([tbr])
                mg_acc.update([mg])
                lbgi_acc.update([lbgi])
                hbgi_acc.update([hbgi])
                
                if is_rare:
                    rare_tir_acc.update([tir])
                    rare_tar_acc.update([tar])
                    rare_tbr_acc.update([tbr])
                    rare_mg_acc.update([mg])
                
        pct = int(end_i / num_traces * 100)
        print(f"[{name}] Processed {end_i}/{num_traces} traces ({pct}%)...", flush=True)

    results = {
        'num_traces': num_traces,
        'windows_per_trace': (N_steps - 1) // hop,
        'total_windows_evaluated': total_windows,
        'windows_skipped': skipped_windows,
        'rare_count': rare_count,
        'non_rare_count': non_rare_count,
        'TIR': {'mean': tir_acc.get_mean(), 'std': tir_acc.get_std()},
        'TAR': {'mean': tar_acc.get_mean(), 'std': tar_acc.get_std()},
        'TBR': {'mean': tbr_acc.get_mean(), 'std': tbr_acc.get_std()},
        'MG': {'mean': mg_acc.get_mean(), 'std': mg_acc.get_std()},
        'LBGI': {'mean': lbgi_acc.get_mean(), 'std': lbgi_acc.get_std()},
        'HBGI': {'mean': hbgi_acc.get_mean(), 'std': hbgi_acc.get_std()},
    }
    
    if rare_count > 0:
        results['RARE_DIAGNOSTICS'] = {
            'TIR_mean': rare_tir_acc.get_mean(),
            'TAR_mean': rare_tar_acc.get_mean(),
            'TBR_mean': rare_tbr_acc.get_mean(),
            'MG_mean': rare_mg_acc.get_mean(),
        }
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/tmp/cugp012/population_development_dataset_merged.mat')
    parser.add_argument('--cohort', choices=['all', 'original', 'paper-scale'], default='all')
    parser.add_argument('--chunk-size', type=int, default=1000)
    parser.add_argument('--output-json', default='dataset_baseline_report.json')
    parser.add_argument('--test', action='store_true', help='Run unit tests and exit')
    args = parser.parse_args()
    
    if args.test:
        run_tests()
        return
        
    print(f"Opening dataset: {args.dataset}")
    f = h5py.File(args.dataset, 'r')
    
    # 1. Inspect Schema & Orientation safely
    cgm_ds = f['dataset_glucose']
    print(f"Detected dataset_glucose shape: {cgm_ds.shape}")
    
    if cgm_ds.shape[0] == 2016:
        orientation = 'time_first'
        num_traces = cgm_ds.shape[1]
    elif cgm_ds.shape[1] == 2016:
        orientation = 'trace_first'
        num_traces = cgm_ds.shape[0]
    else:
        raise ValueError(f"Unrecognized shape: {cgm_ds.shape}. Expected one dimension to be 2016.")
    
    print(f"Orientation: {orientation} with {num_traces} total traces.")
    
    # 2. Extract Splits
    split_ids = np.asarray(f['dataset_split_id']).ravel()
    test_all_idx = np.where(split_ids == 2)[0]
    print(f"Found {len(test_all_idx)} original TEST traces.")
    
    is_rare = None
    if 'dataset_is_rare' in f:
        is_rare = np.asarray(f['dataset_is_rare']).ravel()
        print(f"Found dataset_is_rare flag. Rare TEST traces: {np.sum(is_rare[test_all_idx])}")
        
    report = {
        'dataset_path': args.dataset,
        'timestamp': time.time(),
        'shape': cgm_ds.shape,
        'orientation': orientation,
        'cohorts': {}
    }
        
    if args.cohort in ['all', 'original']:
        print("\n=== EVALUATING COHORT A (ORIGINAL GENERATOR TEST SPLIT) ===")
        res_a = evaluate_cohort(f, test_all_idx, orientation, "Cohort A", is_rare, args.chunk_size)
        report['cohorts']['Cohort_A'] = res_a
        print_comparison(res_a, "Cohort A (Original TEST)")
        
    if args.cohort in ['all', 'paper-scale']:
        print("\n=== EVALUATING COHORT B (PAPER-SCALE RECONSTRUCTED TEST SUBSET) ===")
        day_ids = np.asarray(f['dataset_day_ids'])
        if day_ids.shape[0] == 7:
            day_ids = day_ids.T
            
        PAPER_TEST_TRACES = 9240
        SEED = 0 # From train_population_model.py
        test_subset_idx = get_splits_logic(test_all_idx, day_ids, PAPER_TEST_TRACES, SEED + 103)
        print(f"Reconstructed subset size: {len(test_subset_idx)}")
        
        # Check against out-of-bounds indices
        assert np.all(test_subset_idx >= 0) and np.all(test_subset_idx < num_traces)
        
        res_b = evaluate_cohort(f, test_subset_idx, orientation, "Cohort B", is_rare, args.chunk_size)
        report['cohorts']['Cohort_B'] = res_b
        print_comparison(res_b, "Cohort B (Paper-scale TEST)")
        
    with open(args.output_json, 'w') as outf:
        json.dump(report, outf, indent=2)
    print(f"\nReport saved to {args.output_json}")
    
    print("\n=== SCIENTIFIC INTERPRETATION ===")
    print("If both cohorts deviate heavily from the paper ODE DP_test results,")
    print(" -> the generator dataset distribution itself is the likely mismatch.")
    print("If Cohort A is close but Cohort B is far,")
    print(" -> the paper-scale downsampling logic is distorting it.")
    print("If both are reasonably close,")
    print(" -> dataset generation is adequate and downstream differences may come from NN training.")
    print("\nNOTE: Do not claim causality solely from these aggregate metrics.")
    print("=================================")

def print_comparison(res, name):
    print(f"\n--- {name} Results ---")
    print("Metric | Our mean ± SD         | Paper T1DSimP_ODE on simulated DP_test | Mean diff")
    print("-----------------------------------------------------------------------------------")
    
    paper_tgts = {
        'TIR': (72.9, 35.2),
        'TAR': (23.1, 36.3),
        'TBR': (4.0, 8.2),
        'LBGI': (1.5, 1.7),
        'HBGI': (4.2, 6.2),
        'MG': (132.6, 48.2)
    }
    
    for m in ['TIR', 'TAR', 'TBR', 'LBGI', 'HBGI', 'MG']:
        our_m = res[m]['mean']
        our_sd = res[m]['std']
        p_m = paper_tgts[m][0]
        p_sd = paper_tgts[m][1]
        diff = our_m - p_m
        unit = "mg/dL" if m == 'MG' else ("" if "BGI" in m else "%")
        
        print(f"{m:4s}   | {our_m:5.1f} ± {our_sd:5.1f} {unit:5s} | {p_m:5.1f} ± {p_sd:5.1f} {unit:21s} | {diff:+5.1f}")
        
    print(f"\nTotal traces evaluated: {res['num_traces']}")
    print(f"Rare traces: {res['rare_count']} ({(res['rare_count']/res['num_traces']*100):.1f}%)")
    print(f"Non-rare traces: {res['non_rare_count']}")
    print(f"Total 5-hour windows: {res['total_windows_evaluated']}")
    
    if 'RARE_DIAGNOSTICS' in res:
        print("\n[Diagnostic: Rare Traces Only]")
        print(f"TIR: {res['RARE_DIAGNOSTICS']['TIR_mean']:.1f}%")
        print(f"TAR: {res['RARE_DIAGNOSTICS']['TAR_mean']:.1f}%")
        print(f"TBR: {res['RARE_DIAGNOSTICS']['TBR_mean']:.1f}%")
        print(f"MG:  {res['RARE_DIAGNOSTICS']['MG_mean']:.1f} mg/dL")


def run_tests():
    print("Running Unit Tests...")
    
    # 1. Welford accumulator test
    acc = WelfordAccumulator()
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    acc.update(data)
    assert np.isclose(acc.get_mean(), 3.0)
    assert np.isclose(acc.get_std(), np.std(data, ddof=1))
    
    # 2. Risk metrics test (analytical)
    bg_test = np.array([[100.0]])
    out = window_outcomes_batch(bg_test)
    assert out["LBGI"][0] > 0
    assert out["HBGI"][0] == 0
    
    # 3. TIR/TAR/TBR sum test
    win = np.array([[65]*20 + [100]*21 + [200]*20])
    assert win.shape[1] == 61
    out = window_outcomes_batch(win)
    tir, tar, tbr = out["TIR"][0], out["TAR"][0], out["TBR"][0]
    assert np.isclose(tir + tar + tbr, 100.0)
    
    # 4. Regression test against training_population's exact function
    def original_window_outcomes_batch(bg):
        # Literal copy from evaluate_population_model.py
        tir = 100.0 * np.mean((bg >= 70.0) & (bg <= 180.0), axis=1)
        tar = 100.0 * np.mean(bg > 180.0, axis=1)
        tbr = 100.0 * np.mean(bg < 70.0, axis=1)

        bgc = np.clip(bg, 1.0, None)
        f_bg = 1.509 * (np.log(bgc) ** 1.084 - 5.381)
        r_bg = 10.0 * f_bg ** 2
        neg = f_bg < 0
        pos = ~neg
        n = bg.shape[1]
        lbgi = (r_bg * neg).sum(axis=1) / n
        hbgi = (r_bg * pos).sum(axis=1) / n

        return {"TIR": tir, "TAR": tar, "TBR": tbr, "LBGI": lbgi, "HBGI": hbgi, "MG": bg.mean(axis=1)}

    # Create a realistic shape batch (batch_size=2, window_len=61)
    win_synth = np.array([
        [60, 70, 80, 100, 150, 190, 250, 300]*7 + [100]*5, # 61 elements
        [40]*61                                            # 61 elements
    ], dtype=np.float64)
    
    orig_out = original_window_outcomes_batch(win_synth)
    new_out = window_outcomes_batch(win_synth)
    
    for k in orig_out:
        assert np.allclose(orig_out[k], new_out[k]), f"Mismatch in {k}: {orig_out[k]} vs {new_out[k]}"
        
    print("Regression test against training_population evaluate_population_model passed.")

    print("All unit tests passed.")

if __name__ == '__main__':
    main()
