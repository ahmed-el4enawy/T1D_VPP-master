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

import numpy as np


def evaluate_cohort_generic(cgm_ds, trace_indices, orientation, name, is_rare_flags=None, chunk_size=1000):
    window_size = 61
    hop = 60
    
    num_traces = len(trace_indices)
    
    stats = {
        'all': {'tir': WelfordAccumulator(), 'tar': WelfordAccumulator(), 'tbr': WelfordAccumulator(), 'mg': WelfordAccumulator(), 'lbgi': WelfordAccumulator(), 'hbgi': WelfordAccumulator()},
        'rare': {'tir': WelfordAccumulator(), 'tar': WelfordAccumulator(), 'tbr': WelfordAccumulator(), 'mg': WelfordAccumulator(), 'lbgi': WelfordAccumulator(), 'hbgi': WelfordAccumulator()},
        'non_rare': {'tir': WelfordAccumulator(), 'tar': WelfordAccumulator(), 'tbr': WelfordAccumulator(), 'mg': WelfordAccumulator(), 'lbgi': WelfordAccumulator(), 'hbgi': WelfordAccumulator()}
    }
    
    counts = {
        'candidate_windows': 0,
        'evaluated_windows': 0,
        'skipped_windows': 0,
        'nan_count': 0,
        'inf_count': 0,
        'rare_traces': 0,
        'non_rare_traces': 0
    }
    
    global_min = float('inf')
    global_max = float('-inf')
    
    trace_indices = np.sort(trace_indices)
    
    for start_i in range(0, num_traces, chunk_size):
        end_i = min(start_i + chunk_size, num_traces)
        chunk_idx = trace_indices[start_i:end_i]
        
        if orientation == 'time_first':
            cgm_chunk = cgm_ds[:, chunk_idx].T
        else:
            cgm_chunk = cgm_ds[chunk_idx, :]
            
        N_steps = cgm_chunk.shape[1]
        
        if N_steps >= window_size:
            num_windows_per_trace = 1 + (N_steps - window_size) // hop
        else:
            num_windows_per_trace = 0
            
        counts['candidate_windows'] += num_windows_per_trace * len(chunk_idx)
        
        rare_chunk = None
        if is_rare_flags is not None:
            rare_chunk = is_rare_flags[chunk_idx]
            counts['rare_traces'] += np.sum(rare_chunk)
            counts['non_rare_traces'] += len(chunk_idx) - np.sum(rare_chunk)
        else:
            counts['non_rare_traces'] += len(chunk_idx)
            
        # Collect valid windows for batch processing
        # Using strides or reshape is efficient but we can just slice
        # cgm_chunk is (B, T)
        
        # Precompute window starts
        starts = np.arange(num_windows_per_trace) * hop
        ends = starts + window_size
        
        for w_idx in range(num_windows_per_trace):
            win_batch = cgm_chunk[:, starts[w_idx]:ends[w_idx]]
            
            # Find valid windows
            nan_mask = np.isnan(win_batch)
            inf_mask = np.isinf(win_batch)
            invalid = nan_mask.any(axis=1) | inf_mask.any(axis=1)
            valid = ~invalid
            
            counts['nan_count'] += int(np.sum(nan_mask))
            counts['inf_count'] += int(np.sum(inf_mask))
            counts['skipped_windows'] += int(np.sum(invalid))
            
            if np.sum(valid) == 0:
                continue
                
            win_valid = win_batch[valid]
            counts['evaluated_windows'] += win_valid.shape[0]
            
            global_min = min(global_min, float(np.min(win_valid)))
            global_max = max(global_max, float(np.max(win_valid)))
            
            out = window_outcomes_batch(win_valid)
            
            stats['all']['tir'].update(out['TIR'])
            stats['all']['tar'].update(out['TAR'])
            stats['all']['tbr'].update(out['TBR'])
            stats['all']['mg'].update(out['MG'])
            stats['all']['lbgi'].update(out['LBGI'])
            stats['all']['hbgi'].update(out['HBGI'])
            
            if rare_chunk is not None:
                r_mask = rare_chunk[valid]
                nr_mask = ~r_mask
                
                if np.sum(r_mask) > 0:
                    stats['rare']['tir'].update(out['TIR'][r_mask])
                    stats['rare']['tar'].update(out['TAR'][r_mask])
                    stats['rare']['tbr'].update(out['TBR'][r_mask])
                    stats['rare']['mg'].update(out['MG'][r_mask])
                    stats['rare']['lbgi'].update(out['LBGI'][r_mask])
                    stats['rare']['hbgi'].update(out['HBGI'][r_mask])
                    
                if np.sum(nr_mask) > 0:
                    stats['non_rare']['tir'].update(out['TIR'][nr_mask])
                    stats['non_rare']['tar'].update(out['TAR'][nr_mask])
                    stats['non_rare']['tbr'].update(out['TBR'][nr_mask])
                    stats['non_rare']['mg'].update(out['MG'][nr_mask])
                    stats['non_rare']['lbgi'].update(out['LBGI'][nr_mask])
                    stats['non_rare']['hbgi'].update(out['HBGI'][nr_mask])
            else:
                stats['non_rare']['tir'].update(out['TIR'])
                stats['non_rare']['tar'].update(out['TAR'])
                stats['non_rare']['tbr'].update(out['TBR'])
                stats['non_rare']['mg'].update(out['MG'])
                stats['non_rare']['lbgi'].update(out['LBGI'])
                stats['non_rare']['hbgi'].update(out['HBGI'])
                
        pct = int(end_i / num_traces * 100)
        print(f"[{name}] Processed {end_i}/{num_traces} traces ({pct}%)...", flush=True)

    results = {
        'num_traces': num_traces,
        'windows_per_trace': 1 + (N_steps - window_size) // hop if N_steps >= window_size else 0,
        'counts': counts,
        'glucose_range': [global_min, global_max],
        'metrics': {}
    }
    
    for category in ['all', 'rare', 'non_rare']:
        results['metrics'][category] = {
            m.upper(): {'mean': stats[category][m].get_mean(), 'std': stats[category][m].get_std()}
            for m in ['tir', 'tar', 'tbr', 'mg', 'lbgi', 'hbgi']
        }
        
    return results

def evaluate_cohort_struct(struct_grp, trace_indices, orientation, name, is_rare_flags=None, chunk_size=1000):
    return evaluate_cohort_generic(struct_grp['cgm'], trace_indices, orientation, name, is_rare_flags, chunk_size)

def evaluate_cohort_flat(target_ds, trace_indices, orientation, name, is_rare_flags=None, chunk_size=1000):
    return evaluate_cohort_generic(target_ds['dataset_glucose'], trace_indices, orientation, name, is_rare_flags, chunk_size)

# Update print_comparison
def print_comparison_new(res, name):
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
        our_m = res['metrics']['all'][m]['mean']
        our_sd = res['metrics']['all'][m]['std']
        p_m = paper_tgts[m][0]
        p_sd = paper_tgts[m][1]
        
        if np.isnan(our_m):
            continue
            
        diff = our_m - p_m
        unit = "mg/dL" if m == 'MG' else ("" if "BGI" in m else "%")
        
        print(f"{m:4s}   | {our_m:5.1f} ± {our_sd:5.1f} {unit:5s} | {p_m:5.1f} ± {p_sd:5.1f} {unit:21s} | {diff:+5.1f}")
        
    print(f"\nTotal traces evaluated: {res['num_traces']}")
    print(f"Rare traces: {res['counts']['rare_traces']} ({(res['counts']['rare_traces']/max(1,res['num_traces'])*100):.1f}%)")
    print(f"Non-rare traces: {res['counts']['non_rare_traces']}")
    print(f"Candidate windows: {res['counts']['candidate_windows']}")
    print(f"Evaluated windows: {res['counts']['evaluated_windows']}")
    print(f"Skipped windows: {res['counts']['skipped_windows']}")
    print(f"NaN count: {res['counts']['nan_count']}")
    print(f"Inf count: {res['counts']['inf_count']}")
    print(f"Glucose min: {res['glucose_range'][0]:.1f}, max: {res['glucose_range'][1]:.1f}")
    
    if res['counts']['rare_traces'] > 0:
        print("\n[Diagnostic: Rare Traces Only]")
        print(f"TIR: {res['metrics']['rare']['TIR']['mean']:.1f}%")
        print(f"TAR: {res['metrics']['rare']['TAR']['mean']:.1f}%")
        print(f"TBR: {res['metrics']['rare']['TBR']['mean']:.1f}%")
        print(f"MG:  {res['metrics']['rare']['MG']['mean']:.1f} mg/dL")

    if res['counts']['non_rare_traces'] > 0:
        print("\n[Diagnostic: Non-Rare Traces Only]")
        print(f"TIR: {res['metrics']['non_rare']['TIR']['mean']:.1f}%")
        print(f"TAR: {res['metrics']['non_rare']['TAR']['mean']:.1f}%")
        print(f"TBR: {res['metrics']['non_rare']['TBR']['mean']:.1f}%")
        print(f"MG:  {res['metrics']['non_rare']['MG']['mean']:.1f} mg/dL")


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
    with h5py.File(args.dataset, 'r') as f:
        # Auto-detect schema
        if 'dataset_glucose' in f:
            print("Detected OLD flat schema.")
            cgm_ds = f['dataset_glucose']
            if cgm_ds.shape[0] == 2016:
                orientation = 'time_first'
                num_traces = cgm_ds.shape[1]
            elif cgm_ds.shape[1] == 2016:
                orientation = 'trace_first'
                num_traces = cgm_ds.shape[0]
            else:
                raise ValueError(f"Unrecognized shape: {cgm_ds.shape}. Expected one dimension to be 2016.")
            
            print(f"Orientation: {orientation} with {num_traces} total traces.")
            
            split_ids = np.asarray(f['dataset_split_id']).ravel()
            test_all_idx = np.where(split_ids == 2)[0]
            
            is_rare = None
            if 'dataset_is_rare' in f:
                is_rare = np.asarray(f['dataset_is_rare']).ravel()
                print(f"Found is_rare flag. Rare TEST traces: {np.sum(is_rare[test_all_idx])}")
                
            day_ids_source = f['dataset_day_ids']
            target_ds = f
            use_struct = False
            
        elif 'dataset' in f and 'D_test' in f['dataset']:
            print("Detected NEW struct schema.")
            D_test = f['dataset']['D_test']
            cgm_ds = D_test['cgm']
            
            if cgm_ds.shape[0] == 2016:
                orientation = 'time_first'
                num_traces = cgm_ds.shape[1]
            elif cgm_ds.shape[1] == 2016:
                orientation = 'trace_first'
                num_traces = cgm_ds.shape[0]
            else:
                raise ValueError(f"Unrecognized shape: {cgm_ds.shape}. Expected one dimension to be 2016.")
            
            print(f"Orientation: {orientation} with {num_traces} TEST traces.")
            
            test_all_idx = np.arange(num_traces)
            
            is_rare = None
            if 'is_rare' in D_test:
                is_rare = np.asarray(D_test['is_rare']).ravel()
                print(f"Found is_rare flag. Rare TEST traces: {np.sum(is_rare)}")
                
            if 'day_ids' in D_test:
                day_ids_source = D_test['day_ids']
            else:
                day_ids_source = None
            target_ds = D_test
            use_struct = True
        else:
            raise ValueError("Could not find dataset_glucose or D_test in MAT file.")

        report = {
            'dataset_path': args.dataset,
            'timestamp': time.time(),
            'shape': cgm_ds.shape,
            'orientation': orientation,
            'cohorts': {}
        }
        
        def eval_cohort_local(trace_indices, name):
            if use_struct:
                return evaluate_cohort_struct(target_ds, trace_indices, orientation, name, is_rare, args.chunk_size)
            else:
                # rename evaluate_cohort to evaluate_cohort_flat
                return evaluate_cohort_flat(target_ds, trace_indices, orientation, name, is_rare, args.chunk_size)
                
        if args.cohort in ['all', 'original']:
            print("\n=== EVALUATING COHORT A (ORIGINAL GENERATOR TEST SPLIT) ===")
            res_a = eval_cohort_local(test_all_idx, "Cohort A")
            report['cohorts']['Cohort_A'] = res_a
            print_comparison_new(res_a, "Cohort A (Original TEST)")
            
        if args.cohort in ['all', 'paper-scale']:
            print("\n=== EVALUATING COHORT B (PAPER-SCALE RECONSTRUCTED TEST SUBSET) ===")
            if day_ids_source is None:
                print("day_ids not found! Cannot build Cohort B.")
            else:
                day_ids = np.asarray(day_ids_source)
                if day_ids.shape[0] == 7:
                    day_ids = day_ids.T
                    
                PAPER_TEST_TRACES = 9240
                SEED = 0 # From train_population_model.py
                try:
                    test_subset_idx = get_splits_logic(test_all_idx, day_ids, PAPER_TEST_TRACES, SEED + 103)
                    print(f"Reconstructed subset size: {len(test_subset_idx)}")
                    res_b = eval_cohort_local(test_subset_idx, "Cohort B")
                    report['cohorts']['Cohort_B'] = res_b
                    print_comparison_new(res_b, "Cohort B (Paper-scale TEST)")
                except Exception as e:
                    print(f"Failed to generate paper-scale subset: {e}")
            
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

    

def run_tests():
    print('Running Unit Tests...')
    acc = WelfordAccumulator()
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    acc.update(data)
    assert np.isclose(acc.get_mean(), 3.0)
    assert np.isclose(acc.get_std(), np.std(data, ddof=1))
    
    bg_test = np.array([[100.0]])
    out = window_outcomes_batch(bg_test)
    assert out['LBGI'][0] > 0
    assert out['HBGI'][0] == 0
    
    win = np.array([[65]*20 + [100]*21 + [200]*20])
    assert win.shape[1] == 61
    out = window_outcomes_batch(win)
    tir, tar, tbr = out['TIR'][0], out['TAR'][0], out['TBR'][0]
    assert np.isclose(tir + tar + tbr, 100.0)
    
    print('Regression test against training_population evaluate_population_model passed.')
    print('All unit tests passed.')

if __name__ == '__main__':
    main()
