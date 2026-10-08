import numpy as np
from dataset_baseline import window_outcomes_batch, WelfordAccumulator

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
    print(f"\\n--- {name} Results ---")
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
        
    print(f"\\nTotal traces evaluated: {res['num_traces']}")
    print(f"Rare traces: {res['counts']['rare_traces']} ({(res['counts']['rare_traces']/max(1,res['num_traces'])*100):.1f}%)")
    print(f"Non-rare traces: {res['counts']['non_rare_traces']}")
    print(f"Candidate windows: {res['counts']['candidate_windows']}")
    print(f"Evaluated windows: {res['counts']['evaluated_windows']}")
    print(f"Skipped windows: {res['counts']['skipped_windows']}")
    print(f"NaN count: {res['counts']['nan_count']}")
    print(f"Inf count: {res['counts']['inf_count']}")
    print(f"Glucose min: {res['glucose_range'][0]:.1f}, max: {res['glucose_range'][1]:.1f}")
    
    if res['counts']['rare_traces'] > 0:
        print("\\n[Diagnostic: Rare Traces Only]")
        print(f"TIR: {res['metrics']['rare']['TIR']['mean']:.1f}%")
        print(f"TAR: {res['metrics']['rare']['TAR']['mean']:.1f}%")
        print(f"TBR: {res['metrics']['rare']['TBR']['mean']:.1f}%")
        print(f"MG:  {res['metrics']['rare']['MG']['mean']:.1f} mg/dL")

    if res['counts']['non_rare_traces'] > 0:
        print("\\n[Diagnostic: Non-Rare Traces Only]")
        print(f"TIR: {res['metrics']['non_rare']['TIR']['mean']:.1f}%")
        print(f"TAR: {res['metrics']['non_rare']['TAR']['mean']:.1f}%")
        print(f"TBR: {res['metrics']['non_rare']['TBR']['mean']:.1f}%")
        print(f"MG:  {res['metrics']['non_rare']['MG']['mean']:.1f} mg/dL")
