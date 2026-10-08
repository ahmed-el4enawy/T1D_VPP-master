import argparse
import json
import os
import numpy as np
import scipy.io

def fingerprint_dataset(mat_path):
    print(f"Loading dataset from: {mat_path}")
    if not os.path.isfile(mat_path):
        print(f"File not found: {mat_path}")
        return

    try:
        data = scipy.io.loadmat(mat_path, simplify_cells=True)
        if 'dataset' in data:
            D_train = data['dataset']['D_train']
            D_val = data['dataset']['D_val']
            D_test = data['dataset']['D_test']
        elif 'D_train' in data:
            D_train = data['D_train']
            D_val = data['D_val']
            D_test = data['D_test']
        else:
            print("Dataset structure not recognized.")
            return
    except NotImplementedError:
        import h5py
        print("v7.3 MAT file detected, loading via h5py...")
        f = h5py.File(mat_path, 'r')
        
        def extract_struct(name):
            if 'dataset' in f and name in f['dataset']:
                grp = f['dataset'][name]
            elif name in f:
                grp = f[name]
            else:
                return {}
            
            res = {}
            for k in grp.keys():
                res[k] = np.array(grp[k])
                
            return res
            
        D_train = extract_struct('D_train')
        D_val = extract_struct('D_val')
        D_test = extract_struct('D_test')


    # Combine traces for overall statistics
    # U_cube: [N_steps, 2, n_traces]. index 0: insulin (U/hr), index 1: carbs (g)
    def extract_stats(D_split):
        if 'u' not in D_split or D_split['u'].size == 0:
            return None
            
        u = D_split['u']
        cgm = D_split['cgm']
        
        # Initial glucose is at index 0 of CGM (assuming 5-min intervals)
        if cgm.ndim == 2:
            initial_glucose = cgm[0, :]
        else:
            initial_glucose = np.array([])
            
        # Carbs
        if u.ndim == 3:
            u_carbs = u[:, 1, :]
            # Total carbs per trace
            total_carbs = np.sum(u_carbs, axis=0)
        else:
            total_carbs = np.array([])
            
        return {
            'initial_glucose_mean': float(np.mean(initial_glucose)) if len(initial_glucose) > 0 else 0.0,
            'initial_glucose_std': float(np.std(initial_glucose)) if len(initial_glucose) > 0 else 0.0,
            'total_carbs_7d_mean': float(np.mean(total_carbs)) if len(total_carbs) > 0 else 0.0,
            'total_carbs_7d_std': float(np.std(total_carbs)) if len(total_carbs) > 0 else 0.0,
            'n_traces': int(u.shape[2]) if u.ndim == 3 else 0
        }

    stats = {
        'Train': extract_stats(D_train),
        'Val': extract_stats(D_val),
        'Test': extract_stats(D_test)
    }
    
    overall = {}
    for k in ['initial_glucose_mean', 'total_carbs_7d_mean']:
        vals = [s[k] * s['n_traces'] for s in stats.values() if s is not None and s['n_traces'] > 0]
        total_n = sum([s['n_traces'] for s in stats.values() if s is not None])
        if total_n > 0:
            overall[k] = sum(vals) / total_n
        else:
            overall[k] = 0.0

    stats['Overall_Estimates'] = overall
    
    print("\nStatistical Fingerprint:")
    print(json.dumps(stats, indent=2))
    
    with open('dataset_fingerprint_report.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("Saved report to dataset_fingerprint_report.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Statistically fingerprint the generated dataset')
    parser.add_argument('--dataset_path', type=str, default='test_sample_dataset.mat', help='Path to MAT dataset')
    args = parser.parse_args()
    
    fingerprint_dataset(args.dataset_path)
