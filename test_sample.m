% test_sample.m
% Baseline measurement harness for T1D_VPP generator

clc; clear; close all;

%% 1. Run the Generator
opts.num_scenarios = 50;
opts.variants_per_scenario = 5;
opts.random_seed = 42;
opts.days_per_scenario = 7;
opts.dt = 5;
opts.include_rare_events = true;
test_save_path = 'test_sample_dataset.mat';

fprintf('Running generate_population_dataset with %d scenarios...\n', opts.num_scenarios);

% Call the generator exactly as it currently behaves
[dataset, meta] = generate_population_dataset('num_scenarios', opts.num_scenarios, ...
    'variants_per_scenario', opts.variants_per_scenario, ...
    'random_seed', opts.random_seed, ...
    'days_per_scenario', opts.days_per_scenario, ...
    'dt', opts.dt, ...
    'include_rare_events', opts.include_rare_events, ...
    'save_path', test_save_path);

%% 2. Evaluate ONLY D_test
D_test = dataset.D_test;
D_train = dataset.D_train;
D_val = dataset.D_val;

cgm_test = D_test.cgm; % Size: [N_steps, n_test]
is_rare_test = D_test.is_rare;

%% 3. & 4. Split every 7-day TEST trace into NON-OVERLAPPING 5-hour windows
% 7 days = 10080 minutes
% dt = 5 min -> N_steps = 2016
% 5-hour window = 300 minutes = 60 intervals (advance)
% 61 glucose points per window (inclusive of start and end)
window_size = 61;
advance = 60;
N_steps = size(cgm_test, 1);
n_test = size(cgm_test, 2);

% Calculate number of windows per trace
num_windows_per_trace = floor((N_steps - 1) / advance);

% Arrays to store metrics for all evaluated windows
all_tir = [];
all_tar = [];
all_tbr = [];
all_mg = [];
all_lbgi = [];
all_hbgi = [];
num_skipped = 0;

%% 5. & 6. Calculate Metrics for every 5-hour window
for i = 1:n_test
    trace = cgm_test(:, i);
    for w = 1:num_windows_per_trace
        start_idx = (w - 1) * advance + 1;
        end_idx = start_idx + window_size - 1;
        
        if end_idx > N_steps
            break;
        end
        
        win_cgm = trace(start_idx:end_idx);
        
        if any(isnan(win_cgm)) || any(isinf(win_cgm))
            num_skipped = num_skipped + 1;
            continue;
        end
        
        % Standard metrics
        tir = mean(win_cgm >= 70 & win_cgm <= 180) * 100;
        tar = mean(win_cgm > 180) * 100;
        tbr = mean(win_cgm < 70) * 100;
        mg = mean(win_cgm);
        
        % Kovatchev Risk transformation for LBGI / HBGI
        % f(BG) = 1.509 * ( (ln(BG))^1.084 - 5.381 )
        f_bg = 1.509 * ( (log(win_cgm)).^1.084 - 5.381 );
        
        rl = zeros(size(f_bg));
        rh = zeros(size(f_bg));
        
        rl(f_bg < 0) = 10 * (f_bg(f_bg < 0)).^2;
        rh(f_bg > 0) = 10 * (f_bg(f_bg > 0)).^2;
        
        lbgi = mean(rl);
        hbgi = mean(rh);
        
        all_tir(end+1) = tir;
        all_tar(end+1) = tar;
        all_tbr(end+1) = tbr;
        all_mg(end+1) = mg;
        all_lbgi(end+1) = lbgi;
        all_hbgi(end+1) = hbgi;
    end
end

num_evaluated_windows = numel(all_tir);

%% 9. Basic Stats Printout
n_train = size(D_train.cgm, 2);
n_val = size(D_val.cgm, 2);
total_traces = n_train + n_val + n_test;

rare_train = sum(D_train.is_rare);
rare_val = sum(D_val.is_rare);
rare_test = sum(D_test.is_rare);

fprintf('\n=== Generation Statistics ===\n');
fprintf('Generated Train Traces : %d (Rare: %d)\n', n_train, rare_train);
fprintf('Generated Val Traces   : %d (Rare: %d)\n', n_val, rare_val);
fprintf('Generated Test Traces  : %d (Rare: %d)\n', n_test, rare_test);
fprintf('Total Traces           : %d\n', total_traces);
fprintf('5-hour TEST windows evaluated       : %d\n', num_evaluated_windows);
fprintf('5-hour TEST windows skipped (NaN/Inf) : %d\n', num_skipped);

%% 7. & 8. Print comparison table
fprintf('\n=== Baseline Metrics vs Paper (DP_test) ===\n');
fprintf('Metric | Baseline (Mean ± SD)     | Paper (Mean ± SD)\n');
fprintf('-----------------------------------------------------\n');
if num_evaluated_windows > 0
    fprintf('TIR    | %5.1f ± %5.1f %%        | 72.9 ± 35.2 %%\n', mean(all_tir), std(all_tir));
    fprintf('TAR    | %5.1f ± %5.1f %%        | 23.1 ± 36.3 %%\n', mean(all_tar), std(all_tar));
    fprintf('TBR    | %5.1f ± %5.1f %%        |  4.0 ±  8.2 %%\n', mean(all_tbr), std(all_tbr));
    fprintf('LBGI   | %5.1f ± %5.1f          |  1.5 ±  1.7\n', mean(all_lbgi), std(all_lbgi));
    fprintf('HBGI   | %5.1f ± %5.1f          |  4.2 ±  6.2\n', mean(all_hbgi), std(all_hbgi));
    fprintf('MG     | %5.1f ± %5.1f mg/dL    | 132.6 ± 48.2 mg/dL\n', mean(all_mg), std(all_mg));
else
    fprintf('No windows evaluated.\n');
end

%% 13. Warnings
fprintf('\n========================================================================\n');
fprintf(' WARNING:\n');
fprintf(' - 50 base scenarios is ONLY a smoke-test sample.\n');
fprintf(' - The metrics are expected to have high sampling variance.\n');
fprintf(' - This run must NOT be used to conclude that a generator change is correct or incorrect.\n');
fprintf(' - Promising changes should later be retested with 500-1000 scenarios before full generation.\n');
fprintf('========================================================================\n');
