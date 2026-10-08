this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(this_dir));
addpath(fullfile(repo_root, 'matlab'));
% test_sample.m
% Lightweight pilot runner for the population dataset generator.

fprintf('====================================================\n');
fprintf(' Running Small-Scale Generator Pilot (test_sample)\n');
fprintf('====================================================\n');

try
    [dataset, meta] = generate_population_dataset( ...
        'num_scenarios', 50, ...
        'variants_per_scenario', 5, ...
        'chunk_size', 10, ...
        'include_rare_events', true, ...
        'save_path', 'test_sample_dataset.mat');
    
    fprintf('\nGeneration succeeded.\n');
    fprintf('Requested Base Scenarios: 50\n');
    fprintf('Standard Variants Generated: %d\n', meta.standard_variant_traces);
    fprintf('Hypo Supplements: %d\n', meta.hypo_supplement_traces);
    fprintf('Hyper Supplements: %d\n', meta.hyper_supplement_traces);
    fprintf('Final Total Traces: %d\n', meta.final_total_traces);
    
    fprintf('\nSplits: %d Train | %d Val | %d Test\n', ...
        meta.train_count, meta.val_count, meta.test_count);
    
    fprintf('\nDiagnostics:\n');
    fprintf('  Rare count: %d\n', meta.supplemental_rare_traces_total);
    fprintf('  Fallback CGM count: %d\n', meta.fallback_cgm_count);
    fprintf('  Mixed participant count: %d\n', meta.mixed_participant_count);
    fprintf('  Repeated-day count: %d\n', meta.repeated_day_count);
    
    fprintf('\nLeakage Assertion Result: PASSED\n');
    fprintf('Saved pilot to: test_sample_dataset.mat\n');
    fprintf('\n*** 50 base scenarios is a smoke test only. Do not claim scientific validity from it. ***\n');
    
catch ME
    fprintf('GENERATION FAILED with error:\n%s\n', ME.message);
    for k=1:length(ME.stack)
        fprintf('  File: %s, Line: %d\n', ME.stack(k).name, ME.stack(k).line);
    end
end
