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
    fprintf('Final traces generated: %d\n', meta.final_total_traces);
    fprintf('Saved pilot to: test_sample_dataset.mat\n');
    
    if meta.fallback_cgm_count > 0
        fprintf('WARNING: %d base scenarios used the pooled fallback CGM stats.\n', meta.fallback_cgm_count);
    end
    
catch ME
    fprintf('GENERATION FAILED with error:\n%s\n', ME.message);
    for k=1:length(ME.stack)
        fprintf('  File: %s, Line: %d\n', ME.stack(k).name, ME.stack(k).line);
    end
end
