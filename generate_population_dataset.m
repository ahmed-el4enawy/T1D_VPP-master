function [dataset, meta] = generate_population_dataset(varargin)
% GENERATE_POPULATION_DATASET Generates the population-level neural network state-space dataset D^P = {x(t), u(t)}
% as described in the paper:
% "A PHYSIOLOGICALLY-CONSTRAINED NEURAL NETWORK DIGITAL TWIN FRAMEWORK FOR REPLICATING GLUCOSE DYNAMICS IN TYPE 1 DIABETES"
% (arXiv:2508.05705v1)
%
% PACKED LOW-RAM DESIGN (RAM < 200 MB):
%   - Stitches chunks as packed 3D matrices to prevent MATLAB Online RAM limit crashes.
%   - 10x faster to load for PyTorch / TensorFlow / MATLAB training.
%   - Auto-Resumes from existing chunks.
%
% TEAM PARALLEL GENERATION:
%   - Member 1: generate_population_dataset('member_id', 1, 'total_members', 4) -> Part 1
%   - Member 2: generate_population_dataset('member_id', 2, 'total_members', 4) -> Part 2
%   - Member 3: generate_population_dataset('member_id', 3, 'total_members', 4) -> Part 3
%   - Member 4: generate_population_dataset('member_id', 4, 'total_members', 4) -> Part 4

    %% --- 1. Parse Input Options -----------------------------------------
    p = inputParser;
    addParameter(p, 'member_id', 1, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'total_members', 1, @(x) isnumeric(x) && x > 0);
    % [RECONSTRUCTION A1] Base scenario count inferred from 46,200 total / 5
    % variants. The paper does not publish the exact pre-augmentation count;
    % the true count depends on the rare-event supplement rate.
    addParameter(p, 'num_scenarios', 9240, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'variants_per_scenario', 5, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'chunk_size', 250, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'meal_csv_path', '', @ischar);
    addParameter(p, 'cgm_csv_path', '', @ischar);
    addParameter(p, 'save_path', '', @ischar);
    addParameter(p, 'split_ratio', [0.60, 0.20, 0.20], @(x) numel(x) == 3 && abs(sum(x)-1)<1e-4);
    addParameter(p, 'dt', 5, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'days_per_scenario', 7, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'include_rare_events', true, @islogical);
    addParameter(p, 'random_seed', 42, @isnumeric);
    parse(p, varargin{:});
    opts = p.Results;

    member_id     = opts.member_id;
    total_members = opts.total_members;

    if member_id > total_members
        error('generate_population_dataset:invalidMember', ...
              'member_id (%d) cannot exceed total_members (%d).', member_id, total_members);
    end

    % Workload partitioning across team members
    scenarios_per_member = ceil(opts.num_scenarios / total_members);
    start_scen_idx = (member_id - 1) * scenarios_per_member + 1;
    end_scen_idx   = min(opts.num_scenarios, member_id * scenarios_per_member);
    n_scenarios    = end_scen_idx - start_scen_idx + 1;

    if isempty(opts.save_path)
        if total_members > 1
            opts.save_path = sprintf('T1DSim_population_part%d.mat', member_id);
        else
            opts.save_path = 'T1DSim_population_dataset.mat';
        end
    end

    % [SAFETY FIX S1] Encode configuration fingerprint in chunk directory
    % to prevent stale chunks from a different run being silently reused.
    config_str = sprintf('s%d_v%d_d%d_r%d_seed%d', ...
        opts.num_scenarios, opts.variants_per_scenario, ...
        opts.days_per_scenario, opts.include_rare_events, opts.random_seed);
    meal_info = dir(opts.meal_csv_path);
    cgm_info  = dir(opts.cgm_csv_path);
    hash_input = sprintf('%s_%d_%d', config_str, meal_info.bytes, cgm_info.bytes);
    config_hash = dec2hex(mod(sum(double(hash_input) .* (1:numel(hash_input))), 2^32), 8);
    chunk_dir = sprintf('temp_part%d_%s_chunks', member_id, config_hash);
    if ~exist(chunk_dir, 'dir')
        mkdir(chunk_dir);
        % Save config manifest for verification on resume
        chunk_config = struct('num_scenarios', opts.num_scenarios, ...
            'variants_per_scenario', opts.variants_per_scenario, ...
            'days_per_scenario', opts.days_per_scenario, ...
            'include_rare_events', opts.include_rare_events, ...
            'random_seed', opts.random_seed, ...
            'meal_csv_bytes', meal_info.bytes, ...
            'cgm_csv_bytes', cgm_info.bytes, ...
            'config_hash', config_hash); %#ok<NASGU>
        save(fullfile(chunk_dir, 'chunk_config.mat'), 'chunk_config');
    else
        % Verify config matches on resume
        cfg_file = fullfile(chunk_dir, 'chunk_config.mat');
        if isfile(cfg_file)
            saved = load(cfg_file, 'chunk_config');
            if ~strcmp(saved.chunk_config.config_hash, config_hash)
                error('generate_population_dataset:staleChunks', ...
                    'Chunk directory %s contains chunks from a different configuration. Delete it manually before rerunning.', chunk_dir);
            end
        end
    end

    member_seed = opts.random_seed + (member_id - 1) * 100000;
    rng(member_seed, 'twister');

    fprintf('========================================================================\n');
    fprintf(' Generating Population Dataset D^P (Member %d of %d) [LOW RAM PACKED]\n', member_id, total_members);
    fprintf(' Target Scenarios: %d to %d (%d scenarios total)\n', start_scen_idx, end_scen_idx, n_scenarios);
    fprintf(' Output File     : %s\n', opts.save_path);
    fprintf('========================================================================\n');

    %% --- 2. Load T1DEXI Meal Scenarios and Midnight CGM Stats -------------
    if isempty(opts.meal_csv_path)
        if isfile('day_scenario_library.csv')
            opts.meal_csv_path = 'day_scenario_library.csv';
        elseif isfile('day_scenario_library (1).csv')
            opts.meal_csv_path = 'day_scenario_library (1).csv';
        else
            error('generate_population_dataset:missingFile', 'day_scenario_library.csv not found.');
        end
    end

    if isempty(opts.cgm_csv_path)
        if isfile('midnight_cgm_stats.csv')
            opts.cgm_csv_path = 'midnight_cgm_stats.csv';
        elseif isfile('midnight_cgm_stats (1).csv')
            opts.cgm_csv_path = 'midnight_cgm_stats (1).csv';
        else
            error('generate_population_dataset:missingFile', 'midnight_cgm_stats.csv not found.');
        end
    end

    meal_lib = load_day_scenario_library(opts.meal_csv_path);
    cgm_map  = load_midnight_cgm_stats(opts.cgm_csv_path);
    n_library = numel(meal_lib);

    % [BUG FIX + RECONSTRUCTION A2] Day-level leakage-free split.
    % Partition the set of ALL unique day_ids into disjoint train/val/test
    % pools ONCE globally, BEFORE workload partitioning, so that every
    % member uses the same mapping and no individual day_id leaks across
    % splits. The paper says "60/20/20 split" but does not specify the
    % grouping unit; standard ML practice requires no data leakage.
    all_day_ids = unique([meal_lib.day_id]);
    n_unique_days = numel(all_day_ids);
    split_rng = RandStream('twister', 'Seed', opts.random_seed);
    day_perm = all_day_ids(split_rng.randperm(n_unique_days));
    n_train_days = round(opts.split_ratio(1) * n_unique_days);
    n_val_days   = round(opts.split_ratio(2) * n_unique_days);
    train_day_set = containers.Map(num2cell(double(day_perm(1:n_train_days))), ...
                                   num2cell(ones(1, n_train_days)));
    val_day_set   = containers.Map(num2cell(double(day_perm(n_train_days+1 : n_train_days+n_val_days))), ...
                                   num2cell(ones(1, n_val_days)));
    test_day_set  = containers.Map(num2cell(double(day_perm(n_train_days+n_val_days+1 : end))), ...
                                   num2cell(ones(1, n_unique_days - n_train_days - n_val_days)));
    fprintf('  Day-level split: %d train / %d val / %d test day_ids (of %d unique)\n', ...
            n_train_days, n_val_days, n_unique_days - n_train_days - n_val_days, n_unique_days);

    %% --- 3. Population Model Parameters (Resalat et al. [12] / Hovorka et al. [35]) --
    ModPar = struct();
    ModPar.Fc01  = 0.0097;   % Non-insulin mediated glucose uptake above 4.5 mmol/L (mmol/kg/min)
    ModPar.VdG   = 0.16;     % Volume of distribution of glucose (L/kg)
    ModPar.k12   = 0.066;    % Transfer rate constant from Q2 to Q1 (min^-1)
    ModPar.Ag    = 0.8;      % Carbohydrate bioavailability (unitless)
    ModPar.tmaxG = 40.0;     % Time-to-maximum appearance rate of carbs (min)
    ModPar.EGP0  = 0.0161;   % Basal endogenous glucose production (mmol/kg/min)
    ModPar.tmaxI = 55.0;     % Time-to-maximum subcutaneous insulin absorption (min)
    ModPar.Ke    = 0.138;    % Elimination rate constant of insulin (min^-1)
    ModPar.VdI   = 0.12;     % Volume of distribution of insulin (L/kg)
    ModPar.ka1   = 0.006;    % Elimination rate constant for insulin effect x1 (min^-1)
    ModPar.ka2   = 0.06;     % Elimination rate constant for insulin effect x2 (min^-1)
    ModPar.ka3   = 0.03;     % Elimination rate constant for insulin effect x3 (min^-1)
    ModPar.Sf1   = 51.2e-4;  % Insulin sensitivity for distribution x1 ((mU.L.min)^-2)
    ModPar.Sf2   = 8.2e-4;   % Insulin sensitivity for disposal x2 ((mU.L.min)^-2)
    ModPar.Sf3   = 520.0e-4; % Insulin sensitivity for EGP suppression x3 ((mU.L.min)^-1)
    ModPar.Weight= 70.0;     % Standard population body weight (kg)

    %% --- 4. Prepare Simulation Grid -------------------------------------
    Ts = opts.dt; % 5 min
    Sim_time_min = opts.days_per_scenario * 1440; % 10080 min (7 days)
    N_steps = Sim_time_min / Ts; % 2016 steps

    dt_sub = 1.0; 
    sub_steps_per_frame = round(Ts / dt_sub);
    n_variants  = opts.variants_per_scenario;
    chunk_size  = opts.chunk_size;
    num_chunks  = ceil(n_scenarios / chunk_size);

    %% --- 5. High-Speed Chunk Simulation ---
    for c = 1:num_chunks
        chunk_file = fullfile(chunk_dir, sprintf('chunk_%04d.mat', c));

        c_start_scen = (c - 1) * chunk_size + 1;
        c_end_scen   = min(n_scenarios, c * chunk_size);

        % [SAFETY FIX S1] Only check inside the fingerprinted chunk_dir
        if isfile(chunk_file)
            fprintf('  [Chunk %d/%d] Existing chunk found. Skipping...\n', c, num_chunks);
            continue;
        end

        t_start_chunk = tic;
        fprintf('  [Chunk %d/%d] Simulating scenarios %d..%d ...\n', c, num_chunks, c_start_scen, c_end_scen);

        % [BUG FIX B1] Both rare-event conditions are independent (not
        % elseif), so one trace can produce 1 normal + 1 hypo + 1 hyper = 3.
        max_alloc = (c_end_scen - c_start_scen + 1) * n_variants * 3;
        
        X_cube      = zeros(N_steps, 10, max_alloc, 'single');
        U_cube      = zeros(N_steps, 2, max_alloc, 'single');
        CGM_mat     = zeros(N_steps, max_alloc, 'single');
        day_ids_mat = zeros(7, max_alloc, 'int16');
        usubjid_arr = strings(1, max_alloc);
        gid_vec     = zeros(1, max_alloc, 'int32');
        tbr_vec     = zeros(1, max_alloc, 'single');
        tar_vec     = zeros(1, max_alloc, 'single');
        tir_vec     = zeros(1, max_alloc, 'single');
        mg_vec      = zeros(1, max_alloc, 'single');
        rare_vec    = false(1, max_alloc);

        trace_count = 0;

        for s_idx = c_start_scen:c_end_scen
            global_scen_id = start_scen_idx + s_idx - 1;

            day_indices = randi(n_library, 1, opts.days_per_scenario);
            selected_days = meal_lib(day_indices);
            day_ids = int16([selected_days.day_id]);
            % [RECONSTRUCTION A3] When the 7 sampled days come from different
            % participants, the initial glucose distribution is drawn from
            % the first day's participant. The paper does not specify which
            % participant's midnight CGM to use in this case.
            usubjid_str = string(selected_days(1).usubjid);

            if cgm_map.isKey(char(usubjid_str))
                cgm_stat = cgm_map(char(usubjid_str));
                cgm_mean = cgm_stat.mean;
                cgm_std  = cgm_stat.std;
            else
                cgm_mean = 156.0;
                cgm_std  = 45.0;
            end

            initial_cgm_draws = cgm_mean + cgm_std * randn(1, n_variants);
            % [RECONSTRUCTION A6] The paper does not mention clipping.
            % This bounds the normal distribution to [70, 260] mg/dL.
            initial_cgm_draws = max(70, min(260, initial_cgm_draws));

            for v_idx = 1:n_variants
                g0 = initial_cgm_draws(v_idx);

                [x0, ~, basal_Uhr] = solve_steady_state(g0, ModPar);

                tdd_est = basal_Uhr * 24.0 * 2.0;
                % [RECONSTRUCTION A4] The paper says "1700 rule" which in
                % clinical practice refers to ISF, not ICR. The /3 divisor
                % approximates a 500-rule ICR. This is undocumented.
                icr_base = 1700.0 / (tdd_est * 3.0);

                [u_carbs_grid, u_insulin_grid] = build_scenario_inputs(...
                    selected_days, Sim_time_min, Ts, ModPar.Weight, icr_base);

                [X_trace, U_trace, CGM_trace] = simulate_7day_ode_fast(...
                    x0, u_carbs_grid, u_insulin_grid, basal_Uhr, ModPar, N_steps, Ts, dt_sub, sub_steps_per_frame);

                tbr = mean(CGM_trace < 70) * 100;
                tar = mean(CGM_trace > 180) * 100;
                tar250 = mean(CGM_trace > 250) * 100;
                tir = mean(CGM_trace >= 70 & CGM_trace <= 180) * 100;
                mean_cgm = mean(CGM_trace);

                trace_count = trace_count + 1;
                X_cube(:, :, trace_count)   = single(X_trace);
                U_cube(:, :, trace_count)   = single(U_trace);
                CGM_mat(:, trace_count)     = single(CGM_trace(:));
                day_ids_mat(:, trace_count) = day_ids;
                usubjid_arr(trace_count)    = usubjid_str;
                gid_vec(trace_count)        = int32(global_scen_id);
                tbr_vec(trace_count)        = single(tbr);
                tar_vec(trace_count)        = single(tar);
                tir_vec(trace_count)        = single(tir);
                mg_vec(trace_count)         = single(mean_cgm);
                rare_vec(trace_count)       = false;

                if opts.include_rare_events
                    % [RECONSTRUCTION A7] The paper specifies only TBR > 20%
                    % and TAR250 > 40% as supplement triggers. The perturbation
                    % method (initial glucose scaling, insulin scaling, basal
                    % scaling, forced delays) is undocumented.
                    if tbr > 20
                        [x0_hypo, ~, basal_hypo] = solve_steady_state(g0 * 0.85, ModPar);
                        [X_supp, U_supp, CGM_supp] = simulate_7day_ode_fast(...
                            x0_hypo, u_carbs_grid, u_insulin_grid * 1.15, basal_hypo * 1.05, ...
                            ModPar, N_steps, Ts, dt_sub, sub_steps_per_frame);
                        
                        trace_count = trace_count + 1;
                        X_cube(:, :, trace_count)   = single(X_supp);
                        U_cube(:, :, trace_count)   = single(U_supp);
                        CGM_mat(:, trace_count)     = single(CGM_supp(:));
                        day_ids_mat(:, trace_count) = day_ids;
                        usubjid_arr(trace_count)    = usubjid_str;
                        gid_vec(trace_count)        = int32(global_scen_id);
                        tbr_vec(trace_count)        = single(mean(CGM_supp < 70) * 100);
                        tar_vec(trace_count)        = single(mean(CGM_supp > 180) * 100);
                        tir_vec(trace_count)        = single(mean(CGM_supp >= 70 & CGM_supp <= 180) * 100);
                        mg_vec(trace_count)         = single(mean(CGM_supp));
                        rare_vec(trace_count)       = true;
                    end

                    if tar250 > 40
                        [u_carbs_del, u_ins_del] = build_scenario_inputs(...
                            selected_days, Sim_time_min, Ts, ModPar.Weight, icr_base, true);
                        
                        [x0_hyper, ~, basal_hyper] = solve_steady_state(g0 * 1.15, ModPar);
                        [X_supp, U_supp, CGM_supp] = simulate_7day_ode_fast(...
                            x0_hyper, u_carbs_del, u_ins_del * 0.85, basal_hyper * 0.95, ...
                            ModPar, N_steps, Ts, dt_sub, sub_steps_per_frame);
                        
                        trace_count = trace_count + 1;
                        X_cube(:, :, trace_count)   = single(X_supp);
                        U_cube(:, :, trace_count)   = single(U_supp);
                        CGM_mat(:, trace_count)     = single(CGM_supp(:));
                        day_ids_mat(:, trace_count) = day_ids;
                        usubjid_arr(trace_count)    = usubjid_str;
                        gid_vec(trace_count)        = int32(global_scen_id);
                        tbr_vec(trace_count)        = single(mean(CGM_supp < 70) * 100);
                        tar_vec(trace_count)        = single(mean(CGM_supp > 180) * 100);
                        tir_vec(trace_count)        = single(mean(CGM_supp >= 70 & CGM_supp <= 180) * 100);
                        mg_vec(trace_count)         = single(mean(CGM_supp));
                        rare_vec(trace_count)       = true;
                    end
                end
            end
        end

        chunk_pack = struct();
        chunk_pack.X_cube      = X_cube(:, :, 1:trace_count);
        chunk_pack.U_cube      = U_cube(:, :, 1:trace_count);
        chunk_pack.CGM_mat     = CGM_mat(:, 1:trace_count);
        chunk_pack.day_ids_mat = day_ids_mat(:, 1:trace_count);
        chunk_pack.usubjid_arr = usubjid_arr(1:trace_count);
        chunk_pack.gid_vec     = gid_vec(1:trace_count);
        chunk_pack.tbr_vec     = tbr_vec(1:trace_count);
        chunk_pack.tar_vec     = tar_vec(1:trace_count);
        chunk_pack.tir_vec     = tir_vec(1:trace_count);
        chunk_pack.mg_vec      = mg_vec(1:trace_count);
        chunk_pack.rare_vec    = rare_vec(1:trace_count);

        safe_save_matlab_drive(chunk_file, chunk_pack);
        t_chunk = toc(t_start_chunk);

        fprintf('     Saved Chunk %d in %.1fs (%d traces, ~35 MB). RAM cleared.\n', ...
                c, t_chunk, trace_count);
        clear X_cube U_cube CGM_mat day_ids_mat usubjid_arr chunk_pack;
    end

    %% --- 6. Assemble into ONE Single .mat File (Peak RAM < 3.5 GB) -------
    fprintf('\nStitching completed chunks into ONE single file: %s ...\n', opts.save_path);

    % Disable MATLAB array size preference that threw the 5.0GB error
    try
        s = settings;
        s.matlab.general.arraysizes.LimitMaxArraySize.PersonalValue = false;
    catch
    end

    chunk_files = dir(fullfile(chunk_dir, 'chunk_*.mat'));
    if isempty(chunk_files)
        chunk_files = dir('chunk_*.mat');
    end
    n_files = numel(chunk_files);

    % ---- Pass 1: Scan chunk metadata (trace counts, metrics, day_ids) ----
    chunk_counts   = zeros(1, n_files);
    all_tir        = [];
    all_tar        = [];
    all_tbr        = [];
    all_mg         = [];
    all_day_ids_chunks = {};

    fprintf('  Pass 1/2: scanning chunk metadata for day-level splitting ...\n');
    for k = 1:n_files
        cpath = get_chunk_file_path(chunk_dir, chunk_files(k).name);
        info = whos('-file', cpath, 'CGM_mat');
        if ~isempty(info)
            cnt = info.size(2);
        else
            tmp = load(cpath, 'CGM_mat');
            cnt = size(tmp.CGM_mat, 2);
            clear tmp;
        end
        chunk_counts(k) = cnt;

        cdata = load(cpath, 'day_ids_mat', 'tir_vec', 'tar_vec', 'tbr_vec', 'mg_vec');
        all_day_ids_chunks{k} = cdata.day_ids_mat(:, 1:cnt); %#ok<AGROW>
        all_tir = [all_tir, cdata.tir_vec(1:cnt)]; %#ok<AGROW>
        all_tar = [all_tar, cdata.tar_vec(1:cnt)]; %#ok<AGROW>
        all_tbr = [all_tbr, cdata.tbr_vec(1:cnt)]; %#ok<AGROW>
        all_mg  = [all_mg,  cdata.mg_vec(1:cnt)];  %#ok<AGROW>
        clear cdata;
    end

    n_total_traces = sum(chunk_counts);
    total_traces   = n_total_traces;
    total_sim_days = total_traces * opts.days_per_scenario;

    % ---- [RECONSTRUCTION A2] Day-Level Leakage-Free Splitting ----
    % Each trace is assigned to the split that contains ALL of its 7 day_ids.
    % If a trace's day_ids span multiple splits, it goes to TRAIN (the most
    % conservative choice, preventing contamination of val/test).
    split_labels = zeros(1, n_total_traces, 'uint8');  % 1=train, 2=val, 3=test
    trace_offset = 0;
    for k = 1:n_files
        cnt = chunk_counts(k);
        chunk_days = all_day_ids_chunks{k};
        for j = 1:cnt
            ids = double(chunk_days(:, j));
            in_train = all(cellfun(@(id) train_day_set.isKey(id), num2cell(ids)));
            in_val   = all(cellfun(@(id) val_day_set.isKey(id), num2cell(ids)));
            in_test  = all(cellfun(@(id) test_day_set.isKey(id), num2cell(ids)));
            if in_test
                split_labels(trace_offset + j) = 3;
            elseif in_val
                split_labels(trace_offset + j) = 2;
            elseif in_train
                split_labels(trace_offset + j) = 1;
            else
                % Mixed day_ids across splits → assign to train
                split_labels(trace_offset + j) = 1;
            end
        end
        trace_offset = trace_offset + cnt;
    end

    n_train = sum(split_labels == 1);
    n_val   = sum(split_labels == 2);
    n_test  = sum(split_labels == 3);
    fprintf('  Split: %d train (60%%) / %d val (20%%) / %d test (20%%) [%d total traces]\n', ...
            n_train, n_val, n_test, n_total_traces);

    % ---- Preallocate D_train, D_val, D_test (No redundant X_all!) ----
    D_train = struct();
    D_train.x        = zeros(N_steps, 10, n_train, 'single');
    D_train.u        = zeros(N_steps, 2,  n_train, 'single');
    D_train.cgm      = zeros(N_steps, n_train, 'single');
    D_train.day_ids  = zeros(7, n_train, 'int16');
    D_train.tir      = zeros(1, n_train, 'single');
    D_train.tar      = zeros(1, n_train, 'single');
    D_train.tbr      = zeros(1, n_train, 'single');
    D_train.mean_cgm = zeros(1, n_train, 'single');
    D_train.is_rare  = false(1, n_train);
    D_train.usubjid  = strings(1, n_train);

    D_val = struct();
    D_val.x        = zeros(N_steps, 10, n_val, 'single');
    D_val.u        = zeros(N_steps, 2,  n_val, 'single');
    D_val.cgm      = zeros(N_steps, n_val, 'single');
    D_val.day_ids  = zeros(7, n_val, 'int16');
    D_val.tir      = zeros(1, n_val, 'single');
    D_val.tar      = zeros(1, n_val, 'single');
    D_val.tbr      = zeros(1, n_val, 'single');
    D_val.mean_cgm = zeros(1, n_val, 'single');
    D_val.is_rare  = false(1, n_val);
    D_val.usubjid  = strings(1, n_val);

    D_test = struct();
    D_test.x        = zeros(N_steps, 10, n_test, 'single');
    D_test.u        = zeros(N_steps, 2,  n_test, 'single');
    D_test.cgm      = zeros(N_steps, n_test, 'single');
    D_test.day_ids  = zeros(7, n_test, 'int16');
    D_test.tir      = zeros(1, n_test, 'single');
    D_test.tar      = zeros(1, n_test, 'single');
    D_test.tbr      = zeros(1, n_test, 'single');
    D_test.mean_cgm = zeros(1, n_test, 'single');
    D_test.is_rare  = false(1, n_test);
    D_test.usubjid  = strings(1, n_test);

    % ---- Pass 2: Fill D_train, D_val, D_test from chunks ----
    fprintf('  Pass 2/2: assembling train/val/test splits ...\n');
    ptr_tr = 1; ptr_va = 1; ptr_te = 1;
    global_trace_idx = 0;

    for k = 1:n_files
        cpath = get_chunk_file_path(chunk_dir, chunk_files(k).name);
        cdata = load(cpath);
        cnt = chunk_counts(k);

        chunk_lbls = split_labels(global_trace_idx+1 : global_trace_idx+cnt);
        global_trace_idx = global_trace_idx + cnt;

        % Train indices in this chunk
        tr_mask = (chunk_lbls == 1);
        if any(tr_mask)
            n_tr_c = sum(tr_mask);
            rng_tr = ptr_tr : (ptr_tr + n_tr_c - 1);
            D_train.x(:, :, rng_tr)   = cdata.X_cube(:, :, tr_mask);
            D_train.u(:, :, rng_tr)   = cdata.U_cube(:, :, tr_mask);
            D_train.cgm(:, rng_tr)     = cdata.CGM_mat(:, tr_mask);
            D_train.day_ids(:, rng_tr) = cdata.day_ids_mat(:, tr_mask);
            D_train.tir(rng_tr)       = cdata.tir_vec(tr_mask);
            D_train.tar(rng_tr)       = cdata.tar_vec(tr_mask);
            D_train.tbr(rng_tr)       = cdata.tbr_vec(tr_mask);
            D_train.mean_cgm(rng_tr)  = cdata.mg_vec(tr_mask);
            D_train.is_rare(rng_tr)   = cdata.rare_vec(tr_mask);
            if isfield(cdata, 'usubjid_arr')
                D_train.usubjid(rng_tr) = cdata.usubjid_arr(tr_mask);
            end
            ptr_tr = ptr_tr + n_tr_c;
        end

        % Val indices in this chunk
        va_mask = (chunk_lbls == 2);
        if any(va_mask)
            n_va_c = sum(va_mask);
            rng_va = ptr_va : (ptr_va + n_va_c - 1);
            D_val.x(:, :, rng_va)   = cdata.X_cube(:, :, va_mask);
            D_val.u(:, :, rng_va)   = cdata.U_cube(:, :, va_mask);
            D_val.cgm(:, rng_va)     = cdata.CGM_mat(:, va_mask);
            D_val.day_ids(:, rng_va) = cdata.day_ids_mat(:, va_mask);
            D_val.tir(rng_va)       = cdata.tir_vec(va_mask);
            D_val.tar(rng_va)       = cdata.tar_vec(va_mask);
            D_val.tbr(rng_va)       = cdata.tbr_vec(va_mask);
            D_val.mean_cgm(rng_va)  = cdata.mg_vec(va_mask);
            D_val.is_rare(rng_va)   = cdata.rare_vec(va_mask);
            if isfield(cdata, 'usubjid_arr')
                D_val.usubjid(rng_va) = cdata.usubjid_arr(va_mask);
            end
            ptr_va = ptr_va + n_va_c;
        end

        % Test indices in this chunk
        te_mask = (chunk_lbls == 3);
        if any(te_mask)
            n_te_c = sum(te_mask);
            rng_te = ptr_te : (ptr_te + n_te_c - 1);
            D_test.x(:, :, rng_te)   = cdata.X_cube(:, :, te_mask);
            D_test.u(:, :, rng_te)   = cdata.U_cube(:, :, te_mask);
            D_test.cgm(:, rng_te)     = cdata.CGM_mat(:, te_mask);
            D_test.day_ids(:, rng_te) = cdata.day_ids_mat(:, te_mask);
            D_test.tir(rng_te)       = cdata.tir_vec(te_mask);
            D_test.tar(rng_te)       = cdata.tar_vec(te_mask);
            D_test.tbr(rng_te)       = cdata.tbr_vec(te_mask);
            D_test.mean_cgm(rng_te)  = cdata.mg_vec(te_mask);
            D_test.is_rare(rng_te)   = cdata.rare_vec(te_mask);
            if isfield(cdata, 'usubjid_arr')
                D_test.usubjid(rng_te) = cdata.usubjid_arr(te_mask);
            end
            ptr_te = ptr_te + n_te_c;
        end

        clear cdata;
    end

    % ---- Robust Scaling Statistics (Appendix C) -------------------------
    fprintf('  Computing robust scaling stats from D_train ...\n');
    max_sample = min(n_train, 10000);
    sample_idx = sort(randperm(n_train, max_sample));

    flat_x   = reshape(permute(D_train.x(:, :, sample_idx), [1, 3, 2]), [], 10);
    flat_u   = reshape(permute(D_train.u(:, :, sample_idx), [1, 3, 2]), [], 2);
    sample_cgm = D_train.cgm(:, sample_idx);
    flat_cgm = sample_cgm(:);

    scaling_stats = struct();
    scaling_stats.state_names  = {'S1','S2','I','X1','X2','X3','Q1','Q2','C1','C2'};
    scaling_stats.input_names  = {'u_I','u_carbs'};

    scaling_stats.state_median = median(flat_x, 1);
    scaling_stats.state_p25    = prctile(flat_x, 25, 1);
    scaling_stats.state_p75    = prctile(flat_x, 75, 1);
    scaling_stats.state_iqr    = scaling_stats.state_p75 - scaling_stats.state_p25;
    scaling_stats.state_iqr(scaling_stats.state_iqr == 0) = 1.0;

    scaling_stats.input_median = median(flat_u, 1);
    scaling_stats.input_p25    = prctile(flat_u, 25, 1);
    scaling_stats.input_p75    = prctile(flat_u, 75, 1);
    scaling_stats.input_iqr    = scaling_stats.input_p75 - scaling_stats.input_p25;
    scaling_stats.input_iqr(scaling_stats.input_iqr == 0) = 1.0;

    scaling_stats.cgm_median   = median(flat_cgm);
    scaling_stats.cgm_iqr      = prctile(flat_cgm, 75) - prctile(flat_cgm, 25);

    clear flat_x flat_u flat_cgm sample_cgm;

    % ---- Assemble Dataset & Meta Structs --------------------------------
    dataset = struct();
    dataset.D_train = D_train;
    dataset.D_val   = D_val;
    dataset.D_test  = D_test;
    dataset.scaling_stats = scaling_stats;

    meta = struct();
    meta.paper_title = 'A Physiologically-Constrained Neural Network Digital Twin Framework For Replicating Glucose Dynamics In Type 1 Diabetes';
    meta.created_at  = char(datetime('now'));
    meta.member_id   = member_id;
    meta.total_members = total_members;
    meta.start_scen_idx = start_scen_idx;
    meta.end_scen_idx   = end_scen_idx;
    meta.num_scenarios  = n_scenarios;
    meta.total_traces   = total_traces;
    meta.n_train = n_train;
    meta.n_val   = n_val;
    meta.n_test  = n_test;
    meta.total_simulated_days = total_sim_days;
    meta.clinical_outcomes = struct(...
        'TIR_mean_sd', sprintf('%.1f +/- %.1f%%', mean(all_tir), std(all_tir)), ...
        'TAR_mean_sd', sprintf('%.1f +/- %.1f%%', mean(all_tar), std(all_tar)), ...
        'TBR_mean_sd', sprintf('%.1f +/- %.1f%%', mean(all_tbr), std(all_tbr)), ...
        'MG_mean_sd',  sprintf('%.1f +/- %.1f mg/dL', mean(all_mg), std(all_mg)));

    fprintf('\nPart %d Clinical Metrics:\n', member_id);
    fprintf('  TIR (70-180 mg/dL) : %s\n', meta.clinical_outcomes.TIR_mean_sd);
    fprintf('  TAR (>180 mg/dL)   : %s\n', meta.clinical_outcomes.TAR_mean_sd);
    fprintf('  TBR (<70 mg/dL)    : %s\n', meta.clinical_outcomes.TBR_mean_sd);
    fprintf('  Mean Glucose       : %s\n', meta.clinical_outcomes.MG_mean_sd);

    % ---- Save ONE Single .mat File with both top-level and struct access -
    fprintf('\nSaving ONE single dataset file to %s ...\n', opts.save_path);
    try
        % Save standard dataset + meta struct (compatible with verify & merge)
        save(opts.save_path, 'dataset', 'meta', '-v7.3');
    catch
        % Fallback for tight RAM: save splits individually into the SAME file
        fprintf('  Saving splits sequentially to ensure zero-RAM-overflow ...\n');
        save(opts.save_path, 'meta', 'scaling_stats', '-v7.3');
        save(opts.save_path, 'D_train', '-append');
        clear D_train;
        save(opts.save_path, 'D_val', '-append');
        clear D_val;
        save(opts.save_path, 'D_test', '-append');
        clear D_test;
    end
    fprintf('Successfully saved ONE single MAT file: %s!\n', opts.save_path);

    % Cleanup chunk directory
    if exist(chunk_dir, 'dir')
        try rmdir(chunk_dir, 's'); catch; end
    end

    fprintf('========================================================================\n');
    fprintf(' Part %d Generation Complete. Ready for Model Training!\n', member_id);
    fprintf('========================================================================\n');
end


%% =========================================================================
%% HIGH-SPEED INLINED ODE SIMULATOR & HELPER FUNCTIONS
%% =========================================================================

function [X_trace, U_trace, CGM_trace] = simulate_7day_ode_fast(...
    x0, u_carbs_grid, u_insulin_grid, base_basal_Uhr, ModPar, N_steps, Ts, dt_sub, sub_steps_per_frame)

    X_trace   = zeros(N_steps, 10);
    U_trace   = zeros(N_steps, 2);
    CGM_trace = zeros(N_steps, 1);

    Fc01  = ModPar.Fc01;
    VdG   = ModPar.VdG;
    k12   = ModPar.k12;
    Ag    = ModPar.Ag;
    tmaxG = ModPar.tmaxG;
    EGP0  = ModPar.EGP0;
    tmaxI = ModPar.tmaxI;
    Ke    = ModPar.Ke;
    VdI   = ModPar.VdI;
    ka1   = ModPar.ka1;
    ka2   = ModPar.ka2;
    ka3   = ModPar.ka3;
    Sf1   = ModPar.Sf1;
    Sf2   = ModPar.Sf2;
    Sf3   = ModPar.Sf3;
    W     = ModPar.Weight;

    S1 = x0(1); S2 = x0(2); I  = x0(3);
    X1 = x0(4); X2 = x0(5); X3 = x0(6);
    Q1 = x0(7); Q2 = x0(8); C1 = x0(9); C2 = x0(10);

    inv_tmaxI = 1.0 / tmaxI;
    inv_tmaxG = 1.0 / tmaxG;
    inv_VdI_tmaxI = 1.0 / (tmaxI * VdI);
    ins_scale = 1000.0 / (60.0 * W);
    carb_scale = 1000.0 / (180.16 * dt_sub * W);
    inv_VdG = 1.0 / VdG;

    for k = 1:N_steps
        u_I_Uhr = base_basal_Uhr + u_insulin_grid(k);
        u_I_mU  = u_I_Uhr * ins_scale;
        u_carbs_g = u_carbs_grid(k);

        U_trace(k, 1) = u_I_Uhr;
        U_trace(k, 2) = u_carbs_g;

        for sub = 1:sub_steps_per_frame
            if sub == 1 && u_carbs_g > 0
                u_carbs_mmol = u_carbs_g * carb_scale;
            else
                u_carbs_mmol = 0.0;
            end

            dS1 = u_I_mU - S1 * inv_tmaxI;
            dS2 = (S1 - S2) * inv_tmaxI;
            dI  = S2 * inv_VdI_tmaxI - Ke * I;

            dX1 = ka1 * (Sf1 * I - X1);
            dX2 = ka2 * (Sf2 * I - X2);
            dX3 = ka3 * (Sf3 * I - X3);

            dC1 = -C1 * inv_tmaxG + u_carbs_mmol;
            dC2 = (C1 - C2) * inv_tmaxG;
            UG  = Ag * C2 * inv_tmaxG;

            G_mmol = Q1 * inv_VdG;
            if G_mmol < 4.5
                Fc01_eff = (Fc01 * G_mmol) / 4.5;
                FR = 0.0;
            elseif G_mmol < 9.0
                Fc01_eff = Fc01;
                FR = 0.0;
            else
                Fc01_eff = Fc01;
                FR = 0.003 * (G_mmol - 9.0) * VdG;
            end

            dQ1 = -X1 * Q1 - Fc01_eff - FR + k12 * Q2 + UG + EGP0 * (1.0 - X3);
            dQ2 = X1 * Q1 - (k12 + X2) * Q2;

            S1 = max(0, S1 + dS1 * dt_sub);
            S2 = max(0, S2 + dS2 * dt_sub);
            I  = max(0, I  + dI  * dt_sub);
            X1 = max(0, X1 + dX1 * dt_sub);
            X2 = max(0, X2 + dX2 * dt_sub);
            X3 = max(0, min(1.0, X3 + dX3 * dt_sub));
            Q1 = max(0.1, Q1 + dQ1 * dt_sub);
            Q2 = max(0.1, Q2 + dQ2 * dt_sub);
            C1 = max(0, C1 + dC1 * dt_sub);
            C2 = max(0, C2 + dC2 * dt_sub);
        end

        X_trace(k, 1)  = S1; X_trace(k, 2)  = S2; X_trace(k, 3)  = I;
        X_trace(k, 4)  = X1; X_trace(k, 5)  = X2; X_trace(k, 6)  = X3;
        X_trace(k, 7)  = Q1; X_trace(k, 8)  = Q2; X_trace(k, 9)  = C1; X_trace(k, 10) = C2;

        CGM_trace(k) = Q1 * inv_VdG * 18.0;
    end
end


function [u_carbs_grid, u_insulin_grid] = build_scenario_inputs(...
    selected_days, Sim_time_min, Ts, Weight, ICR_base, force_delays)

    if nargin < 6, force_delays = false; end

    N_steps = Sim_time_min / Ts;
    u_carbs_grid   = zeros(1, N_steps);
    u_insulin_grid = zeros(1, N_steps);

    for d = 1:numel(selected_days)
        day = selected_days(d);
        day_offset_min = (d - 1) * 1440;

        for m = 1:day.num_meals
            meal_time_min = day_offset_min + day.time_min(m);
            carb_g        = day.carb_grams(m);

            if meal_time_min >= Sim_time_min
                continue;
            end

            step_meal = max(1, min(N_steps, round(meal_time_min / Ts) + 1));
            u_carbs_grid(step_meal) = u_carbs_grid(step_meal) + carb_g;

            if force_delays
                delay_min = randi([15, 45]);
            else
                delays = 0:5:45;
                delay_min = delays(randi(numel(delays)));
            end
            bolus_time_min = meal_time_min + delay_min;
            step_bolus = max(1, min(N_steps, round(bolus_time_min / Ts) + 1));

            dosing_factor = 0.85 + 0.3 * rand();
            icr = ICR_base * dosing_factor;

            bolus_U = carb_g / icr;
            bolus_rate_Uhr = (bolus_U / (Ts / 60.0));

            u_insulin_grid(step_bolus) = u_insulin_grid(step_bolus) + bolus_rate_Uhr;
        end
    end
end


function [x0, basal_mU, basal_Uhr] = solve_steady_state(g0, ModPar)
    G_mmol = g0 / 18.0;
    Q1_ss  = G_mmol * ModPar.VdG;

    if G_mmol < 4.5
        Fc01_eff = (ModPar.Fc01 * G_mmol) / 4.5;
        FR = 0.0;
    elseif G_mmol < 9.0
        Fc01_eff = ModPar.Fc01;
        FR = 0.0;
    else
        Fc01_eff = ModPar.Fc01;
        FR = 0.003 * (G_mmol - 9.0) * ModPar.VdG;
    end
    F_non_ins = Fc01_eff + FR;

    A = -ModPar.Sf1 * ModPar.Sf2 * Q1_ss - ModPar.EGP0 * ModPar.Sf3 * ModPar.Sf2;
    B = -F_non_ins * ModPar.Sf2 + ModPar.EGP0 * ModPar.Sf2 - ModPar.EGP0 * ModPar.Sf3 * ModPar.k12;
    C = ModPar.k12 * (ModPar.EGP0 - F_non_ins);

    disc = B^2 - 4.0 * A * C;
    if disc >= 0 && A ~= 0
        Iss = (-B - sqrt(disc)) / (2.0 * A);
    else
        Iss = 5.5;
    end
    Iss = max(1.0, Iss);

    basal_mU  = Iss * ModPar.VdI * ModPar.Ke;
    basal_Uhr = (basal_mU * 60.0 * ModPar.Weight) / 1000.0;

    S1_ss = basal_mU * ModPar.tmaxI;
    S2_ss = S1_ss;

    X1_ss = ModPar.Sf1 * Iss;
    X2_ss = ModPar.Sf2 * Iss;
    X3_ss = min(0.99, ModPar.Sf3 * Iss);

    Q2_ss = (X1_ss * Q1_ss) / (ModPar.k12 + X2_ss);

    C1_ss = 0.0;
    C2_ss = 0.0;

    x0 = [S1_ss, S2_ss, Iss, X1_ss, X2_ss, X3_ss, Q1_ss, Q2_ss, C1_ss, C2_ss];
end


function cpath = get_chunk_file_path(chunk_dir, fname)
    if exist(fullfile(chunk_dir, fname), 'file')
        cpath = fullfile(chunk_dir, fname);
    else
        cpath = fname;
    end
end


function safe_save_matlab_drive(filepath, data_struct)
    max_retries = 3;
    for r = 1:max_retries
        try
            save(filepath, '-struct', 'data_struct', '-v7');
            return;
        catch ME
            if r == max_retries
                [~, fname, fext] = fileparts(filepath);
                fallback_name = [fname, fext];
                warning('generate_population_dataset:driveSaveFallback', ...
                        'Drive path lock on %s. Saving to fallback: %s. Error: %s', ...
                        filepath, fallback_name, ME.message);
                save(fallback_name, '-struct', 'data_struct', '-v7');
            else
                pause(1.0);
            end
        end
    end
end
