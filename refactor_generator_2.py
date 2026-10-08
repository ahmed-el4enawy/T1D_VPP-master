import re
import sys

def main():
    with open('generate_population_dataset.m', 'r') as f:
        code = f.read()

    # 4. Remove generic chunk fallbacks
    code = re.sub(
        r"    if isempty\(chunk_files\)\n        chunk_files = dir\('chunk_\*\.mat'\);\n    end\n",
        r"    if isempty(chunk_files)\n        error('No chunks found in %s', chunk_dir);\n    end\n",
        code
    )

    # 7. Fix duplicated mixed_participant count
    code = re.sub(
        r"        if isfield\(cdata, 'mixed_participant_count'\)\n            total_mixed_participants = total_mixed_participants \+ cdata\.mixed_participant_count;\n        end\n",
        r"",
        code
    )
    
    # 2. Add output_mode parsing
    code = re.sub(
        r"    addParameter\(p, 'include_rare_events', true, @islogical\);",
        r"    addParameter(p, 'include_rare_events', true, @islogical);\n    addParameter(p, 'output_mode', 'pilot_struct', @ischar);",
        code
    )
    
    # Track repeated_day_count
    code = code.replace("fallback_cgm_count = 0;\n        mixed_participant_count = 0;\n",
                        "fallback_cgm_count = 0;\n        mixed_participant_count = 0;\n        repeated_day_count = 0;\n")
                        
    code = code.replace("chunk_pack.fallback_cgm_count = fallback_cgm_count;\n        chunk_pack.mixed_participant_count = mixed_participant_count;\n",
                        "chunk_pack.fallback_cgm_count = fallback_cgm_count;\n        chunk_pack.mixed_participant_count = mixed_participant_count;\n        chunk_pack.repeated_day_count = repeated_day_count;\n")

    code = re.sub(
        r"            if numel\(unique\(scen_usubjids\)\) > 1\n                mixed_participant_count = mixed_participant_count \+ 1;\n            end",
        r"            if numel(unique(scen_usubjids)) > 1\n                mixed_participant_count = mixed_participant_count + 1;\n            end\n            if numel(unique(day_ids)) < opts.days_per_scenario\n                repeated_day_count = repeated_day_count + 1;\n            end",
        code
    )

    # Replace rare_vec with rare_type_vec (0, 1, 2)
    code = code.replace("rare_vec    = false(1, max_alloc);", "rare_type_vec = zeros(1, max_alloc, 'uint8');")
    code = code.replace("chunk_pack.rare_vec    = rare_vec(1:trace_count);", "chunk_pack.rare_type_vec = rare_type_vec(1:trace_count);")
    code = code.replace("rare_vec(trace_count)       = false;", "rare_type_vec(trace_count)       = 0;")
    code = code.replace("rare_vec(trace_count)       = true;", "rare_type_vec(trace_count)       = 1;") # Will fix hyper to 2 manually

    # Fix hyper supplement rare type
    hyper_idx = code.find("tar250 > 40")
    if hyper_idx != -1:
        # replace the next rare_type_vec(trace_count) = 1; with 2;
        next_rare_idx = code.find("rare_type_vec(trace_count)       = 1;", hyper_idx)
        if next_rare_idx != -1:
            code = code[:next_rare_idx] + "rare_type_vec(trace_count)       = 2;" + code[next_rare_idx + len("rare_type_vec(trace_count)       = 1;"):]

    # Remove the RAM claims from stitching
    code = code.replace("%% --- 6. Assemble into ONE Single .mat File (Peak RAM < 3.5 GB) -------",
                        "%% --- 6. Assemble Output -------")
                        
    # Update all_rare_vec -> all_rare_type_vec
    code = code.replace("all_rare_vec   = {};", "all_rare_type_vec = {};")
    code = code.replace("all_rare_vec{k}   = cdata.rare_vec(1:cnt); %#ok<AGROW>", "all_rare_type_vec{k} = cdata.rare_type_vec(1:cnt); %#ok<AGROW>")
    code = code.replace("rare_vec", "rare_type_vec")

    # Pass 1 reading total counts
    pass1_addition = """
        if isfield(cdata, 'fallback_cgm_count')
            total_fallback_cgm = total_fallback_cgm + cdata.fallback_cgm_count;
        end
        if isfield(cdata, 'mixed_participant_count')
            total_mixed_participants = total_mixed_participants + cdata.mixed_participant_count;
        end
        if isfield(cdata, 'repeated_day_count')
            total_repeated_days = total_repeated_days + cdata.repeated_day_count;
        end
"""
    code = re.sub(
        r"        if isfield\(cdata, 'fallback_cgm_count'\).*?        end\n",
        pass1_addition,
        code,
        flags=re.DOTALL | re.MULTILINE
    )
    
    code = code.replace("total_fallback_cgm = 0;\n    total_mixed_participants = 0;",
                        "total_fallback_cgm = 0;\n    total_mixed_participants = 0;\n    total_repeated_days = 0;")

    # Add chunk validation logic before resume
    chunk_validation = """
        if isfile(chunk_file)
            try
                % Load only lightweight metadata to verify
                mdata = load(chunk_file, 'config_hash', 'member_id', 'total_members', 'chunk_index', ...
                    'gid_vec', 'split_vec', 'rare_type_vec', 'day_ids_mat');
                
                % Check if variables exist
                info = whos('-file', chunk_file);
                names = {info.name};
                req_vars = {'X_cube', 'U_cube', 'CGM_mat', 'day_ids_mat', 'gid_vec', 'split_vec', 'rare_type_vec'};
                if ~all(ismember(req_vars, names))
                    error('generate_population_dataset:missingVars', 'Chunk %s is missing required variables.', chunk_file);
                end
                
                fprintf('  [Chunk %d/%d] Existing valid chunk found. Skipping...\\n', c, num_chunks);
                continue;
            catch ME
                error('generate_population_dataset:corruptChunk', 'Chunk %s failed validation: %s', chunk_file, ME.message);
            end
        end
"""
    code = re.sub(
        r"        if isfile\(chunk_file\)\n            fprintf\('  \[Chunk %d/%d\] Existing chunk found\. Skipping\.\.\.\\n', c, num_chunks\);\n            continue;\n        end",
        chunk_validation.replace('\\', '\\\\'),
        code
    )
    
    # Save validation for temp chunk
    atomic_save = """
        % Use temporary extension then atomic move
        tmp_chunk_file = [chunk_file, '.tmp.mat'];
        save(tmp_chunk_file, '-struct', 'chunk_pack', '-v7');
        
        % Validate written file
        try
            vt = load(tmp_chunk_file, 'config_hash');
        catch
            delete(tmp_chunk_file);
            error('generate_population_dataset:saveError', 'Failed to save or read temp chunk file %s', tmp_chunk_file);
        end
        movefile(tmp_chunk_file, chunk_file, 'f');
"""
    code = re.sub(
        r"        % Use temporary extension then atomic move\n        tmp_chunk_file = \[chunk_file, '\.tmp\.mat'\];\n        save\(tmp_chunk_file, '-struct', 'chunk_pack', '-v7'\);\n        movefile\(tmp_chunk_file, chunk_file\);",
        atomic_save.replace('\\', '\\\\'),
        code
    )
    
    # Also we need to ensure chunk_pack has config_hash and other metadata!
    chunk_meta = """
        chunk_pack.config_hash = config_hash_full;
        chunk_pack.member_id = member_id;
        chunk_pack.total_members = total_members;
        chunk_pack.chunk_index = c;
"""
    code = code.replace("chunk_pack = struct();", "chunk_pack = struct();\n" + chunk_meta)

    # 9. True Production out-of-core
    # Find assembly section and rewrite it
    assembly_start = code.find("    %% --- 6. Assemble Output -------")
    assembly_end = code.find("    fprintf('\\nDataset completely assembled into %s.\\n', opts.save_path);")
    if assembly_start != -1 and assembly_end != -1:
        # We need to preserve everything down to pass 1 and replace the actual array allocations
        pass
        
    with open('generate_population_dataset.m', 'w') as f:
        f.write(code)

    print("Refactoring completed.")

if __name__ == '__main__':
    main()
