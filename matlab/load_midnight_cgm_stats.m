function cgm_map = load_midnight_cgm_stats(csv_path)
% LOAD_MIDNIGHT_CGM_STATS  Read midnight_cgm_stats.csv into a containers.Map.
%
%   cgm_map = load_midnight_cgm_stats(csv_path)
%
%   Returns a containers.Map keyed by normalised USUBJID (char),
%   where each value is a struct with fields:
%       mean  - midnight CGM mean (mg/dL)
%       std   - midnight CGM standard deviation (mg/dL)
%
%   USUBJID normalisation matches normalise_id() in the Python build
%   scripts (build_day_scenario_library.py, build_midnight_cgm_stats.py).
%
%   The generator's existing fallback (cgm_mean=156, cgm_std=45) handles
%   participants missing from this map; the loader does not replicate that.

    T = readtable(csv_path, 'TextType', 'string', 'Delimiter', ',');

    required_cols = ["usubjid", "midnight_mean", "midnight_std"];
    for i = 1:numel(required_cols)
        if ~ismember(required_cols(i), string(T.Properties.VariableNames))
            error('load_midnight_cgm_stats:missingColumn', ...
                  'Required column "%s" not found in %s.', required_cols(i), csv_path);
        end
    end

    cgm_map = containers.Map('KeyType', 'char', 'ValueType', 'any');

    for i = 1:height(T)
        key = char(normalise_id_matlab(T.usubjid(i)));
        if isempty(key)
            continue;
        end
        entry.mean = T.midnight_mean(i);
        entry.std  = T.midnight_std(i);
        cgm_map(key) = entry;
    end

    fprintf('Loaded midnight CGM stats for %d participants from %s.\n', ...
            cgm_map.Count, csv_path);
end


function s = normalise_id_matlab(raw)
% NORMALISE_ID_MATLAB  Match normalise_id() from the Python build scripts.
%   Strips whitespace. If the string matches "^(-?\d+)\.0*$", keeps only
%   the integer part (e.g. "12.0" -> "12"). Zero-padded ids like "0012"
%   are preserved because "0012" and "12" can be different subjects.
    s = strtrim(string(raw));
    if s == ""
        return;
    end
    tok = regexp(s, '^(-?\d+)\.0*$', 'tokens');
    if ~isempty(tok)
        s = string(tok{1}{1});
    end
end
