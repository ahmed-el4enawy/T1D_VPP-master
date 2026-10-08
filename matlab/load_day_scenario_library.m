function meal_lib = load_day_scenario_library(csv_path)
% LOAD_DAY_SCENARIO_LIBRARY  Read day_scenario_library.csv into a struct array.
%
%   meal_lib = load_day_scenario_library(csv_path)
%
%   Returns a struct array with fields:
%       day_id      - integer, must equal the 1-based row index
%       usubjid     - string (normalised: "12.0" -> "12")
%       num_meals   - integer
%       carb_grams  - numeric row vector of per-meal carb amounts (g)
%       time_min    - numeric row vector of per-meal times (minutes from midnight)
%
%   Asserts day_id == row index. Errors on any non-positive carb value.
%   Matches normalise_id() in build_day_scenario_library.py and
%   build_midnight_cgm_stats.py.

    T = readtable(csv_path, 'TextType', 'string', 'Delimiter', ',');

    required_cols = ["day_id", "usubjid", "num_meals", "carb_grams", "time_min"];
    for i = 1:numel(required_cols)
        if ~ismember(required_cols(i), string(T.Properties.VariableNames))
            error('load_day_scenario_library:missingColumn', ...
                  'Required column "%s" not found in %s.', required_cols(i), csv_path);
        end
    end

    n = height(T);
    meal_lib(n) = struct('day_id', 0, 'usubjid', "", 'num_meals', 0, ...
                         'carb_grams', [], 'time_min', []);

    for i = 1:n
        did = T.day_id(i);
        if did ~= i
            error('load_day_scenario_library:dayIdMismatch', ...
                  'Row %d has day_id=%d. day_id must equal the 1-based row index.', i, did);
        end

        meal_lib(i).day_id    = did;
        meal_lib(i).usubjid   = normalise_id_matlab(T.usubjid(i));
        meal_lib(i).num_meals = T.num_meals(i);

        % Parse semicolon-delimited carb_grams
        cg_str = string(T.carb_grams(i));
        cg_parts = split(cg_str, ";");
        cg_vals = str2double(cg_parts);
        if any(isnan(cg_vals))
            error('load_day_scenario_library:parseError', ...
                  'Row %d: could not parse carb_grams "%s".', i, cg_str);
        end
        if any(cg_vals <= 0)
            error('load_day_scenario_library:nonPositiveCarb', ...
                  'Row %d has a non-positive carb value in carb_grams.', i);
        end
        if numel(cg_vals) ~= meal_lib(i).num_meals
            error('load_day_scenario_library:lengthMismatch', ...
                  'Row %d: numel(carb_grams) does not match num_meals.', i);
        end
        if any(~isfinite(cg_vals))
            error('load_day_scenario_library:nonFiniteCarb', ...
                  'Row %d: carb_grams contains non-finite values.', i);
        end
        meal_lib(i).carb_grams = cg_vals(:)';

        % Parse semicolon-delimited time_min
        tm_str = string(T.time_min(i));
        tm_parts = split(tm_str, ";");
        tm_vals = str2double(tm_parts);
        if any(isnan(tm_vals))
            error('load_day_scenario_library:parseError', ...
                  'Row %d: could not parse time_min "%s".', i, tm_str);
        end
        if numel(tm_vals) ~= meal_lib(i).num_meals
            error('load_day_scenario_library:lengthMismatch', ...
                  'Row %d: numel(time_min) does not match num_meals.', i);
        end
        if any(tm_vals < 0 | tm_vals >= 1440)
            error('load_day_scenario_library:invalidTime', ...
                  'Row %d: time_min contains values out of bounds [0, 1440).', i);
        end
        if any(~isfinite(tm_vals))
            error('load_day_scenario_library:nonFiniteTime', ...
                  'Row %d: time_min contains non-finite values.', i);
        end
        meal_lib(i).time_min = tm_vals(:)';
    end

    fprintf('Loaded %d daily meal scenarios from %s.\n', n, csv_path);
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
