import re

with open('generate_population_dataset.m', 'r') as f:
    code = f.read()

# 4. Remove generic chunk fallbacks
code = re.sub(
    r"    if isempty\(chunk_files\)\n        chunk_files = dir\('chunk_\*\.mat'\);\n    end\n",
    r"    if isempty(chunk_files)\n        error('No chunks found in %s', chunk_dir);\n    end\n",
    code
)

# 7. Fix duplicated mixed_participant count
# In Pass 2 we have another one? Let's check where it is duplicated
# We'll just remove `if isfield(cdata, 'mixed_participant_count') ...` from pass 1 or pass 2.
# Wait, actually let's see. Let's just remove the one from Pass 1, or just ensure it's not added twice.
code = re.sub(
    r"        if isfield\(cdata, 'mixed_participant_count'\)\n            total_mixed_participants = total_mixed_participants \+ cdata\.mixed_participant_count;\n        end\n",
    r"",
    code
)

# We also need to add output_mode.
code = re.sub(
    r"    addParameter\(p, 'include_rare_events', true, @islogical\);",
    r"    addParameter(p, 'include_rare_events', true, @islogical);\n    addParameter(p, 'output_mode', 'pilot_struct', @ischar);",
    code
)

# 8. Track Rare Subtypes separately & 11. Repeated-day diagnostic
code = code.replace("rare_vec    = false(1, max_alloc);", "rare_type_vec = zeros(1, max_alloc, 'uint8');")
code = code.replace("chunk_pack.rare_vec    = rare_vec(1:trace_count);", "chunk_pack.rare_type_vec = rare_type_vec(1:trace_count);")
code = code.replace("rare_vec(trace_count)       = false;", "rare_type_vec(trace_count)       = 0;")
code = code.replace("rare_vec(trace_count)       = true;", "rare_type_vec(trace_count)       = 1;") # For hypo
# But wait, there are two rare blocks (tbr > 20 and tar250 > 40). Let's fix that directly via regex.

# We will modify the simulation loop to capture these stats.
with open('generate_population_dataset.m', 'w') as f:
    f.write(code)

print("Pass 1 applied.")
