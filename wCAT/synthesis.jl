

function json_to_plain(x)
    if x isa JSON3.Object
        return Dict(String(k) => json_to_plain(v) for (k, v) in pairs(x))
    elseif x isa JSON3.Array
        return [json_to_plain(v) for v in x]
    else
        return x
    end
end

function load_automation_json(path::String)
    if isempty(path) || !isfile(path)
        return Dict{String,Any}(
            "version" => 4,
            "selection" => Dict{String,Any}(),
            "feature_weights" => Dict{String,Any}())
    end

    raw = JSON3.read(read(path, String))
    return json_to_plain(raw)
end


function has_column(df, name::String)
    return name in names(df)
end

function selected_feature_columns(df, feature_groups)
    columns = String[]
    owners = String[]

    for group in feature_groups
        label = String(group["label"])
        cols = feature_group_columns(df, group)

        if isempty(cols)
            @warn "Feature group '$label' requested in config but no matching columns were found in corpus CSV. It will be ignored during matching."
            continue
        end

        for col in cols
            push!(columns, col)
            push!(owners, label)
        end
    end

    isempty(columns) && error("No usable feature columns found in corpus CSV. Check config and regenerate the corpus index.")

    return columns, owners
end

function safe_float(v; fallback=NaN)
    if ismissing(v) || isnothing(v)
        return fallback
    end

    try
        x = Float64(v)
        return isfinite(x) ? x : fallback
    catch
        return fallback
    end
end

function row_to_vector(row, columns)
    out = zeros(Float64, length(columns))

    for (i, col) in enumerate(columns)
        key = Symbol(col)
        v = haskey(row, key) ? row[key] : missing
        out[i] = safe_float(v; fallback=NaN)
    end

    return out
end

function dataframe_to_matrix(df, columns)
    mat = zeros(Float64, nrow(df), length(columns))

    for (j, col) in enumerate(columns)
        values = df[!, Symbol(col)]

        for i in 1:nrow(df)
            mat[i, j] = safe_float(values[i]; fallback=NaN)
        end
    end

    return mat
end

function percentile_value(values, q)
    clean = sort([Float64(v) for v in values if isfinite(Float64(v))])
    isempty(clean) && return 0.0

    q = clamp(Float64(q), 0.0, 1.0)
    return quantile(clean, q)
end

function compute_normalization_stats(corpus_matrix, config)
    method = String(cfg(config, ["synthesis", "normalization", "method"], "robust_range"))
    q_low = Float64(cfg(config, ["synthesis", "normalization", "q_low"], 0.01))
    q_high = Float64(cfg(config, ["synthesis", "normalization", "q_high"], 0.99))
    eps = Float64(cfg(config, ["synthesis", "normalization", "eps"], 1e-12))

    n_cols = size(corpus_matrix, 2)
    lo = zeros(Float64, n_cols)
    hi = ones(Float64, n_cols)

    for j in 1:n_cols
        col = corpus_matrix[:, j]
        clean = [v for v in col if isfinite(v)]

        if isempty(clean)
            lo[j] = 0.0
            hi[j] = 1.0
            continue
        end

        if method == "minmax"
            lo[j] = minimum(clean)
            hi[j] = maximum(clean)
        elseif method == "robust_range"
            lo[j] = percentile_value(clean, q_low)
            hi[j] = percentile_value(clean, q_high)
        else
            error("Unsupported normalization method: $method")
        end

        if hi[j] - lo[j] <= eps
            hi[j] = lo[j] + 1.0
        end
    end

    return Dict{String,Any}(
        "lo" => lo,
        "hi" => hi,
        "eps" => eps,
        "clamp" => cfg(config, ["synthesis", "normalization", "clamp"], true))
end

function normalize_vector(raw, stats)
    lo = stats["lo"]
    hi = stats["hi"]
    eps = stats["eps"]
    clamp_values = stats["clamp"]

    out = zeros(Float64, length(raw))

    @inbounds for j in eachindex(raw)
        v = raw[j]

        if !isfinite(v)
            z = 0.0
        else
            z = (v - lo[j]) / (hi[j] - lo[j] + eps)
        end

        out[j] = clamp_values ? clamp(z, 0.0, 1.0) : z
    end

    return out
end

function normalize_matrix(raw_matrix, stats)
    out = similar(raw_matrix)

    for i in 1:size(raw_matrix, 1)
        out[i, :] .= normalize_vector(vec(raw_matrix[i, :]), stats)
    end

    return out
end

function interpolate_points(points, t, default_value)
    if isnothing(points) || !(points isa AbstractVector) || isempty(points)
        return Float64(default_value)
    end

    parsed = Tuple{Float64,Float64}[]

    for p in points
        if p isa AbstractVector && length(p) >= 2
            push!(parsed, (Float64(p[1]), Float64(p[2])))
        elseif p isa AbstractDict
            push!(parsed, (Float64(get_key_any(p, "t", 0.0)), Float64(get_key_any(p, "v", default_value))))
        end
    end

    isempty(parsed) && return Float64(default_value)
    sort!(parsed, by=p -> p[1])

    t = clamp(Float64(t), 0.0, 1.0)

    if t <= parsed[1][1]
        return parsed[1][2]
    end

    if t >= parsed[end][1]
        return parsed[end][2]
    end

    for i in 1:(length(parsed) - 1)
        t0, v0 = parsed[i]
        t1, v1 = parsed[i + 1]

        if t0 <= t <= t1
            denom = t1 - t0
            abs(denom) <= 1e-12 && return v1

            a = (t - t0) / denom
            return v0 + a * (v1 - v0)
        end
    end

    return Float64(default_value)
end

function automation_curve_value(curve, t, default_value; randomize_range=true)
    if isnothing(curve) || !(curve isa AbstractDict)
        return Float64(default_value)
    end

    if get_key_any(curve, "active", true) == false
        return Float64(default_value)
    end

    mode = String(get_key_any(curve, "mode", "single"))

    if mode == "range"
        lo = interpolate_points(get_key_any(curve, "lower", Any[]), t, default_value)
        hi = interpolate_points(get_key_any(curve, "upper", Any[]), t, default_value)

        lo, hi = min(lo, hi), max(lo, hi)
        return randomize_range ? lo + rand() * (hi - lo) : 0.5 * (lo + hi)
    end

    return interpolate_points(get_key_any(curve, "points", Any[]), t, default_value)
end

function selection_value(automation, key::String, t, default_value)
    selection = get_key_any(automation, "selection", Dict{String,Any}())
    curve = selection isa AbstractDict ? get_key_any(selection, key, nothing) : nothing
    return automation_curve_value(curve, t, default_value; randomize_range=false)
end

function feature_weight_value(automation, feature::String, t)
    feature_weights = get_key_any(automation, "feature_weights", Dict{String,Any}())
    curve = feature_weights isa AbstractDict ? get_key_any(feature_weights, feature, nothing) : nothing
    return max(0.0, automation_curve_value(curve, t, 1.0; randomize_range=true))
end

function build_feature_to_indices(owners)
    d = Dict{String,Vector{Int}}()

    for (i, feature) in enumerate(owners)
        if !haskey(d, feature)
            d[feature] = Int[]
        end

        push!(d[feature], i)
    end

    return d
end

function column_weights(automation, owners, feature_to_indices, t; eps=1e-12)
    weights = zeros(Float64, length(owners))

    for (feature, indices) in feature_to_indices
        w_feature = feature_weight_value(automation, feature, t)
        per_column = isempty(indices) ? 0.0 : w_feature / length(indices)

        for idx in indices
            weights[idx] = per_column
        end
    end

    if sum(weights) <= eps
        weights .= 1.0 / length(weights)
    end

    return weights
end

function weighted_euclidean_distance(x, y, w; eps=1e-12)
    denom = sum(w) + eps
    return sqrt(sum(w .* ((x .- y) .^ 2)) / denom)
end

function weighted_manhattan_distance(x, y, w; eps=1e-12)
    denom = sum(w) + eps
    return sum(w .* abs.(x .- y)) / denom
end

function weighted_cosine_distance(x, y, w; eps=1e-12)
    dot_xy = sum(w .* x .* y)
    nx = sqrt(sum(w .* x .* x))
    ny = sqrt(sum(w .* y .* y))

    if nx <= eps || ny <= eps
        return 1.0
    end

    return 1.0 - dot_xy / (nx * ny + eps)
end

function weighted_canberra_distance(x, y, w; eps=1e-12)
    denom = sum(w) + eps
    terms = abs.(x .- y) ./ (abs.(x) .+ abs.(y) .+ eps)
    return sum(w .* terms) / denom
end

function distance_vector(target_norm, corpus_norm, weights, metric::String)
    dists = zeros(Float64, size(corpus_norm, 1))

    for i in 1:size(corpus_norm, 1)
        y = vec(corpus_norm[i, :])

        if metric == "weighted_euclidean"
            dists[i] = weighted_euclidean_distance(target_norm, y, weights)
        elseif metric == "weighted_manhattan"
            dists[i] = weighted_manhattan_distance(target_norm, y, weights)
        elseif metric == "weighted_cosine"
            dists[i] = weighted_cosine_distance(target_norm, y, weights)
        elseif metric == "weighted_canberra"
            dists[i] = weighted_canberra_distance(target_norm, y, weights)
        else
            error("Unsupported distance metric: $metric")
        end
    end

    return dists
end

function safe_df_value(df, i, col::String, default)
    sym = Symbol(col)

    if !(col in names(df))
        return default
    end

    v = df[i, sym]
    return ismissing(v) ? default : v
end

function choose_corpus_index(target_norm, corpus_norm, weights, metric, df, recent_indices, recent_files, config, k)
    dists = distance_vector(target_norm, corpus_norm, weights, metric)

    recent_chunk_penalty = Float64(cfg(config, ["synthesis", "matching", "recent_chunk_penalty"], 0.20))
    recent_file_penalty = Float64(cfg(config, ["synthesis", "matching", "recent_file_penalty"], 0.05))
    temperature = Float64(cfg(config, ["synthesis", "matching", "temperature"], 0.15))

    recent_index_set = Set(recent_indices)
    recent_file_set = Set(recent_files)

    for i in 1:length(dists)
        if i in recent_index_set
            dists[i] += recent_chunk_penalty
        end

        file_path = String(safe_df_value(df, i, "file_path", ""))

        if file_path in recent_file_set
            dists[i] += recent_file_penalty
        end
    end

    k = clamp(Int(k), 1, length(dists))
    candidates = partialsortperm(dists, 1:k)

    if temperature <= 0.0 || length(candidates) == 1
        return candidates[1]
    end

    local_d = dists[candidates]
    base = minimum(local_d)
    probs = exp.(-((local_d .- base) ./ max(temperature, 1e-12)))
    total = sum(probs)

    if total <= 1e-12
        return candidates[1]
    end

    r = rand() * total
    acc = 0.0

    for i in eachindex(candidates)
        acc += probs[i]

        if r <= acc
            return candidates[i]
        end
    end

    return candidates[end]
end



function target_candidate_chunks(x, config)
    frame_size = Int(cfg(config, ["synthesis", "target", "frame_size"], cfg(config, ["chunking", "fixed", "sizes"], [2048])[1]))
    hop_size = Int(cfg(config, ["synthesis", "target", "hop_size"], cfg(config, ["chunking", "fixed", "hops"], [1024])[1]))
    jitter = Int(cfg(config, ["synthesis", "target", "analysis_jitter_samples"], 0))

    chunks = Vector{Tuple{Int,Int}}()
    start_sample = 1

    while start_sample + frame_size - 1 <= length(x)
        s = start_sample

        if jitter > 0
            s += rand(-jitter:jitter)
            s = clamp(s, 1, max(1, length(x) - frame_size + 1))
        end

        e = s + frame_size - 1
        push!(chunks, (s, e))

        start_sample += hop_size
    end

    return chunks
end

function target_events(x, sr, feature_groups, config, columns, stats)
    chunks = target_candidate_chunks(x, config)
    mode = String(cfg(config, ["synthesis", "target", "mode"], "adaptive"))
    energy_gate_db = Float64(cfg(config, ["synthesis", "target", "energy_gate_db"], -70.0))
    min_novelty = Float64(cfg(config, ["synthesis", "target", "min_novelty"], 0.08))
    steady_keep_probability = Float64(cfg(config, ["synthesis", "target", "steady_keep_probability"], 0.20))
    force_every_samples = Int(cfg(config, ["synthesis", "target", "force_every_samples"], 4096))

    target_layers = analysis_layer_signals(x, sr, config)
    events = Vector{Dict{String,Any}}()

    previous_layer_chunks = Dict{String,Any}()
    previous_accepted_norm = nothing
    last_accepted_start = -typemax(Int)

    for (start_sample, end_sample) in chunks
        full_chunk = x[start_sample:end_sample]
        layer_chunks = layer_chunks_from_bounds(target_layers, start_sample, end_sample)
        row = Dict{Symbol,Any}()

        extract_layer_features!(row, layer_chunks, previous_layer_chunks, sr, config)
        previous_layer_chunks = layer_chunks

        energy = if layer_analysis_enabled(config) && haskey(row, :full_loudness_db)
            Float64(row[:full_loudness_db])
        elseif haskey(row, :loudness_db)
            Float64(row[:loudness_db])
        else
            loudness_db(full_chunk)
        end

        if energy < energy_gate_db
            continue
        end

        raw = row_to_vector(row, columns)
        normv = normalize_vector(raw, stats)

        novelty = isnothing(previous_accepted_norm) ? Inf : sqrt(mean((normv .- previous_accepted_norm) .^ 2))
        forced = force_every_samples > 0 && start_sample - last_accepted_start >= force_every_samples

        if mode == "adaptive" && !forced && novelty < min_novelty
            if rand() > steady_keep_probability
                continue
            end
        end

        push!(events, Dict{String,Any}(
            "start_sample" => start_sample,
            "end_sample" => end_sample,
            "norm" => normv,
            "energy_db" => energy,
            "novelty" => novelty))

        previous_accepted_norm = normv
        last_accepted_start = start_sample
    end

    return events
end


function load_layers_cached!(cache::Dict{String,Dict{String,Any}}, path::String, config; normalize=true)
    if haskey(cache, path)
        return cache[path]
    end

    x, sr = read_audio_file(path; normalize=normalize)
    layers = get_render_source_mode(config) == "full" ? Dict("full" => x) : analysis_layer_signals(x, sr, config)
    item = Dict{String,Any}("layers" => layers, "sr" => sr)
    cache[path] = item
    return item
end

function segment_bounds_with_ratio(original_start, original_end, total_len, ratio, spray_samples, config)
    original_start = clamp(Int(original_start), 1, total_len)
    original_end = clamp(Int(original_end), original_start, total_len)

    original_len = max(1, original_end - original_start + 1)

    min_ratio = Float64(cfg(config, ["synthesis", "grain_length", "min_ratio"], 0.10))
    max_ratio = Float64(cfg(config, ["synthesis", "grain_length", "max_ratio"], 2.00))
    ratio = clamp(Float64(ratio), min_ratio, max_ratio)

    desired_len = max(1, round(Int, original_len * ratio))
    desired_len = min(desired_len, total_len)

    center = 0.5 * (original_start + original_end)

    spray = max(0, Int(round(spray_samples)))
    if spray > 0
        mode = String(cfg(config, ["synthesis", "spray", "mode"], "symmetric"))

        if mode == "forward"
            center += rand(0:spray)
        else
            center += rand(-spray:spray)
        end
    end

    s = round(Int, center - 0.5 * (desired_len - 1))
    e = s + desired_len - 1

    if s < 1
        e += 1 - s
        s = 1
    end

    if e > total_len
        s -= e - total_len
        e = total_len
    end

    s = clamp(s, 1, total_len)
    e = clamp(e, s, total_len)

    return s, e
end

function crop_or_pad_chunk_with_ratio(x, ratio, config)
    n = length(x)
    n == 0 && return x

    min_ratio = Float64(cfg(config, ["synthesis", "grain_length", "min_ratio"], 0.10))
    max_ratio = Float64(cfg(config, ["synthesis", "grain_length", "max_ratio"], 2.00))
    ratio = clamp(Float64(ratio), min_ratio, max_ratio)

    desired_len = max(1, round(Int, n * ratio))

    if desired_len == n
        return x
    elseif desired_len < n
        center = (n + 1) / 2
        s = round(Int, center - 0.5 * (desired_len - 1))
        s = clamp(s, 1, n - desired_len + 1)
        return x[s:s + desired_len - 1]
    else
        out = zeros(Float64, desired_len)
        offset = max(0, (desired_len - n) ÷ 2)
        out[offset + 1:offset + n] .= x
        out[1:offset] .= x[1]
        tail_start = offset + n + 1
        if tail_start <= desired_len
            out[tail_start:end] .= x[end]
        end
        return out
    end
end

function dominant_layer_from_weights(owners, weights)
    scores = Dict("full" => 0.0, "harmonic" => 0.0, "percussive" => 0.0, "residual" => 0.0)

    for (owner, w) in zip(owners, weights)
        layer = group_label_to_layer(String(owner))
        scores[layer] = get(scores, layer, 0.0) + Float64(w)
    end

    best_layer = "full"
    best_value = -Inf

    for (layer, value) in scores
        if value > best_value
            best_layer = layer
            best_value = value
        end
    end

    return best_layer
end

function corpus_segment_from_layer_signal(signal, original_start, original_end, config, spray_samples, grain_length_ratio)
    s, e = segment_bounds_with_ratio(
        original_start,
        original_end,
        length(signal),
        grain_length_ratio,
        spray_samples,
        config)

    return signal[s:e]
end

function corpus_segment(df, idx, target_sr, cache, config, spray_samples, grain_length_ratio; render_layer_override="full")
    normalize = cfg(config, ["audio", "normalize"], true)
    file_path = String(safe_df_value(df, idx, "file_path", ""))
    render_mode = get_render_source_mode(config)

    if render_mode == "matched_layer"
        render_mode = render_layer_override
    end

    if !isempty(file_path) && isfile(file_path)
        item = load_layers_cached!(cache, file_path, config; normalize=normalize)
        layers = item["layers"]
        sr = Int(item["sr"])
        original_start = Int(safe_df_value(df, idx, "start_sample", 1))
        original_end = Int(safe_df_value(df, idx, "end_sample", original_start))

        if render_mode == "layer_mix"
            mix = render_layer_mix_config(config)
            segment = nothing

            for (layer, gain) in mix
                abs(gain) <= 1e-12 && continue
                if !haskey(layers, layer)
                    continue
                end

                part = corpus_segment_from_layer_signal(layers[layer], original_start, original_end, config, spray_samples, grain_length_ratio)

                if isnothing(segment)
                    segment = zeros(Float64, length(part))
                elseif length(part) != length(segment)
                    n = min(length(part), length(segment))
                    segment = segment[1:n]
                    part = part[1:n]
                end

                segment .+= gain .* part
            end

            isnothing(segment) && (segment = corpus_segment_from_layer_signal(layers["full"], original_start, original_end, config, spray_samples, grain_length_ratio))
            return linear_resample(segment, sr, target_sr)
        else
            layer = render_mode in WCAT_KNOWN_LAYERS ? render_mode : "full"
            if !haskey(layers, layer)
                layer = "full"
            end

            segment = corpus_segment_from_layer_signal(layers[layer], original_start, original_end, config, spray_samples, grain_length_ratio)
            return linear_resample(segment, sr, target_sr)
        end
    end

    chunk_path = String(safe_df_value(df, idx, "chunk_audio_path", ""))

    if !isempty(chunk_path) && isfile(chunk_path)
        if render_mode != "full"
            @warn "Only saved full chunks are available for row $idx; render_source '$render_mode' falls back to full."
        end

        x, sr = read_audio_file(chunk_path; normalize=false)
        segment = crop_or_pad_chunk_with_ratio(x, grain_length_ratio, config)
        return linear_resample(segment, sr, target_sr)
    end

    error("Cannot load corpus audio for row $idx. Missing file_path and chunk_audio_path.")
end


function apply_grain_window!(x, config)
    n = length(x)
    n == 0 && return x

    window_type = String(cfg(config, ["synthesis", "window", "type"], "tukey"))
    window_type = lowercase(window_type)

    if window_type in ["none", "rect", "rectangular"]
        return x
    end

    if window_type in ["hann_full", "full_hann", "hann"]
        if n == 1
            return x
        end

        @inbounds for i in 1:n
            a = 0.5 - 0.5 * cos(2.0 * pi * (i - 1) / (n - 1))
            x[i] *= a
        end

        return x
    end

    if window_type in ["sine_full", "full_sine"]
        if n == 1
            return x
        end

        @inbounds for i in 1:n
            a = sin(pi * (i - 1) / (n - 1))
            x[i] *= a
        end

        return x
    end

    fade_samples = Int(cfg(
        config,
        ["synthesis", "window", "fade_samples"],
        cfg(config, ["synthesis", "rendering", "crossfade_samples"], 128)))

    fade_ratio = Float64(cfg(config, ["synthesis", "window", "fade_ratio"], 0.25))
    fade = min(fade_samples, floor(Int, n * fade_ratio), n ÷ 2)
    fade <= 0 && return x

    @inbounds for i in 1:fade
        u = (i - 1) / fade

        if window_type in ["linear", "fade"]
            a = u
        else
            a = 0.5 - 0.5 * cos(pi * u)
        end

        x[i] *= a
        x[n - i + 1] *= a
    end

    return x
end

function mix_grain!(out, grain, start_sample)
    isempty(grain) && return out

    s = clamp(Int(start_sample), 1, length(out))
    e = min(length(out), s + length(grain) - 1)
    n = e - s + 1

    n <= 0 && return out

    @inbounds for i in 1:n
        out[s + i - 1] += grain[i]
    end

    return out
end

function poisson_count(lambda)
    lambda = max(0.0, Float64(lambda))
    lambda <= 0.0 && return 0

    limit = exp(-lambda)
    k = 0
    p = 1.0

    while p > limit
        k += 1
        p *= rand()
    end

    return k - 1
end

function density_to_grain_count(density, config)
    min_density = Float64(cfg(config, ["synthesis", "density", "min"], 0.1))
    max_density = Float64(cfg(config, ["synthesis", "density", "max"], 4.0))
    mode = String(cfg(config, ["synthesis", "density", "mode"], "dust"))

    d = clamp(Float64(density), 0.0, max_density)
    d <= 0.0 && return 0

    if mode == "dust" || mode == "poisson"
        return poisson_count(d)
    elseif mode == "floor_fraction"
        base = floor(Int, d)
        frac = d - base
        return base + (rand() <= frac ? 1 : 0)
    elseif mode == "bernoulli"
        return rand() <= clamp(d, 0.0, 1.0) ? 1 : 0
    else
        error("Unsupported synthesis.density.mode: $mode")
    end
end

function event_grain_position(event, grain_length, out_length, config)
    start_sample = Int(event["start_sample"])
    end_sample = Int(event["end_sample"])
    placement = String(cfg(config, ["synthesis", "density", "placement"], "inside_event"))

    if placement == "grid"
        place = start_sample
    else
        lo = min(start_sample, end_sample)
        hi = max(start_sample, end_sample)
        place = rand(lo:hi)
    end

    jitter = Int(cfg(config, ["synthesis", "rendering", "grain_time_jitter_samples"], 0))
    if jitter > 0
        place += rand(-jitter:jitter)
    end

    return clamp(place, 1, max(1, out_length - max(1, grain_length) + 1))
end

function update_recent!(recent_indices, recent_files, idx, file_path, memory)
    push!(recent_indices, idx)
    push!(recent_files, file_path)

    while length(recent_indices) > memory
        popfirst!(recent_indices)
    end

    while length(recent_files) > memory
        popfirst!(recent_files)
    end
end

function render_synthesis(target, target_sr, events, corpus_df, corpus_norm, automation, owners, feature_to_indices, config)
    metric = String(cfg(config, ["synthesis", "distance", "metric"], "weighted_euclidean"))
    k_multiplier = Float64(cfg(config, ["synthesis", "matching", "k_multiplier"], 1.0))
    recent_memory = Int(cfg(config, ["synthesis", "matching", "recent_memory"], 12))

    default_density = Float64(cfg(config, ["synthesis", "density", "default"], 1.0))

    default_spray = Float64(cfg(config, ["synthesis", "spray", "default_samples"], 0.0))
    max_spray = Float64(cfg(config, ["synthesis", "spray", "max_samples"], 100.0))

    default_grain_ratio = Float64(cfg(config, ["synthesis", "grain_length", "default_ratio"], 1.0))
    min_grain_ratio = Float64(cfg(config, ["synthesis", "grain_length", "min_ratio"], 0.10))
    max_grain_ratio = Float64(cfg(config, ["synthesis", "grain_length", "max_ratio"], 2.00))

    max_chunk = has_column(corpus_df, "chunk_length") ? maximum(skipmissing(corpus_df[!, :chunk_length])) : 8192
    out = zeros(Float64, length(target) + round(Int, Int(max_chunk) * max_grain_ratio) + 4096)

    cache = Dict{String,Dict{String,Any}}()
    last_end = length(target)
    recent_indices = Int[]
    recent_files = String[]

    for event in events
        t = Float64(event["start_sample"] - 1) / max(1, length(target) - 1)

        density = selection_value(automation, "density", t, default_density)
        n_grains = density_to_grain_count(density, config)
        n_grains == 0 && continue

        k_value = selection_value(automation, "k", t, 1.0)
        k = max(1, round(Int, k_value * k_multiplier))

        spray = selection_value(automation, "spray_samples", t, default_spray)
        spray = clamp(spray, 0.0, max_spray)

        grain_length_ratio = selection_value(automation, "grain_length_ratio", t, default_grain_ratio)
        grain_length_ratio = clamp(grain_length_ratio, min_grain_ratio, max_grain_ratio)

        weights = column_weights(automation, owners, feature_to_indices, t)
        target_norm = event["norm"]

        for grain_i in 1:n_grains
            idx = choose_corpus_index(
                target_norm,
                corpus_norm,
                weights,
                metric,
                corpus_df,
                recent_indices,
                recent_files,
                config,
                k)

            file_path = String(safe_df_value(corpus_df, idx, "file_path", ""))
            render_layer = get_render_source_mode(config) == "matched_layer" ? dominant_layer_from_weights(owners, weights) : "full"
            grain = corpus_segment(corpus_df, idx, target_sr, cache, config, spray, grain_length_ratio; render_layer_override=render_layer)
            grain = copy(grain)
            apply_grain_window!(grain, config)

            place = event_grain_position(event, length(grain), length(out), config)
            mix_grain!(out, grain, place)
            last_end = max(last_end, place + length(grain) - 1)
            update_recent!(recent_indices, recent_files, idx, file_path, recent_memory)
        end
    end

    return out[1:min(last_end, length(out))]
end

function normalize_output_audio!(x, config; eps=1e-12)
    do_normalize = cfg(config, ["synthesis", "rendering", "normalize_output"], true)
    gain = Float64(cfg(config, ["synthesis", "rendering", "output_gain"], 0.95))

    if !do_normalize
        x .*= gain
        return x
    end

    peak = maximum(abs, x)

    if peak > eps
        x .*= gain / peak
    end

    return x
end


function run_concatenative_synthesis(; config, target_path, automation_path, corpus_path, output_path)
    seed = cfg(config, ["synthesis", "seed"], nothing)
    !isnothing(seed) && Random.seed!(Int(seed))

    feature_groups = config_feature_groups(config)
    isempty(feature_groups) && error("No analysis features configured. Check use_features or analysis_layers.features.")

    corpus_df = CSV.read(corpus_path, DataFrame)
    nrow(corpus_df) == 0 && error("Corpus CSV is empty: $corpus_path")

    columns, owners = selected_feature_columns(corpus_df, feature_groups)
    feature_to_indices = build_feature_to_indices(owners)

    corpus_raw = dataframe_to_matrix(corpus_df, columns)
    stats = compute_normalization_stats(corpus_raw, config)
    corpus_norm = normalize_matrix(corpus_raw, stats)

    normalize_target = cfg(config, ["audio", "normalize"], true)
    target, target_sr = read_audio_file(target_path; normalize=normalize_target)

    automation = load_automation_json(automation_path)

    events = target_events(target, target_sr, feature_groups, config, columns, stats)
    isempty(events) && error("No target events survived gating. Lower synthesis.target.energy_gate_db or min_novelty.")

    out = render_synthesis(target, target_sr, events, corpus_df, corpus_norm, automation, owners, feature_to_indices, config)
    normalize_output_audio!(out, config)

    mkpath(dirname(output_path))
    wavwrite(out, output_path; Fs=target_sr)

    println("Wrote concatenative synthesis output: $(output_path)")
    println("Target events used: $(length(events))")
    println("Distance metric: $(cfg(config, ["synthesis", "distance", "metric"], "weighted_euclidean"))")
    println("Normalization: $(cfg(config, ["synthesis", "normalization", "method"], "robust_range"))")

    return out
end
