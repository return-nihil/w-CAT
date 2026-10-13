const WCAT_KNOWN_LAYERS = ["full", "harmonic", "percussive", "residual"]

function layer_analysis_enabled(config)
    return parse_bool_value(cfg(config, ["analysis_layers", "enabled"], false))
end

function config_layer_names(config)
    if !layer_analysis_enabled(config)
        return ["full"]
    end

    raw = cfg(config, ["analysis_layers", "layers"], ["full", "harmonic", "percussive"])
    layers = [String(x) for x in raw]

    if !("full" in layers)
        pushfirst!(layers, "full")
    end

    return unique(layers)
end

function config_layer_features(config)
    if !layer_analysis_enabled(config)
        return Dict("full" => config_use_features(config))
    end

    raw = cfg(config, ["analysis_layers", "features"], Dict{String,Any}())
    out = Dict{String,Vector{String}}()

    for layer in config_layer_names(config)
        if raw isa AbstractDict && haskey(raw, layer)
            out[layer] = [String(x) for x in get_key_any(raw, layer, String[])]
        else
            out[layer] = String[]
        end
    end

    return out
end

function config_feature_groups(config)
    groups = Vector{Dict{String,String}}()

    if layer_analysis_enabled(config)
        layer_features = config_layer_features(config)

        for layer in config_layer_names(config)
            for feature in get(layer_features, layer, String[])
                label = "$(layer)_$(feature)"
                push!(groups, Dict(
                    "label" => label,
                    "layer" => layer,
                    "feature" => feature))
            end
        end
    else
        for feature in config_use_features(config)
            push!(groups, Dict(
                "label" => feature,
                "layer" => "full",
                "feature" => feature))
        end
    end

    return groups
end

function config_feature_group_labels(config)
    return [g["label"] for g in config_feature_groups(config)]
end

function prefix_row!(row, prefix::String, source)
    for (k, v) in source
        row[Symbol("$(prefix)_$(String(k))")] = v
    end

    return row
end

function extract_layer_features!(row, layer_chunks, previous_layer_chunks, sr, config)
    layer_features = config_layer_features(config)

    if layer_analysis_enabled(config)
        for layer in config_layer_names(config)
            features = get(layer_features, layer, String[])
            isempty(features) && continue

            if !haskey(layer_chunks, layer)
                @warn "Layer '$layer' requested but missing from layer_chunks. Skipping."
                continue
            end

            tmp = Dict{Symbol,Any}()
            chunk = layer_chunks[layer]
            previous_chunk = get(previous_layer_chunks, layer, nothing)
            extract_features!(tmp, chunk, previous_chunk, sr, features, config)
            prefix_row!(row, layer, tmp)
        end
    else
        features = config_use_features(config)
        chunk = layer_chunks["full"]
        previous_chunk = get(previous_layer_chunks, "full", nothing)
        extract_features!(row, chunk, previous_chunk, sr, features, config)
    end

    return row
end

function feature_group_columns(df, group)
    label = String(group["label"])
    feature = String(group["feature"])
    cols = names(df)

    if feature == "yin_pitch"
        out = String[]
        c1 = "$(label)_hz"
        c2 = "$(label)_confidence"
        c1 in cols && push!(out, c1)
        c2 in cols && push!(out, c2)
        return out
    elseif feature == "mfcc" || feature == "chroma" || feature == "tristimulus"
        return sort([c for c in cols if startswith(c, "$(label)_")]; by=c -> parse(Int, split(c, "_")[end]))
    elseif feature == "band_energy_ratios"
        prefix = layer_analysis_enabled_name(label) ? replace(label, "band_energy_ratios" => "band_energy_ratio") : "band_energy_ratio"
        return sort([c for c in cols if startswith(c, "$(prefix)_")]; by=c -> parse(Int, split(c, "_")[end]))
    else
        return label in cols ? [label] : String[]
    end
end

function layer_analysis_enabled_name(label::String)
    for layer in WCAT_KNOWN_LAYERS
        if startswith(label, "$(layer)_")
            return true
        end
    end
    return false
end

function group_label_to_layer(label::String)
    for layer in WCAT_KNOWN_LAYERS
        if label == layer || startswith(label, "$(layer)_")
            return layer
        end
    end

    return "full"
end

function median_in_window(v, center, radius)
    lo = max(1, center - radius)
    hi = min(length(v), center + radius)
    return median(@view v[lo:hi])
end

function stft_matrix(x, n_fft, hop_size; window=:hann)
    x = as_float_vector(x)
    n = length(x)
    n == 0 && return zeros(ComplexF64, 0, 0), Int[]

    n_fft = Int(n_fft)
    hop_size = Int(hop_size)
    n_fft <= 0 && error("HPSS n_fft must be > 0")
    hop_size <= 0 && error("HPSS hop_size must be > 0")

    if n < n_fft
        padded = zeros(Float64, n_fft)
        padded[1:n] .= x
        x = padded
        n = n_fft
    end

    starts = collect(1:hop_size:(n - n_fft + 1))
    if isempty(starts)
        starts = [1]
    end

    w = analysis_window(n_fft; window=window)
    S = zeros(ComplexF64, n_fft ÷ 2 + 1, length(starts))

    for (t, s) in enumerate(starts)
        frame = x[s:s + n_fft - 1] .* w
        S[:, t] .= rfft(frame)
    end

    return S, starts
end

function istft_matrix(S, starts, out_len, n_fft, hop_size; window=:hann, eps=1e-12)
    n_fft = Int(n_fft)
    y_len = max(out_len, isempty(starts) ? out_len : starts[end] + n_fft - 1)
    y = zeros(Float64, y_len)
    wsum = zeros(Float64, y_len)
    w = analysis_window(n_fft; window=window)

    for (t, s) in enumerate(starts)
        frame = irfft(S[:, t], n_fft)
        e = s + n_fft - 1
        y[s:e] .+= frame .* w
        wsum[s:e] .+= w .^ 2
    end

    @inbounds for i in eachindex(y)
        if wsum[i] > eps
            y[i] /= wsum[i]
        end
    end

    if length(y) >= out_len
        return y[1:out_len]
    end

    out = zeros(Float64, out_len)
    out[1:length(y)] .= y
    return out
end

function hpss_separate(x, sr, config)
    hcfg = cfg(config, ["analysis_layers", "hpss"], Dict{String,Any}())

    n_fft = Int(get_key_any(hcfg, "n_fft", 2048))
    hop_size = Int(get_key_any(hcfg, "hop_size", 512))
    window = Symbol(String(get_key_any(hcfg, "window", cfg(config, ["analysis", "window"], "hann"))))
    harmonic_kernel = max(1, Int(get_key_any(hcfg, "harmonic_kernel", 31)))
    percussive_kernel = max(1, Int(get_key_any(hcfg, "percussive_kernel", 31)))
    mask_power = Float64(get_key_any(hcfg, "mask_power", 2.0))
    margin_h = Float64(get_key_any(hcfg, "margin_harmonic", 1.0))
    margin_p = Float64(get_key_any(hcfg, "margin_percussive", 1.0))

    harmonic_kernel += iseven(harmonic_kernel) ? 1 : 0
    percussive_kernel += iseven(percussive_kernel) ? 1 : 0

    S, starts = stft_matrix(x, n_fft, hop_size; window=window)
    isempty(S) && return Dict("full" => as_float_vector(x), "harmonic" => as_float_vector(x), "percussive" => zeros(Float64, length(x)), "residual" => zeros(Float64, length(x)))

    mag = abs.(S)
    n_freqs, n_frames = size(mag)
    Hmed = zeros(Float64, n_freqs, n_frames)
    Pmed = zeros(Float64, n_freqs, n_frames)

    time_radius = harmonic_kernel ÷ 2
    freq_radius = percussive_kernel ÷ 2

    for f in 1:n_freqs
        row = vec(mag[f, :])
        for t in 1:n_frames
            Hmed[f, t] = median_in_window(row, t, time_radius)
        end
    end

    for t in 1:n_frames
        col = vec(mag[:, t])
        for f in 1:n_freqs
            Pmed[f, t] = median_in_window(col, f, freq_radius)
        end
    end

    Hp = Hmed .^ mask_power
    Pp = Pmed .^ mask_power

    Hmask = Hp ./ (Hp .+ (margin_h .* Pp) .+ 1e-12)
    Pmask = Pp ./ ((margin_p .* Hp) .+ Pp .+ 1e-12)

    total_mask = Hmask .+ Pmask
    over = total_mask .> 1.0
    Hmask[over] ./= total_mask[over]
    Pmask[over] ./= total_mask[over]
    Rmask = clamp.(1.0 .- Hmask .- Pmask, 0.0, 1.0)

    Xh = istft_matrix(S .* Hmask, starts, length(x), n_fft, hop_size; window=window)
    Xp = istft_matrix(S .* Pmask, starts, length(x), n_fft, hop_size; window=window)
    Xr = istft_matrix(S .* Rmask, starts, length(x), n_fft, hop_size; window=window)

    return Dict(
        "full" => as_float_vector(x),
        "harmonic" => Xh,
        "percussive" => Xp,
        "residual" => Xr)
end

function analysis_layer_signals(x, sr, config)
    x = as_float_vector(x)

    if !layer_analysis_enabled(config)
        return Dict("full" => x)
    end

    layers = config_layer_names(config)
    need_separation = any(layer -> layer != "full", layers)

    if need_separation
        separated = hpss_separate(x, sr, config)
    else
        separated = Dict("full" => x)
    end

    out = Dict{String,Vector{Float64}}()
    for layer in layers
        if haskey(separated, layer)
            out[layer] = separated[layer]
        else
            out[layer] = zeros(Float64, length(x))
        end
    end

    return out
end

function layer_chunks_from_bounds(layer_signals, start_sample, end_sample)
    out = Dict{String,Vector{Float64}}()

    for (layer, signal) in layer_signals
        s = clamp(start_sample, 1, length(signal))
        e = clamp(end_sample, s, length(signal))
        out[layer] = signal[s:e]
    end

    return out
end

function get_render_source_mode(config)
    raw = cfg(config, ["synthesis", "render_source"], "full")

    if raw isa AbstractDict
        return String(get_key_any(raw, "mode", "full"))
    end

    return String(raw)
end

function render_layer_mix_config(config)
    raw = cfg(config, ["synthesis", "render_source", "layer_mix"], Dict{String,Any}())

    if !(raw isa AbstractDict)
        raw = cfg(config, ["synthesis", "render_layer_mix"], Dict{String,Any}())
    end

    out = Dict{String,Float64}()
    for layer in WCAT_KNOWN_LAYERS
        out[layer] = Float64(get_key_any(raw, layer, layer == "full" ? 1.0 : 0.0))
    end

    return out
end
