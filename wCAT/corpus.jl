# AUDIO DISCOVERY / LOADING

function is_audio_file(path)
    ext = lowercase(splitext(path)[2])
    return ext == ".wav"
end


function collect_audio_files(audio_dir)
    if !isdir(audio_dir)
        error("Audio directory does not exist: $audio_dir")
    end

    files = String[]

    for (root, _, names) in walkdir(audio_dir)
        for name in names
            path = joinpath(root, name)

            if is_audio_file(path)
                push!(files, path)
            end
        end
    end

    sort!(files)
    return files
end


# CHUNKING

function fixed_chunks(n_total_samples, chunk_size, hop_size; keep_tail=false)
    chunks = Vector{Tuple{Int,Int}}()
    start_sample = 1

    while start_sample <= n_total_samples
        end_sample = start_sample + chunk_size - 1

        if end_sample <= n_total_samples
            push!(chunks, (start_sample, end_sample))
        elseif keep_tail
            push!(chunks, (start_sample, n_total_samples))
        end

        start_sample += hop_size
    end

    return chunks
end


function onset_flux_envelope(x, sr; frame_size=1024, hop_size=256, n_fft=nothing, window=:hann, eps=1e-12)
    flux = Float64[]
    frame_starts = Int[]

    length(x) < frame_size && return flux, frame_starts

    fft_size = isnothing(n_fft) ? nextpow(2, frame_size) : Int(n_fft)
    fft_size = max(fft_size, frame_size)

    w = analysis_window(frame_size; window=window)

    previous_mag = nothing
    start_sample = 1

    while start_sample + frame_size - 1 <= length(x)
        frame = x[start_sample:start_sample + frame_size - 1]
        windowed = frame .* w

        buffer = zeros(Float64, fft_size)
        buffer[1:frame_size] .= windowed

        mag = abs.(rfft(buffer))

        if isnothing(previous_mag)
            push!(flux, 0.0)
        else
            n = min(length(mag), length(previous_mag))
            diff = mag[1:n] .- previous_mag[1:n]
            value = sum(max.(diff, 0.0)) / (sum(previous_mag[1:n]) + eps)
            push!(flux, value)
        end

        push!(frame_starts, start_sample)

        previous_mag = mag
        start_sample += hop_size
    end

    return flux, frame_starts
end


function onset_threshold(values, sens; eps=1e-12)
    isempty(values) && return Inf

    med = median(values)
    mad = median(abs.(values .- med))
    sigma = 1.4826 * mad

    if sigma <= eps
        sigma = std(values)
    end

    return med + sigma / max(sens, eps)
end


function pick_onsets(flux, frame_starts; sens=1.0, min_gap=512)
    onsets = Int[]
    length(flux) < 3 && return onsets

    th = onset_threshold(flux, sens)
    last_onset = -typemax(Int)

    @inbounds for i in 2:(length(flux) - 1)
        is_local_peak = flux[i] >= flux[i - 1] && flux[i] > flux[i + 1]
        is_above_threshold = flux[i] >= th

        if is_local_peak && is_above_threshold
            sample = frame_starts[i]

            if isempty(onsets) || sample - last_onset >= min_gap
                push!(onsets, sample)
                last_onset = sample
            end
        end
    end

    return onsets
end


function dynamic_chunks(x, sr; sens=1.0, frame_size=1024, hop_size=256, n_fft=nothing, min_len=512, window=:hann)
    flux, frame_starts = onset_flux_envelope(
        x,
        sr;
        frame_size=frame_size,
        hop_size=hop_size,
        n_fft=n_fft,
        window=window)

    onsets = pick_onsets(
        flux,
        frame_starts;
        sens=sens,
        min_gap=min_len)

    boundaries = unique(sort(vcat([1], onsets, [length(x) + 1])))
    chunks = Vector{Tuple{Int,Int}}()

    for i in 1:(length(boundaries) - 1)
        start_sample = boundaries[i]
        end_sample = boundaries[i + 1] - 1
        chunk_length = end_sample - start_sample + 1

        if chunk_length >= min_len
            push!(chunks, (start_sample, end_sample))
        end
    end

    return chunks
end



# OPTIONAL CHUNK STORAGE

function safe_name(s)
    name = replace(String(s), r"[^A-Za-z0-9_\-]" => "_")
    return isempty(name) ? "audio" : name
end


function layer_folder_name(chunk_mode, layer_param)
    safe_param = replace(layer_param, "=" => "")
    safe_param = replace(safe_param, ";" => "_")
    safe_param = replace(safe_param, "." => "p")
    safe_param = safe_name(safe_param)

    return "$(chunk_mode)_$(safe_param)"
end


function save_chunk_audio(chunk, sr, source_path, chunk_index, output_root, chunk_mode, layer_param)
    folder = joinpath(output_root, layer_folder_name(chunk_mode, layer_param))
    mkpath(folder)

    source_base = splitext(basename(source_path))[1]
    source_base = "$(safe_name(source_base))_$(string(hash(source_path), base=16)[1:8])"

    index_string = lpad(string(chunk_index - 1), 4, '0')
    out_path = joinpath(folder, "$(source_base)_$(index_string).wav")

    wavwrite(chunk, out_path; Fs=sr)

    return out_path
end



# FEATURE EXTRACTION 

function analyze_chunks!(rows, path, x, sr, chunks, layer_signals, config; chunk_mode, layer_id, layer_param, store_chunks=false, chunk_audio_root="chunked_audio")
    previous_layer_chunks = Dict{String,Any}()

    for (chunk_index, bounds) in enumerate(chunks)
        start_sample, end_sample = bounds
        full_chunk = x[start_sample:end_sample]
        layer_chunks = layer_chunks_from_bounds(layer_signals, start_sample, end_sample)

        row = Dict{Symbol,Any}()

        row[:file_path] = path
        row[:start_sample] = start_sample
        row[:end_sample] = end_sample
        row[:chunk_length] = end_sample - start_sample + 1
        row[:chunk_mode] = chunk_mode
        row[:layer_id] = layer_id
        row[:layer_param] = layer_param
        row[:chunk_index] = chunk_index
        row[:sample_rate] = sr
        row[:layer_analysis_enabled] = layer_analysis_enabled(config)

        if store_chunks
            chunk_path = save_chunk_audio(full_chunk, sr, path, chunk_index, chunk_audio_root, chunk_mode, layer_param)
            row[:chunk_audio_path] = chunk_path
        end

        extract_layer_features!(row, layer_chunks, previous_layer_chunks, sr, config)

        push!(rows, row)
        previous_layer_chunks = layer_chunks
    end

    return rows
end

function rows_to_dataframe(rows)
    isempty(rows) && return DataFrame()

    all_keys = Set{Symbol}()

    for row in rows
        for key in keys(row)
            push!(all_keys, key)
        end
    end

    base_columns = [
        :file_path,
        :start_sample,
        :end_sample,
        :chunk_length,
        :chunk_mode,
        :layer_id,
        :layer_param,
        :chunk_index,
        :sample_rate,
        :chunk_audio_path]

    remaining_columns = sort(
        collect(setdiff(all_keys, Set(base_columns)));
        by=string)

    columns = vcat(
        [c for c in base_columns if c in all_keys],
        remaining_columns)

    df = DataFrame()

    for column in columns
        df[!, column] = [get(row, column, missing) for row in rows]
    end

    return df
end


# MAIN CORPUS CREATION

function create_corpus_index(config)
    audio_dir = cfg(config, ["audio", "audio_dir"], "audio")
    output_csv = cfg(config, ["audio", "output_csv"], "corpus_index.csv")
    normalize = cfg(config, ["audio", "normalize"], true)

    modes = cfg(config, ["chunking", "modes"], "both")

    fixed_sizes = cfg(config, ["chunking", "fixed", "sizes"], [2048])
    fixed_hops = cfg(config, ["chunking", "fixed", "hops"], [1024])
    keep_tail = cfg(config, ["chunking", "fixed", "keep_tail"], false)

    onset_sensitivities = cfg(config, ["chunking", "dynamic", "onset_sensitivities"], [0.75])
    onset_frame_size = cfg(config, ["chunking", "dynamic", "onset_frame_size"], 1024)
    onset_hop_size = cfg(config, ["chunking", "dynamic", "onset_hop_size"], 256)
    onset_n_fft = cfg(config, ["chunking", "dynamic", "onset_n_fft"], nothing)
    min_chunk_samples = cfg(config, ["chunking", "dynamic", "min_chunk_samples"], 2048)

    store_chunks = cfg(config, ["chunking", "store_chunks"], false)
    chunk_audio_root = cfg(config, ["chunking", "chunk_audio_root"], "chunked_audio")

    feature_groups = config_feature_groups(config)
    isempty(feature_groups) && error("No analysis features configured. Check use_features or analysis_layers.features.")

    if !(modes in ["fixed", "dynamic", "both"])
        error("chunking.modes must be one of: fixed, dynamic, both")
    end

    files = collect_audio_files(audio_dir)

    if isempty(files)
        error("No .wav files found in: $audio_dir")
    end

    if store_chunks
        mkpath(chunk_audio_root)
    end

    rows = Vector{Dict{Symbol,Any}}()

    run_fixed = modes in ["fixed", "both"]
    run_dynamic = modes in ["dynamic", "both"]


    for path in files
        path = abspath(path)
        x, sr = read_audio_file(path; normalize=normalize)
        layer_signals = analysis_layer_signals(x, sr, config)

        if run_fixed
            layer_id = 1

            for chunk_size in fixed_sizes
                for hop_size in fixed_hops
                    chunks = fixed_chunks(length(x), Int(chunk_size), Int(hop_size); keep_tail=keep_tail)
                    layer_param = "size=$(chunk_size);hop=$(hop_size)"

                    analyze_chunks!(
                        rows,
                        path,
                        x,
                        sr,
                        chunks,
                        layer_signals,
                        config;
                        chunk_mode="fixed",
                        layer_id=layer_id,
                        layer_param=layer_param,
                        store_chunks=store_chunks,
                        chunk_audio_root=chunk_audio_root)

                    layer_id += 1
                end
            end
        end

        if run_dynamic
            layer_id = 1
            onset_window = Symbol(cfg(config, ["analysis", "window"], "hann"))

            for sens in onset_sensitivities
                chunks = dynamic_chunks(
                    x,
                    sr;
                    sens=Float64(sens),
                    frame_size=Int(onset_frame_size),
                    hop_size=Int(onset_hop_size),
                    n_fft=onset_n_fft,
                    min_len=Int(min_chunk_samples),
                    window=onset_window)

                layer_param = "sens=$(sens);frame=$(onset_frame_size);hop=$(onset_hop_size)"

                analyze_chunks!(
                    rows,
                    path,
                    x,
                    sr,
                    chunks,
                    layer_signals,
                    config;
                    chunk_mode="dynamic",
                    layer_id=layer_id,
                    layer_param=layer_param,
                    store_chunks=store_chunks,
                    chunk_audio_root=chunk_audio_root)

                layer_id += 1
            end
        end

        println("Indexed $(basename(path))")
    end

    if isempty(rows)
        error("No chunks were produced. Check chunk sizes and onset settings.")
    end

    df = rows_to_dataframe(rows)
    mkpath(dirname(output_csv))
    CSV.write(output_csv, df)

    println()
    println("Wrote $(nrow(df)) chunks to $(output_csv)")

    if store_chunks
        println("Saved chunk audio files to $(chunk_audio_root)")
    else
        println("Chunk audio storage disabled. Set chunking.store_chunks: true to save chunks.")
    end

    return df
end
