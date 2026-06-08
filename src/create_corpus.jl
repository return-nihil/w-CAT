include("features_functions.jl")

using WAV
using CSV
using DataFrames
using ProgressMeter
using Statistics
using LinearAlgebra
using FFTW
using DSP



# PARSER

function parse_cli_args(args)
    parsed = Dict{String,String}()
    i = 1

    while i <= length(args)
        key = args[i]

        if !startswith(key, "--")
            error("Unexpected argument: $key")
        end

        clean_key = replace(key, "--" => "")

        if i == length(args) || startswith(args[i + 1], "--")
            parsed[clean_key] = "true"
            i += 1
        else
            parsed[clean_key] = args[i + 1]
            i += 2
        end
    end

    return parsed
end


function get_arg(args, key, default)
    return haskey(args, key) ? args[key] : default
end


function parse_int_list(s)
    s = strip(String(s))
    isempty(s) && return Int[]
    return [parse(Int, strip(x)) for x in split(s, ",")]
end


function parse_float_list(s)
    s = strip(String(s))
    isempty(s) && return Float64[]
    return [parse(Float64, strip(x)) for x in split(s, ",")]
end


function parse_string_list(s)
    s = strip(String(s))
    isempty(s) && return String[]
    return [strip(x) for x in split(s, ",")]
end


function parse_bool(s)
    s = lowercase(strip(String(s)))
    return s in ["true", "1", "yes", "y"]
end


function parse_optional_int(s)
    s = lowercase(strip(String(s)))

    if s in ["", "nothing", "none", "null"]
        return nothing
    end

    return parse(Int, s)
end



# AUDIO LOAD

function is_audio_file(path)
    ext = lowercase(splitext(path)[2])
    return ext == ".wav" # <--- per ora solo wav
end


function collect_audio_files(audio_dir)
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


function read_audio_file(path; normalize=true)
    x, sr = wavread(path)
    x = to_mono(x)

    if normalize
        x = normalize_audio(x)
    else
        x = Float64.(x)
    end

    return x, Int(sr)
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
    σ = 1.4826 * mad

    if σ <= eps
        σ = std(values)
    end

    return med + σ / max(sens, eps)
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



# SAVE CHUNKS

function safe_name(s)
    name = replace(s, r"[^A-Za-z0-9_\-]" => "_")
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
    source_base = safe_name(source_base)

    index_string = lpad(string(chunk_index - 1), 4, '0')
    out_path = joinpath(folder, "$(source_base)_$(index_string).wav")

    wavwrite(chunk, out_path; Fs=sr)

    return out_path
end



# HELPERS (spostare!!!!)

function add_scalar!(row, name, value)
    row[Symbol(name)] = Float64(value)
    return row
end


function add_vector!(row, prefix, values)
    for i in eachindex(values)
        row[Symbol("$(prefix)_$(i)")] = Float64(values[i])
    end

    return row
end


function add_tuple3!(row, prefix, values)
    row[Symbol("$(prefix)_1")] = Float64(values[1])
    row[Symbol("$(prefix)_2")] = Float64(values[2])
    row[Symbol("$(prefix)_3")] = Float64(values[3])

    return row
end


function require_function(name::String)
    isdefined(Main, Symbol(name)) || error("Feature function '$name' undefined.")
end



# FEATURE EXTRACTION

function estimate_f0_for_harmonic_features(chunk, sr)
    require_function("yin_pitch")

    pitch = yin_pitch(chunk, sr)

    return pitch.frequency_hz, pitch.confidence
end


function extract_features!(row, chunk, previous_chunk, sr, feat_list; n_fft=nothing, window=:hann)
    # per il momento tutte le fft sono passate uguali a tutte le funzioni.
    cached_f0 = nothing
    cached_pitch_confidence = nothing

    function get_f0()
        if isnothing(cached_f0)
            f0, conf = estimate_f0_for_harmonic_features(chunk, sr)
            cached_f0 = f0
            cached_pitch_confidence = conf
        end

        return cached_f0, cached_pitch_confidence
    end

    for feature in feat_list
        if feature == "loudness_db"
            require_function("loudness_db")
            value = loudness_db(chunk)
            add_scalar!(row, "loudness_db", value)

        elseif feature == "peak_amplitude"
            require_function("peak_amplitude")
            value = peak_amplitude(chunk)
            add_scalar!(row, "peak_amplitude", value)

        elseif feature == "zero_crossing_rate"
            require_function("zero_crossing_rate")
            value = zero_crossing_rate(chunk)
            add_scalar!(row, "zero_crossing_rate", value)

        elseif feature == "temporal_centroid_s"
            require_function("temporal_centroid_s")
            value = temporal_centroid_s(chunk, sr)
            add_scalar!(row, "temporal_centroid_s", value)

        elseif feature == "spectral_centroid_hz"
            require_function("spectral_centroid_hz")
            value = spectral_centroid_hz(chunk, sr; n_fft=n_fft, window=window)
            add_scalar!(row, "spectral_centroid_hz", value)

        elseif feature == "spectral_spread_hz"
            require_function("spectral_spread_hz")
            value = spectral_spread_hz(chunk, sr; n_fft=n_fft, window=window)
            add_scalar!(row, "spectral_spread_hz", value)

        elseif feature == "spectral_rolloff_hz"
            require_function("spectral_rolloff_hz")
            value = spectral_rolloff_hz(chunk, sr; n_fft=n_fft, window=window)
            add_scalar!(row, "spectral_rolloff_hz", value)

        elseif feature == "spectral_flatness"
            require_function("spectral_flatness")
            value = spectral_flatness(chunk, sr; n_fft=n_fft, window=window)
            add_scalar!(row, "spectral_flatness", value)

        elseif feature == "spectral_flux"
            require_function("spectral_flux")
            value = spectral_flux(chunk, previous_chunk, sr; n_fft=n_fft, window=window)
            add_scalar!(row, "spectral_flux", value)

        elseif feature == "brightness"
            require_function("brightness")
            value = brightness(chunk, sr; n_fft=n_fft, window=window)
            add_scalar!(row, "brightness", value)

        elseif feature == "yin_pitch"
            require_function("yin_pitch")
            pitch = yin_pitch(chunk, sr)
            add_scalar!(row, "yin_pitch_hz", pitch.frequency_hz)
            add_scalar!(row, "yin_pitch_confidence", pitch.confidence)

        elseif feature == "harmonicity"
            require_function("harmonicity")
            value = harmonicity(chunk, sr)
            add_scalar!(row, "harmonicity", value)

        #=elseif feature == "band_energy_ratios"
            require_function("band_energy_ratios")

            # Default broad bands. Change these if you want.
            bands = [
                (20.0, 250.0),
                (250.0, 1000.0),
                (1000.0, 4000.0),
                (4000.0, Inf),
            ]

            values = band_energy_ratios(chunk, sr, bands; n_fft=n_fft, window=window)
            add_vector!(row, "band_energy_ratio", values)=#

        elseif feature == "mfcc"
            require_function("mfcc")
            values = mfcc(chunk, sr; n_fft=n_fft, window=window)
            add_vector!(row, "mfcc", values)

        elseif feature == "chroma"
            require_function("chroma")
            values = chroma(chunk, sr; n_fft=n_fft, window=window)
            add_vector!(row, "chroma", values)

        elseif feature == "inharmonicity"
            require_function("inharmonicity")
            f0, _ = get_f0()
            value = inharmonicity(chunk, sr, f0; n_fft=n_fft, window=window)
            add_scalar!(row, "inharmonicity", value)

        elseif feature == "tristimulus"
            require_function("tristimulus")
            f0, _ = get_f0()
            values = tristimulus(chunk, sr, f0; n_fft=n_fft, window=window)
            add_tuple3!(row, "tristimulus", values)

        elseif feature == "odd_even_ratio"
            require_function("odd_even_ratio")
            f0, _ = get_f0()
            value = odd_even_ratio(chunk, sr, f0; n_fft=n_fft, window=window)
            add_scalar!(row, "odd_even_ratio", value)

        elseif feature == "roughness"
            require_function("roughness")
            value = roughness(chunk, sr; n_fft=n_fft, window=window)
            add_scalar!(row, "roughness", value)
            
        elseif feature == "sharpness_acum"
            require_function("sharpness_acum")
            value = sharpness_acum(chunk, sr; n_fft=n_fft, window=window)
            add_scalar!(row, "sharpness_acum", value)

        else
            error("Unknown feature: $feature")
        end
    end

    return row
end



# ANALYZE CHUNKS AND BUILD ROWS

function analyze_chunks!(rows, path, x, sr, chunks, features; chunk_mode, layer_id, layer_param, n_fft=nothing, window=:hann, save_chunks=false, chunk_audio_root="chunked_audio")
    previous_chunk = nothing

    for (chunk_index, bounds) in enumerate(chunks)
        start_sample, end_sample = bounds
        chunk = x[start_sample:end_sample]

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

        if save_chunks
            chunk_path = save_chunk_audio(chunk, sr, path, chunk_index, chunk_audio_root, chunk_mode, layer_param)
            row[:chunk_audio_path] = chunk_path
        end

        extract_features!(row, chunk, previous_chunk, sr, features; n_fft=n_fft, window=window)

        push!(rows, row)
        previous_chunk = chunk
    end

    return rows
end



# DATAFRAME

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



# CORPUS CREATION

function create_corpus_index(; audio_dir="audio_data", output_csv="corpus_index.csv", modes="both", fixed_sizes=[2048], fixed_hops=[1024], onset_sensitivities=[1.0], onset_frame_size=1024, onset_hop_size=256, onset_n_fft=nothing, min_chunk_samples=2048, keep_tail=false, normalize=true, save_chunks=false, chunk_audio_root="chunked_audio", features=["loudness_db", "spectral_centroid_hz", "spectral_spread_hz", "spectral_flatness", "yin_pitch", "roughness"], n_fft=nothing, window=:hann)
    if !(modes in ["fixed", "dynamic", "both"])
        error("--modes: fixed, dynamic, both")
    end

    files = collect_audio_files(audio_dir)

    if isempty(files)
        error("No .wav files found in: $audio_dir")
    end

    if save_chunks
        mkpath(chunk_audio_root)
    end

    rows = Vector{Dict{Symbol,Any}}()

    run_fixed = modes in ["fixed", "both"]
    run_dynamic = modes in ["dynamic", "both"]

    progress = Progress(length(files); desc="Analyzing corpus", showspeed=true) # sta roba al momento va solo per più files, check

    for path in files
        x, sr = read_audio_file(path; normalize=normalize)

        if run_fixed
            layer_id = 1

            for chunk_size in fixed_sizes
                for hop_size in fixed_hops
                    chunks = fixed_chunks(length(x), chunk_size, hop_size; keep_tail=keep_tail)
                    layer_param = "size=$(chunk_size);hop=$(hop_size)"

                    analyze_chunks!(rows, path, x, sr, chunks, features; chunk_mode="fixed", layer_id=layer_id, layer_param=layer_param, n_fft=n_fft, window=window, save_chunks=save_chunks, chunk_audio_root=chunk_audio_root)

                    layer_id += 1
                end
            end
        end

        if run_dynamic
            layer_id = 1

            for sens in onset_sensitivities
                chunks = dynamic_chunks(x, sr; sens=sens, frame_size=onset_frame_size, hop_size=onset_hop_size, n_fft=onset_n_fft, min_len=min_chunk_samples, window=window)
                layer_param = "sens=$(sens);frame=$(onset_frame_size);hop=$(onset_hop_size)"

                analyze_chunks!(rows, path, x, sr, chunks, features; chunk_mode="dynamic", layer_id=layer_id, layer_param=layer_param, n_fft=n_fft, window=window, save_chunks=save_chunks, chunk_audio_root=chunk_audio_root)

                layer_id += 1
            end
        end

        next!(progress)
    end

    if isempty(rows)
        error("No chunks were produced. Check chunk sizes and onset settings.")
    end

    df = rows_to_dataframe(rows)
    CSV.write(output_csv, df)

    println()
    println("Wrote $(nrow(df)) chunks to $(output_csv)")

    if save_chunks
        println("Saved chunk audio files to $(chunk_audio_root)")
    end

    return df
end



# MAIN

function main()
    args = parse_cli_args(ARGS)

    audio_dir = get_arg(args, "audio_dir", "audio")
    output_csv = get_arg(args, "output", "corpus_index.csv")
    modes = get_arg(args, "modes", "both")

    fixed_sizes = parse_int_list(get_arg(args, "fixed_sizes", "2048"))
    fixed_hops = parse_int_list(get_arg(args, "fixed_hops", "1024"))

    onset_sensitivities = parse_float_list(get_arg(args, "onset_sensitivities", "0.75"))

    onset_frame_size = parse(Int, get_arg(args, "onset_frame_size", "1024"))
    onset_hop_size = parse(Int, get_arg(args, "onset_hop_size", "256"))
    onset_n_fft = parse_optional_int(get_arg(args, "onset_n_fft", "nothing"))

    min_chunk_samples = parse(Int, get_arg(args, "min_chunk_samples", "2048"))

    keep_tail = parse_bool(get_arg(args, "keep_tail", "false"))
    normalize = parse_bool(get_arg(args, "normalize", "true"))

    save_chunks = parse_bool(get_arg(args, "save_chunks", "true"))
    chunk_audio_root = get_arg(args, "chunk_audio_root", "chunked_audio")

    n_fft = parse_optional_int(get_arg(args, "n_fft", "nothing"))
    window = Symbol(get_arg(args, "window", "hann"))

    features = parse_string_list(get_arg(args, "features", "loudness_db,spectral_centroid_hz,spectral_spread_hz,spectral_flatness,yin_pitch,roughness"))

    isempty(fixed_sizes) && error("--fixed_sizes cannot be empty")
    isempty(fixed_hops) && error("--fixed_hops cannot be empty")
    isempty(onset_sensitivities) && error("--onset_sensitivities cannot be empty")
    isempty(features) && error("--features cannot be empty")

    create_corpus_index(
        audio_dir=audio_dir,
        output_csv=output_csv,
        modes=modes,
        fixed_sizes=fixed_sizes,
        fixed_hops=fixed_hops,
        onset_sensitivities=onset_sensitivities,
        onset_frame_size=onset_frame_size,
        onset_hop_size=onset_hop_size,
        onset_n_fft=onset_n_fft,
        min_chunk_samples=min_chunk_samples,
        keep_tail=keep_tail,
        normalize=normalize,
        save_chunks=save_chunks,
        chunk_audio_root=chunk_audio_root,
        features=features,
        n_fft=n_fft,
        window=window)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end