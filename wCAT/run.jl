using WAV, CSV, DataFrames, FFTW, YAML, JSON3
using Statistics, LinearAlgebra, Random


function cfg(config, keys::Vector{String}, default)
    value = config
    for key in keys
        value = value isa AbstractDict ? get(value, key, nothing) : nothing
        isnothing(value) && return default
    end
    return value
end

get_key_any(dict, key, default=nothing) = get(dict, key, default)
parse_bool_value(x) = x isa Bool ? x : lowercase(string(x)) in ("true", "1", "yes")
config_use_features(config) = String.(cfg(config, ["use_features"], String[]))
function feature_default_params(config, name)
    return get(cfg(config, ["feature_defaults"], Dict()), name, Dict())
end

const FFT_FEATURES = Set(("spectral_centroid_hz", "spectral_spread_hz", "spectral_rolloff_hz", "spectral_flatness", "spectral_flux", "brightness", "mfcc", "chroma", "inharmonicity", "tristimulus", "odd_even_ratio", "roughness", "sharpness_acum", "band_energy_ratios"))
function build_feature_kwargs(config, feature::String; include_fft=feature in FFT_FEATURES)
    kwargs = Dict{Symbol,Any}()
    if include_fft
        kwargs[:n_fft] = cfg(config, ["analysis", "n_fft"], nothing)
        kwargs[:window] = Symbol(cfg(config, ["analysis", "window"], "hann"))
    end
    for (key, value) in feature_default_params(config, feature)
        kwargs[Symbol(key)] = key == "window" ? Symbol(value) : value
    end
    return kwargs
end

function read_audio_file(path; normalize=true)
    x, sr = wavread(path)
    mono = to_mono(x)
    return (normalize ? normalize_audio(mono) : mono), Int(sr)
end

include("features.jl")
include("layers.jl")
include("corpus.jl")
include("synthesis.jl")

function project_path(base, path)
    p = expanduser(String(path))
    return isabspath(p) ? normpath(p) : normpath(joinpath(base, p))
end

function load_config(path)
    config_file = abspath(expanduser(path))
    config = YAML.load_file(config_file)
    base = dirname(config_file)
    audio = get!(config, "audio", Dict{String,Any}())
    for (key, fallback) in (("audio_dir", "audio"), ("output_csv", "corpus_index.csv"))
        audio[key] = project_path(base, get(audio, key, fallback))
    end
    chunk = get!(config, "chunking", Dict{String,Any}())
    chunk["chunk_audio_root"] = project_path(base, get(chunk, "chunk_audio_root", "chunks"))
    return config, base
end

function cli(args)
    isempty(args) && error("Usage: julia --project=. run.jl index | synth --target file.wav [--config config.yaml] [--output out.wav] [--automation automation.json] [--corpus corpus_index.csv]")
    command, rest = first(args), args[2:end]
    options = Dict{String,String}()
    i = 1
    while i <= length(rest)
        startswith(rest[i], "--") || error("Expected --option, got $(rest[i])")
        i < length(rest) || error("Missing value for $(rest[i])")
        options[rest[i][3:end]] = rest[i+1]
        i += 2
    end
    config, base = load_config(get(options, "config", joinpath(@__DIR__, "config.yaml")))
    if command == "index"
        if haskey(options, "audio")
            config["audio"]["audio_dir"] = project_path(base, options["audio"])
        end
        create_corpus_index(config)
    elseif command == "synth"
        target = get(options, "target", "target.wav")
        automation = get(options, "automation", "automation.json")
        corpus = get(options, "corpus", config["audio"]["output_csv"])
        output = get(options, "output", "output.wav")
        run_concatenative_synthesis(config=config, target_path=project_path(base, target), automation_path=project_path(base, automation), corpus_path=project_path(base, corpus), output_path=project_path(base, output))
    else
        error("Unknown command '$command'. Use 'index' or 'synth'.")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    cli(ARGS)
end
