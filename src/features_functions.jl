include("audio_helpers.jl")

using Statistics
using LinearAlgebra


# TIME-DOMAIN

function loudness_db(frame; eps=1e-12)
    x = as_float_vector(frame)
    isempty(x) && return -240.0

    r = sqrt(mean(abs2, x))
    return 20.0 * log10(r + eps)
end


function peak_amplitude(frame)
    x = as_float_vector(frame)
    isempty(x) && return 0.0

    return maximum(abs, x)
end


function zero_crossing_rate(frame)
    x = as_float_vector(frame)
    length(x) < 2 && return 0.0

    count = 0

    @inbounds for i in 1:(length(x) - 1)
        if (x[i] >= 0.0) != (x[i + 1] >= 0.0)
            count += 1
        end
    end

    return count / (length(x) - 1)
end


function temporal_centroid_s(frame, sr; eps=1e-12)
    x = as_float_vector(frame)
    isempty(x) && return 0.0

    energy = abs2.(x)
    total = sum(energy)

    total <= eps && return 0.0

    t = collect(0:(length(x) - 1)) ./ Float64(sr)
    return sum(t .* energy) / total
end



# SPECTRUM-DOMAIN

function spectral_centroid_hz(frame, sr; n_fft=nothing, window=:hann, eps=1e-12)
    freqs, mag = magnitude_spectrum(frame, sr; n_fft=n_fft, window=window)

    total = sum(mag)
    total <= eps && return 0.0

    return sum(freqs .* mag) / total
end


function spectral_spread_hz(frame, sr; n_fft=nothing, window=:hann, eps=1e-12)
    freqs, mag = magnitude_spectrum(frame, sr; n_fft=n_fft, window=window)

    total = sum(mag)
    total <= eps && return 0.0

    centroid = sum(freqs .* mag) / total
    return sqrt(sum(((freqs .- centroid) .^ 2) .* mag) / total)
end


function spectral_rolloff_hz(frame, sr; rolloff=0.85, n_fft=nothing, window=:hann, eps=1e-12)
    freqs, mag = magnitude_spectrum(frame, sr; n_fft=n_fft, window=window)

    total = sum(mag)
    total <= eps && return 0.0

    threshold = rolloff * total
    cumulative = cumsum(mag)
    idx = findfirst(v -> v >= threshold, cumulative)

    return isnothing(idx) ? freqs[end] : freqs[idx]
end


function spectral_flatness(frame, sr; n_fft=nothing, window=:hann, eps=1e-12)
    _, mag = magnitude_spectrum(frame, sr; n_fft=n_fft, window=window)

    isempty(mag) && return 0.0

    geometric_mean = exp(mean(log.(mag .+ eps)))
    arithmetic_mean = mean(mag) + eps

    return geometric_mean / arithmetic_mean
end


function spectral_flux(frame, previous_frame, sr; n_fft=nothing, window=:hann, eps=1e-12)
    isnothing(previous_frame) && return 0.0

    _, mag = magnitude_spectrum(frame, sr; n_fft=n_fft, window=window)
    _, previous_mag = magnitude_spectrum(previous_frame, sr; n_fft=n_fft, window=window)

    n = min(length(mag), length(previous_mag))
    n == 0 && return 0.0

    a = mag[1:n] ./ (norm(mag[1:n]) + eps)
    b = previous_mag[1:n] ./ (norm(previous_mag[1:n]) + eps)

    return sqrt(sum((a .- b) .^ 2))
end


function brightness(frame, sr; cutoff_hz=1500.0, n_fft=nothing, window=:hann, eps=1e-12)
    freqs, power = power_spectrum(frame, sr; n_fft=n_fft, window=window)

    total = sum(power)
    total <= eps && return 0.0

    high_energy = 0.0

    @inbounds for i in eachindex(freqs)
        if freqs[i] >= cutoff_hz
            high_energy += power[i]
        end
    end

    return high_energy / total
end



# PITCH + HARMONICITY

function yin_pitch(frame, sr; fmin=50.0, fmax=2000.0, threshold=0.1, eps=1e-12)
    # YIN pitch estimator: squared diff, cumulative mean, CMNDF, absolute-threshold lag search, parabolic lag refinement.
    # Check se funziona
    x = as_float_vector(frame)
    x = x .- mean(x)

    n = length(x)

    if n < 3 || maximum(abs, x) < 1e-8 # più per check frame di silenzio
        return (frequency_hz=0.0, confidence=0.0)
    end

    maxlag = min(n ÷ 2, ceil(Int, sr / fmin))
    minlag = max(2, floor(Int, sr / fmax))

    minlag >= maxlag && return (frequency_hz=0.0, confidence=0.0)

    difference = zeros(Float64, maxlag)

    @inbounds for τ in 1:maxlag
        s = 0.0
        limit = n - τ

        for j in 1:limit
            d = x[j] - x[j + τ]
            s += d * d
        end

        difference[τ] = s
    end

    normalized = ones(Float64, maxlag)
    running_sum = 0.0

    @inbounds for τ in 2:maxlag
        running_sum += difference[τ]
        normalized[τ] = running_sum > eps ? difference[τ] * τ / running_sum : 1.0
    end

    τ_estimate = 0

    @inbounds for τ0 in minlag:maxlag
        τ = τ0

        if normalized[τ] < threshold
            while τ + 1 <= maxlag && normalized[τ + 1] < normalized[τ]
                τ += 1
            end

            τ_estimate = τ
            break
        end
    end

    if τ_estimate == 0
        _, idx = findmin(@view normalized[minlag:maxlag])
        τ_estimate = minlag + idx - 1
    end

    τ_refined = Float64(τ_estimate)

    if 2 <= τ_estimate <= maxlag - 1
        y1 = normalized[τ_estimate - 1]
        y2 = normalized[τ_estimate]
        y3 = normalized[τ_estimate + 1]

        denominator = y1 - 2.0 * y2 + y3

        if abs(denominator) > eps
            τ_refined += 0.5 * (y1 - y3) / denominator
        end
    end

    frequency = Float64(sr) / τ_refined

    if !(fmin <= frequency <= fmax)
        return (frequency_hz=0.0, confidence=0.0)
    end

    confidence = clamp(1.0 - normalized[τ_estimate], 0.0, 1.0)

    return (frequency_hz=frequency, confidence=confidence)
end


function harmonicity(frame, sr; fmin=50.0, fmax=2000.0)
    ac = autocorrelation(frame)

    isempty(ac) && return 0.0

    minlag = max(2, floor(Int, sr / fmax))
    maxlag = min(length(ac), ceil(Int, sr / fmin))

    minlag >= maxlag && return 0.0

    return clamp(maximum(@view ac[minlag:maxlag]), 0.0, 1.0)
end



# PER-BAND

function band_energy_ratio(frame, sr, low_hz, high_hz; n_fft=nothing, window=:hann, eps=1e-12)
    freqs, power = power_spectrum(frame, sr; n_fft=n_fft, window=window)

    total = sum(power)
    total <= eps && return 0.0

    band_total = 0.0
    upper = isinf(high_hz) ? maximum(freqs) + 1.0 : high_hz

    @inbounds for i in eachindex(freqs)
        if low_hz <= freqs[i] < upper
            band_total += power[i]
        end
    end

    return band_total / total
end


function band_energy_ratios(frame, sr, bands; n_fft=nothing, window=:hann)
    values = Float64[]

    for band in bands
        low_hz, high_hz = band
        push!(values, band_energy_ratio(frame, sr, low_hz, high_hz; n_fft=n_fft, window=window))
    end

    return values
end



# CEPSTRUM

function mfcc(frame, sr; ncoeffs=8, nbands=24, n_fft=nothing, window=:hann, eps=1e-12)
    # MFCC semplificato: power spectrum -> mel tri filterbank -> log band energies -> DCT-II
    x = as_float_vector(frame)

    fft_size = isnothing(n_fft) ? length(x) : Int(n_fft)

    _, power = power_spectrum(x, sr; n_fft=fft_size, window=window)
    filters = mel_filterbank(fft_size, sr; nbands=nbands)

    n = min(size(filters, 2), length(power))
    energies = filters[:, 1:n] * power[1:n]

    log_energies = log.(energies .+ eps)

    return dct_ii(log_energies, ncoeffs)
end



# TONAL

function chroma(frame, sr; fmin=27.5, n_fft=nothing, window=:hann, eps=1e-12)
    freqs, power = power_spectrum(frame, sr; n_fft=n_fft, window=window)

    values = zeros(Float64, 12)

    @inbounds for i in eachindex(freqs)
        f = freqs[i]

        if f >= fmin && power[i] > eps
            midi = round(Int, 69 + 12 * log2(f / 440.0))
            pitch_class = mod(midi, 12) + 1
            values[pitch_class] += power[i]
        end
    end

    total = sum(values)
    total <= eps && return values

    return values ./ total
end



# HARMONIC

function inharmonicity(frame, sr, f0; n_harmonics=10, tolerance=0.04, n_fft=nothing, window=:hann, eps=1e-12)
    freqs, mag = magnitude_spectrum(frame, sr; n_fft=n_fft, window=window)

    f0 <= 0.0 && return 0.0

    nyquist = maximum(freqs)
    weighted_deviation = 0.0
    total_weight = 0.0

    for h in 1:n_harmonics
        target = h * f0
        target > nyquist && break

        width = max(20.0, tolerance * target)

        candidates = findall(f -> target - width <= f <= target + width, freqs)
        isempty(candidates) && continue

        local_mag = mag[candidates]
        amplitude, local_idx = findmax(local_mag)
        peak_frequency = freqs[candidates[local_idx]]

        weighted_deviation += amplitude * abs(peak_frequency - target) / target
        total_weight += amplitude
    end

    return total_weight > eps ? weighted_deviation / total_weight : 0.0
end


function tristimulus(frame, sr, f0; n_harmonics=10, tolerance=0.04, n_fft=nothing, window=:hann, eps=1e-12)
    # Tristimulus stile Essentia 
    # (https://github.com/MTG/essentia/blob/v2.1_beta5-1445-gb9fa6cb6/src/algorithms/tonal/tristimulus.cpp)
    # forse inutile, check
    amps = harmonic_amplitudes(
        frame,
        sr,
        f0;
        n_harmonics=n_harmonics,
        tolerance=tolerance,
        n_fft=n_fft,
        window=window)

    total = sum(amps)
    total <= eps && return (0.0, 0.0, 0.0)

    first_harmonic = amps[1] / total
    middle_harmonics = length(amps) >= 2 ? sum(@view amps[2:min(4, length(amps))]) / total : 0.0
    upper_harmonics = length(amps) >= 5 ? sum(@view amps[5:end]) / total : 0.0

    return first_harmonic, middle_harmonics, upper_harmonics
end



function odd_even_ratio(frame, sr, f0; n_harmonics=10, tolerance=0.04, n_fft=nothing, window=:hann, eps=1e-12)
    amps = harmonic_amplitudes(
        frame,
        sr,
        f0;
        n_harmonics=n_harmonics,
        tolerance=tolerance,
        n_fft=n_fft,
        window=window)

    odd = sum(amps[1:2:end])
    even = length(amps) >= 2 ? sum(amps[2:2:end]) : 0.0

    return odd / (even + eps)
end



# PERCEPTUAL

function roughness(frame, sr; max_peaks=30, min_prom=0.01, n_fft=nothing, window=:hann, eps=1e-12)
    # Basato su Dissonance in Essentia 
    # https://essentia.upf.edu/reference/std_Dissonance.html
    peaks = spectral_peaks(
        frame,
        sr;
        min_freq=20.0,
        max_peaks=max_peaks,
        min_prom=min_prom,
        n_fft=n_fft,
        window=window,
        eps=eps)

    length(peaks) < 2 && return 0.0
    sort!(peaks, by=p -> p[1])

    frequencies = [p[1] for p in peaks]
    loudness = [p[2]^2 for p in peaks] # mag -> power 

    total_loudness = 0.0

    @inbounds for i in eachindex(loudness)
        aw = a_weighting(frequencies[i])
        loudness[i] *= aw * aw
        total_loudness += loudness[i]
    end

    total_loudness <= eps && return 0.0

    total_dissonance = 0.0
    n = length(frequencies)

    @inbounds for p1 in 1:n
        f1 = frequencies[p1]
        f1 <= 50.0 && continue # filtro 50 Hx (togliere???)

        bark_f1 = hz_to_bark(f1)
        start_f = bark_to_hz(bark_f1 - 1.18)
        end_f = bark_to_hz(bark_f1 + 1.18)

        peak_dissonance = 0.0

        for p2 in 1:n
            f2 = frequencies[p2]

            if f2 >= start_f && f2 < end_f && f2 < 10000.0
                d = 1.0 - pair_consonance(f1, f2)

                if d > 0.0
                    peak_dissonance += d * (loudness[p1] + loudness[p2]) / total_loudness
                end
            end
        end

        partial_loudness = loudness[p1] / total_loudness
        peak_dissonance = min(peak_dissonance, partial_loudness)

        total_dissonance += peak_dissonance
    end

    return max(total_dissonance / 2.0, 0.0)
end