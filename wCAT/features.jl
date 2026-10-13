
function as_float_vector(frame)
    if frame isa AbstractVector
        return Float64.(frame)
    end

    if frame isa AbstractMatrix
        return vec(Float64.(mean(frame; dims=2)))
    end

    return Float64.(vec(frame))
end

function to_mono(x)
    if x isa AbstractVector
        return Float64.(x)
    end

    if x isa AbstractMatrix
        return vec(Float64.(mean(x; dims=2)))
    end

    return Float64.(vec(x))
end

function normalize_audio(x; eps=1e-12)
    y = as_float_vector(x)
    isempty(y) && return y

    peak = maximum(abs, y)
    peak <= eps && return y

    return y ./ peak
end

function analysis_window(n::Integer; window=:hann)
    n <= 0 && return Float64[]
    n == 1 && return [1.0]

    w = Symbol(window)

    if w == :hann || w == :hanning
        return [0.5 - 0.5 * cos(2.0 * pi * (i - 1) / (n - 1)) for i in 1:n]
    elseif w == :hamming
        return [0.54 - 0.46 * cos(2.0 * pi * (i - 1) / (n - 1)) for i in 1:n]
    elseif w == :blackman
        return [0.42 - 0.5 * cos(2.0 * pi * (i - 1) / (n - 1)) + 0.08 * cos(4.0 * pi * (i - 1) / (n - 1)) for i in 1:n]
    elseif w == :rect || w == :rectangular || w == :none
        return ones(Float64, n)
    else
        error("Unsupported analysis window: $(window)")
    end
end

function fft_size_for_frame(n::Integer, n_fft)
    fft_size = isnothing(n_fft) ? nextpow(2, n) : Int(n_fft)
    return max(fft_size, n)
end

function magnitude_spectrum(frame, sr; n_fft=nothing, window=:hann)
    x = as_float_vector(frame)
    isempty(x) && return Float64[], Float64[]

    n = length(x)
    fft_size = fft_size_for_frame(n, n_fft)
    w = analysis_window(n; window=window)

    buffer = zeros(Float64, fft_size)
    buffer[1:n] .= x .* w

    spectrum = rfft(buffer)
    mag = abs.(spectrum)
    freqs = collect(0:(length(mag) - 1)) .* (Float64(sr) / fft_size)

    return freqs, mag
end

function power_spectrum(frame, sr; n_fft=nothing, window=:hann)
    freqs, mag = magnitude_spectrum(frame, sr; n_fft=n_fft, window=window)
    return freqs, mag .^ 2
end

function autocorrelation(frame; maxlag=nothing, eps=1e-12)
    x = as_float_vector(frame)
    n = length(x)
    n < 2 && return Float64[]

    x = x .- mean(x)
    energy = sum(abs2, x)
    energy <= eps && return zeros(Float64, n)

    lag_max = isnothing(maxlag) ? n - 1 : min(Int(maxlag), n - 1)
    ac = zeros(Float64, lag_max + 1)

    @inbounds for lag in 0:lag_max
        s = 0.0
        limit = n - lag

        for i in 1:limit
            s += x[i] * x[i + lag]
        end

        ac[lag + 1] = s / energy
    end

    return ac
end

function hz_to_mel(f)
    return 2595.0 * log10(1.0 + Float64(f) / 700.0)
end

function mel_to_hz(m)
    return 700.0 * (10.0 ^ (Float64(m) / 2595.0) - 1.0)
end

function mel_filterbank(fft_size, sr; nbands=24, fmin=0.0, fmax=nothing)
    n_bins = fft_size ÷ 2 + 1
    max_hz = isnothing(fmax) ? Float64(sr) / 2.0 : Float64(fmax)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(max_hz)
    mel_points = range(mel_min, mel_max; length=nbands + 2)
    hz_points = mel_to_hz.(mel_points)
    bin_points = clamp.(floor.(Int, (fft_size .* hz_points ./ Float64(sr)) .+ 1), 1, n_bins)

    filters = zeros(Float64, nbands, n_bins)

    for band in 1:nbands
        left = bin_points[band]
        center = bin_points[band + 1]
        right = bin_points[band + 2]

        if center > left
            for k in left:center
                filters[band, k] = (k - left) / max(center - left, 1)
            end
        end

        if right > center
            for k in center:right
                filters[band, k] = max(filters[band, k], (right - k) / max(right - center, 1))
            end
        end
    end

    return filters
end

function dct_ii(x, ncoeffs)
    values = Float64.(x)
    n = length(values)
    out = zeros(Float64, ncoeffs)

    n == 0 && return out

    @inbounds for k in 1:ncoeffs
        s = 0.0

        for i in 1:n
            s += values[i] * cos(pi * (i - 0.5) * (k - 1) / n)
        end

        out[k] = s
    end

    return out
end

function spectral_peaks(frame, sr; min_freq=20.0, max_freq=nothing, max_peaks=30, min_prom=0.01, n_fft=nothing, window=:hann, eps=1e-12)
    freqs, mag = magnitude_spectrum(frame, sr; n_fft=n_fft, window=window)
    length(mag) < 3 && return Vector{Tuple{Float64,Float64}}()

    upper = isnothing(max_freq) ? Float64(sr) / 2.0 : Float64(max_freq)
    max_mag = maximum(mag)
    max_mag <= eps && return Vector{Tuple{Float64,Float64}}()

    threshold = min_prom * max_mag
    peaks = Vector{Tuple{Float64,Float64}}()

    @inbounds for i in 2:(length(mag) - 1)
        if freqs[i] < min_freq || freqs[i] > upper
            continue
        end

        if mag[i] >= threshold && mag[i] >= mag[i - 1] && mag[i] > mag[i + 1]
            push!(peaks, (Float64(freqs[i]), Float64(mag[i])))
        end
    end

    sort!(peaks, by=p -> p[2], rev=true)

    if length(peaks) > max_peaks
        resize!(peaks, max_peaks)
    end

    return peaks
end

function a_weighting(f)
    f = Float64(f)
    f <= 0.0 && return 0.0

    f2 = f * f
    ra_num = (12194.0 ^ 2) * (f2 ^ 2)
    ra_den = (f2 + 20.6 ^ 2) * sqrt((f2 + 107.7 ^ 2) * (f2 + 737.9 ^ 2)) * (f2 + 12194.0 ^ 2)
    ra = ra_num / ra_den
    db = 20.0 * log10(max(ra, 1e-30)) + 2.0

    return 10.0 ^ (db / 20.0)
end

function hz_to_bark(f)
    return 6.0 * asinh(Float64(f) / 600.0)
end

function bark_to_hz(z)
    return 600.0 * sinh(Float64(z) / 6.0)
end

function pair_consonance(f1, f2; eps=1e-12)
    f1 = Float64(f1)
    f2 = Float64(f2)

    if f1 <= 0.0 || f2 <= 0.0
        return 1.0
    end

    df = abs(f2 - f1)
    df <= eps && return 1.0

    fmin = min(f1, f2)
    s = 0.24 / (0.021 * fmin + 19.0)
    rough = exp(-3.5 * s * df) - exp(-5.75 * s * df)

    return clamp(1.0 - rough / 0.25, 0.0, 1.0)
end

function harmonic_amplitudes(frame, sr, f0; n_harmonics=10, tolerance=0.04, n_fft=nothing, window=:hann)
    freqs, mag = magnitude_spectrum(frame, sr; n_fft=n_fft, window=window)
    amps = zeros(Float64, n_harmonics)

    f0 <= 0.0 && return amps
    isempty(freqs) && return amps

    nyquist = maximum(freqs)

    for h in 1:n_harmonics
        target = h * f0
        target > nyquist && break

        width = max(20.0, tolerance * target)
        candidates = findall(f -> target - width <= f <= target + width, freqs)
        isempty(candidates) && continue

        amps[h] = maximum(mag[candidates])
    end

    return amps
end

function linear_resample(x, sr_in, sr_out)
    y = as_float_vector(x)

    sr_in == sr_out && return y
    isempty(y) && return y

    ratio = Float64(sr_out) / Float64(sr_in)
    n_out = max(1, round(Int, length(y) * ratio))
    out = zeros(Float64, n_out)

    @inbounds for i in 1:n_out
        pos = 1.0 + (i - 1) / ratio
        j = floor(Int, pos)
        a = pos - j

        if j < 1
            out[i] = y[1]
        elseif j >= length(y)
            out[i] = y[end]
        else
            out[i] = (1.0 - a) * y[j] + a * y[j + 1]
        end
    end

    return out
end




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
    x = as_float_vector(frame)
    isempty(x) && return (frequency_hz=0.0, confidence=0.0)

    x = x .- mean(x)
    n = length(x)

    if n < 3 || maximum(abs, x) < 1e-8
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
    x = as_float_vector(frame)
    isempty(x) && return zeros(Float64, ncoeffs)

    fft_size = isnothing(n_fft) ? length(x) : Int(n_fft)
    fft_size = max(fft_size, length(x))

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
    isempty(freqs) && return 0.0

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
    loudness = [p[2]^2 for p in peaks]

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
        f1 <= 50.0 && continue

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

function sharpness_acum(frame, sr; n_fft=nothing, window=:hann, eps=1e-12)
    # Lightweight sharpness proxy, not a calibrated DIN 45692 implementation.
    # It returns a relative high-Bark weighted specific-energy centroid.
    freqs, power = power_spectrum(frame, sr; n_fft=n_fft, window=window)

    total = sum(power)
    total <= eps && return 0.0

    weighted = 0.0

    @inbounds for i in eachindex(freqs)
        f = freqs[i]
        f <= 0.0 && continue

        z = hz_to_bark(f)
        g = z <= 15.8 ? 1.0 : 0.15 * exp(0.42 * (z - 15.8)) + 0.85
        weighted += power[i] * z * g
    end

    return weighted / total
end


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

function estimate_f0_for_harmonic_features(chunk, sr, config)
    kw = build_feature_kwargs(config, "yin_pitch"; include_fft=false)
    pitch = yin_pitch(chunk, sr; kw...)

    return pitch.frequency_hz, pitch.confidence
end

function extract_features!(row, chunk, previous_chunk, sr, features, config)
    cached_f0 = nothing
    cached_pitch_confidence = nothing

    function get_f0()
        if isnothing(cached_f0)
            f0, conf = estimate_f0_for_harmonic_features(chunk, sr, config)
            cached_f0 = f0
            cached_pitch_confidence = conf
        end

        return cached_f0, cached_pitch_confidence
    end

    for feature in features
        feature = String(feature)

        if feature == "loudness_db"
            kw = build_feature_kwargs(config, feature; include_fft=false)
            value = loudness_db(chunk; kw...)
            add_scalar!(row, "loudness_db", value)

        elseif feature == "peak_amplitude"
            value = peak_amplitude(chunk)
            add_scalar!(row, "peak_amplitude", value)

        elseif feature == "zero_crossing_rate"
            value = zero_crossing_rate(chunk)
            add_scalar!(row, "zero_crossing_rate", value)

        elseif feature == "temporal_centroid_s"
            kw = build_feature_kwargs(config, feature; include_fft=false)
            value = temporal_centroid_s(chunk, sr; kw...)
            add_scalar!(row, "temporal_centroid_s", value)

        elseif feature == "spectral_centroid_hz"
            kw = build_feature_kwargs(config, feature)
            value = spectral_centroid_hz(chunk, sr; kw...)
            add_scalar!(row, "spectral_centroid_hz", value)

        elseif feature == "spectral_spread_hz"
            kw = build_feature_kwargs(config, feature)
            value = spectral_spread_hz(chunk, sr; kw...)
            add_scalar!(row, "spectral_spread_hz", value)

        elseif feature == "spectral_rolloff_hz"
            kw = build_feature_kwargs(config, feature)
            value = spectral_rolloff_hz(chunk, sr; kw...)
            add_scalar!(row, "spectral_rolloff_hz", value)

        elseif feature == "spectral_flatness"
            kw = build_feature_kwargs(config, feature)
            value = spectral_flatness(chunk, sr; kw...)
            add_scalar!(row, "spectral_flatness", value)

        elseif feature == "spectral_flux"
            kw = build_feature_kwargs(config, feature)
            value = spectral_flux(chunk, previous_chunk, sr; kw...)
            add_scalar!(row, "spectral_flux", value)

        elseif feature == "brightness"
            kw = build_feature_kwargs(config, feature)
            value = brightness(chunk, sr; kw...)
            add_scalar!(row, "brightness", value)

        elseif feature == "yin_pitch"
            kw = build_feature_kwargs(config, feature; include_fft=false)
            pitch = yin_pitch(chunk, sr; kw...)
            add_scalar!(row, "yin_pitch_hz", pitch.frequency_hz)
            add_scalar!(row, "yin_pitch_confidence", pitch.confidence)

        elseif feature == "harmonicity"
            kw = build_feature_kwargs(config, feature; include_fft=false)
            value = harmonicity(chunk, sr; kw...)
            add_scalar!(row, "harmonicity", value)

        elseif feature == "band_energy_ratios"
            kw = build_feature_kwargs(config, feature)
            bands = pop!(kw, :bands, [[20.0, 250.0], [250.0, 1000.0], [1000.0, 4000.0], [4000.0, Inf]])
            add_vector!(row, "band_energy_ratio", band_energy_ratios(chunk, sr, bands; kw...))

        elseif feature == "mfcc"
            kw = build_feature_kwargs(config, feature)
            values = mfcc(chunk, sr; kw...)
            add_vector!(row, "mfcc", values)

        elseif feature == "chroma"
            kw = build_feature_kwargs(config, feature)
            values = chroma(chunk, sr; kw...)
            add_vector!(row, "chroma", values)

        elseif feature == "inharmonicity"
            f0, _ = get_f0()
            kw = build_feature_kwargs(config, feature)
            value = inharmonicity(chunk, sr, f0; kw...)
            add_scalar!(row, "inharmonicity", value)

        elseif feature == "tristimulus"
            f0, _ = get_f0()
            kw = build_feature_kwargs(config, feature)
            values = tristimulus(chunk, sr, f0; kw...)
            add_tuple3!(row, "tristimulus", values)

        elseif feature == "odd_even_ratio"
            f0, _ = get_f0()
            kw = build_feature_kwargs(config, feature)
            value = odd_even_ratio(chunk, sr, f0; kw...)
            add_scalar!(row, "odd_even_ratio", value)

        elseif feature == "roughness"
            kw = build_feature_kwargs(config, feature)
            value = roughness(chunk, sr; kw...)
            add_scalar!(row, "roughness", value)

        elseif feature == "sharpness_acum"
            kw = build_feature_kwargs(config, feature)
            value = sharpness_acum(chunk, sr; kw...)
            add_scalar!(row, "sharpness_acum", value)

        else
            error("Unknown feature: $feature")
        end
    end

    return row
end

function extract_features_prefixed!(row, prefix::String, chunk, previous_chunk, sr, features, config)
    tmp = Dict{Symbol,Any}()
    extract_features!(tmp, chunk, previous_chunk, sr, features, config)

    for (k, v) in tmp
        row[Symbol("$(prefix)_$(String(k))")] = v
    end

    return row
end
