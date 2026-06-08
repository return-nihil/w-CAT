using Statistics
using LinearAlgebra
using FFTW
using DSP


# AUDIO 

function to_mono(x)
    if ndims(x) == 1
        return Float64.(x)
    elseif ndims(x) == 2
        return vec(mean(Float64.(x), dims=2))
    end
end


function normalize_audio(x)
    m = maximum(abs, x)
    return m > 0 ? Float64.(x) ./ m : Float64.(x)
end


function frame_signal(x, frame_size, hop_size)
    frames = Vector{Vector{Float64}}()
    i = 1

    while i + frame_size - 1 <= length(x)
        push!(frames, Float64.(x[i:i + frame_size - 1]))
        i += hop_size
    end

    return frames
end


function as_float_vector(x)
    return Float64.(x)
end


# WINDOWING

function gaussian_window(n; σ=0.4)
    if n == 1 # <--- check, forse inutile
        return ones(Float64, 1)
    end

    center = (n - 1) / 2
    scale = σ * center

    return [exp(-0.5 * ((i - 1 - center) / scale)^2) for i in 1:n]
end


function trapezoid_window(n; base=0.5)
    if n == 1 # <--- check, forse inutile
        return ones(Float64, 1)
    end

    w = zeros(Float64, n)

    base_samples = round(Int, base * n)
    ramp_total = n - base_samples
    ramp_len = ramp_total ÷ 2

    if ramp_len <= 0 # questo é un rettangolare
        return ones(Float64, n)
    end

    left_end = ramp_len
    right_start = n - ramp_len + 1

    @inbounds for i in 1:n
        if i <= left_end
            w[i] = (i - 1) / ramp_len
        elseif i >= right_start
            w[i] = (n - i) / ramp_len
        else
            w[i] = 1.0
        end
    end

    return w
end


function analysis_window(n; window=:hann, gauss_σ=0.4, trapez_base=0.5)
    if window == :hann
        return DSP.Windows.hann(n)
    elseif window == :hamming
        return DSP.Windows.hamming(n)
    elseif window == :rect
        return ones(Float64, n)
    elseif window == :gaussian
        return gaussian_window(n; σ=gauss_σ)
    elseif window == :trapezoid
        return trapezoid_window(n; base=trapez_base)
    end
end


# SPECTRAL ANALYSIS

function magnitude_spectrum(frame, sr; n_fft=nothing, window=:hann)
    x = as_float_vector(frame)
    w = analysis_window(length(x); window=window)
    xw = x .* w

    fft_size = isnothing(n_fft) ? length(x) : Int(n_fft)
    if fft_size < length(xw)
        buffer = xw[1:fft_size]
    elseif fft_size > length(xw)
        buffer = zeros(Float64, fft_size)
        buffer[1:length(xw)] .= xw
    else
        buffer = xw
    end

    mag = abs.(rfft(buffer))
    freqs = collect(0:(length(mag) - 1)) .* (Float64(sr) / fft_size)

    return freqs, mag
end


function power_spectrum(frame, sr; n_fft=nothing, window=:hann)
    freqs, mag = magnitude_spectrum(frame, sr; n_fft=n_fft, window=window)
    return freqs, mag .^ 2
end


# AUTOCORRELATION

function autocorrelation(frame; eps=1e-12)
    x = as_float_vector(frame)
    x = x .- mean(x)

    if isempty(x) || maximum(abs, x) < 1e-8
        return zeros(Float64, length(x))
    end

    n = length(x)
    fft_size = nextpow(2, 2n - 1)

    buffer = zeros(Float64, fft_size)
    buffer[1:n] .= x

    spectrum = fft(buffer)
    ac = real.(ifft(abs2.(spectrum)))[1:n]
    ac ./= maximum(abs, ac) + eps

    return ac
end


# MEL SCALE

function hz_to_mel(f::Real)
    return 2595.0 * log10(1.0 + f / 700.0)
end


function mel_to_hz(m::Real)
    return 700.0 * (10.0 ^ (m / 2595.0) - 1.0)
end


function mel_filterbank(n_fft, sr; nbands=24, fmin=0.0, fmax=nothing)
    n_bins = n_fft ÷ 2 + 1
    high_hz = isnothing(fmax) ? Float64(sr) / 2.0 : Float64(fmax)

    mel_points = collect(range(hz_to_mel(fmin), hz_to_mel(high_hz), length=nbands + 2))
    hz_points = mel_to_hz.(mel_points)

    bin_points = floor.(Int, (n_fft + 1) .* hz_points ./ Float64(sr)) .+ 1 # 1-based Julia indices!!
    bin_points = clamp.(bin_points, 1, n_bins)

    filters = zeros(Float64, nbands, n_bins)

    for band in 1:nbands
        left = bin_points[band]
        center = bin_points[band + 1]
        right = bin_points[band + 2]

        if center > left
            for k in left:center
                filters[band, k] = (k - left) / (center - left)
            end
        end

        if right > center
            for k in center:right
                filters[band, k] = max(filters[band, k], (right - k) / (right - center))
            end
        end
    end

    return filters
end


function dct_ii(x, n_coeffs)
    # Implementazione DCT-II semplificata

    y = as_float_vector(x)
    n = length(y)

    out = zeros(Float64, n_coeffs)

    for k in 1:n_coeffs
        s = 0.0

        @inbounds for i in 1:n
            s += y[i] * cos(pi * (k - 1) * (2 * (i - 1) + 1) / (2 * n))
        end

        out[k] = s
    end

    return out
end


# SPECTRAL PEAKS

function spectral_peaks(frame, sr; min_freq=20.0, max_peaks=30, min_prom=0.01, n_fft=nothing, window=:hann, eps=1e-12)
    freqs, mag = magnitude_spectrum(frame, sr; n_fft=n_fft, window=window)

    peaks = Vector{Tuple{Float64,Float64}}()

    length(mag) < 3 && return peaks

    max_mag = maximum(mag) + eps

    @inbounds for i in 2:(length(mag) - 1)
        if freqs[i] >= min_freq &&
           mag[i] > mag[i - 1] &&
           mag[i] >= mag[i + 1] &&
           mag[i] >= min_prom * max_mag

            push!(peaks, (Float64(freqs[i]), Float64(mag[i])))
        end
    end

    sort!(peaks, by=p -> p[2], rev=true)

    return peaks[1:min(max_peaks, length(peaks))]
end


# HARMONIC ANALYSIS

function harmonic_amplitudes(frame, sr, f0; n_harmonics=10, tolerance=0.04, n_fft=nothing, window=:hann)
    freqs, mag = magnitude_spectrum(frame, sr; n_fft=n_fft, window=window)

    values = zeros(Float64, n_harmonics)

    f0 <= 0.0 && return values

    nyquist = maximum(freqs)

    for h in 1:n_harmonics
        target = h * f0
        target > nyquist && break

        width = max(20.0, tolerance * target)

        candidates = findall(f -> target - width <= f <= target + width, freqs)
        isempty(candidates) && continue

        local_mag = mag[candidates]
        _, local_idx = findmax(local_mag)

        values[h] = local_mag[local_idx]
    end

    return values
end


# BARK 
# cose varie per algoritmo roughness su stile Dissonance di Essentia:
# https://essentia.upf.edu/reference/std_Dissonance.html
function a_weighting(f)
        f2 = f * f
        return 1.25893 * 12200.0^2 * f2^2 /
               ((f2 + 20.6^2) *
                (f2 + 12200.0^2) *
                sqrt(f2 + 107.7^2) *
                sqrt(f2 + 737.9^2))
    end


function hz_to_bark(f)
    b = (26.81 * f) / (1960.0 + f) - 0.53

    if b < 2.0
        b += 0.15 * (2.0 - b)
    elseif b > 20.1
        b += 0.22 * (b - 20.1)
    end

    return b
end


function bark_to_hz(z)
    if z < 2.0
        z = (z - 0.3) / 0.85
    elseif z > 20.1
        z = (z - 4.422) / 1.22
    end

    return 1960.0 * (z + 0.53) / (26.28 - z)
end


function bark_critical_bandwidth(z)
    return 52548.0 / (z^2 - 52.56 * z + 690.39)
end


function plomp_levelt_consonance(df_cbw)
    if df_cbw < 0.0
        return 1.0
    elseif df_cbw > 1.18
        return 1.0
    end

    y =
        -6.58977878 * df_cbw^5 +
        28.58224226 * df_cbw^4 -
        47.36739986 * df_cbw^3 +
        35.70679761 * df_cbw^2 -
        10.36526344 * df_cbw +
        1.00026609

    return clamp(y, 0.0, 1.0)
end


function pair_consonance(f1, f2)
    cbw1 = bark_critical_bandwidth(hz_to_bark(f1))
    cbw2 = bark_critical_bandwidth(hz_to_bark(f2))
    cbw = min(cbw1, cbw2)

    return plomp_levelt_consonance(abs(f2 - f1) / cbw)
end