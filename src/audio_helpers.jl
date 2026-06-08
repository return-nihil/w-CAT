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

    bin_points = floor.(Int, (n_fft + 1) .* hz_points ./ Float64(sr)) .+ 1
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


function dct_ii(x, ncoeffs)
    n = length(x)
    out = zeros(Float64, ncoeffs)

    for k in 1:ncoeffs
        s = 0.0

        @inbounds for i in 1:n
            s += x[i] * cos(pi * (k - 1) * (2 * (i - 1) + 1) / (2 * n))
        end

        out[k] = s
    end

    return out
end


# SPECTRAL PEAKS

function spectral_peaks(frame, sr; min_freq=20.0, max_peaks=30, min_prominence=0.01, n_fft=nothing, window=:hann, eps=1e-12)
    freqs, mag = magnitude_spectrum(frame, sr; n_fft=n_fft, window=window)

    peaks = Vector{Tuple{Float64,Float64}}()

    length(mag) < 3 && return peaks

    max_mag = maximum(mag) + eps

    @inbounds for i in 2:(length(mag) - 1)
        if freqs[i] >= min_freq &&
           mag[i] > mag[i - 1] &&
           mag[i] >= mag[i + 1] &&
           mag[i] >= min_prominence * max_mag

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