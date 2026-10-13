using Statistics
using LinearAlgebra
using FFTW
using DSP

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
