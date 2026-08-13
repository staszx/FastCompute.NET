namespace FastCompute;

public static partial class Compute
{
    /// <summary>Calculates squared magnitude for a complex spectrum.</summary>
    public static float[] PowerSpectrum(Complex32[] spectrum, ComputeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(spectrum);
        return spectrum
            .AsCompute(options ?? ComputeOptions.Default)
            .Select(value => (value.Real * value.Real) + (value.Imaginary * value.Imaginary))
            .ToArray();
    }

    /// <summary>Calculates magnitude for a complex spectrum.</summary>
    public static float[] MagnitudeSpectrum(Complex32[] spectrum, ComputeOptions? options = null)
    {
        float[] power = PowerSpectrum(spectrum, options);
        return RunInPlace(power, value => ComputeMath.Sqrt(value), options);
    }

    /// <summary>Calculates phase angles in radians for a complex spectrum.</summary>
    public static float[] PhaseSpectrum(Complex32[] spectrum, ComputeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(spectrum);
        if (spectrum.Length == 0) return [];
        ComputeOptions effective = options ?? ComputeOptions.Default;
        if (effective.Backend == ComputeBackendKind.Simd)
            throw new ComputeBackendNotSupportedException(ComputeBackendKind.Simd, "phase spectrum (Atan2)", "Scalar, ParallelCpu, Gpu, Auto");
        if (effective.Backend == ComputeBackendKind.Gpu ||
            (effective.Backend == ComputeBackendKind.Auto && spectrum.Length >= effective.Thresholds.GpuSimpleThreshold &&
             (effective.GpuContext is not null || Backends.Gpu.GpuComputeBackend.HasHardwareAccelerator)))
            return Backends.Gpu.GpuComputeBackend.ResolveContext(effective).ExecutePhaseSpectrum(spectrum);
        var result = GC.AllocateUninitializedArray<float>(spectrum.Length);
        if (effective.Backend == ComputeBackendKind.ParallelCpu ||
            (effective.Backend == ComputeBackendKind.Auto && spectrum.Length >= effective.Thresholds.ParallelThreshold))
        {
            Parallel.For(0, spectrum.Length, new ParallelOptions
            {
                CancellationToken = effective.CancellationToken,
                MaxDegreeOfParallelism = effective.MaxDegreeOfParallelism ?? -1
            }, index => result[index] = MathF.Atan2(spectrum[index].Imaginary, spectrum[index].Real));
        }
        else
        {
            for (int index = 0; index < spectrum.Length; index++)
            {
                if ((index & 0xFFFF) == 0) effective.CancellationToken.ThrowIfCancellationRequested();
                result[index] = MathF.Atan2(spectrum[index].Imaginary, spectrum[index].Real);
            }
        }
        return result;
    }

    /// <summary>Returns local maxima that meet the minimum value and separation constraints.</summary>
    public static SignalPeak[] FindPeaks(ReadOnlySpan<float> values, float minimumValue = float.NegativeInfinity, int minimumDistance = 1)
    {
        if (minimumDistance <= 0) throw new ArgumentOutOfRangeException(nameof(minimumDistance));
        if (values.Length < 3) return [];
        var candidates = new List<SignalPeak>();
        for (int index = 1; index < values.Length - 1; index++)
        {
            float value = values[index];
            if (value >= minimumValue && value > values[index - 1] && value >= values[index + 1])
                candidates.Add(new SignalPeak(index, value));
        }
        if (minimumDistance == 1 || candidates.Count < 2) return candidates.ToArray();
        var selected = new List<SignalPeak>();
        foreach (SignalPeak candidate in candidates.OrderByDescending(peak => peak.Value))
        {
            if (selected.All(peak => Math.Abs(peak.Index - candidate.Index) >= minimumDistance))
                selected.Add(candidate);
        }
        selected.Sort(static (left, right) => left.Index.CompareTo(right.Index));
        return selected.ToArray();
    }

    /// <summary>Calculates the largest local-peak value divided by the signal median.</summary>
    public static double PeakToMedianRatio(ReadOnlySpan<float> values)
    {
        if (values.IsEmpty) return 0d;
        SignalPeak[] peaks = FindPeaks(values);
        if (peaks.Length == 0) return 0d;
        float[] copy = values.ToArray();
        float median = Median(copy);
        return median > 1e-30f ? peaks.Max(peak => peak.Value) / median : 0d;
    }

    /// <summary>Calculates mean absolute difference between adjacent samples.</summary>
    public static float MeanAbsoluteDifference(ReadOnlySpan<float> values, ComputeOptions? options = null)
    {
        if (values.Length < 2) return 0f;
        float[] left = values[..^1].ToArray();
        float[] right = values[1..].ToArray();
        float[] differences = Zip(left, right, (first, second) => ComputeMath.Abs(second - first), options);
        return (float)Mean(differences, options);
    }

    /// <summary>Returns the requested percentile and reorders the supplied working span.</summary>
    public static float Percentile(Span<float> data, double percentile)
    {
        if (data.IsEmpty) throw new ArgumentException("Percentile is not defined for an empty sequence.", nameof(data));
        if (!double.IsFinite(percentile) || percentile < 0d || percentile > 100d) throw new ArgumentOutOfRangeException(nameof(percentile));
        data.Sort();
        double position = percentile / 100d * (data.Length - 1);
        int lower = (int)Math.Floor(position);
        int upper = (int)Math.Ceiling(position);
        if (lower == upper) return data[lower];
        float fraction = (float)(position - lower);
        return data[lower] + ((data[upper] - data[lower]) * fraction);
    }

    /// <summary>Returns the requested quantile and reorders the supplied working span.</summary>
    public static float Quantile(Span<float> data, double quantile)
    {
        if (!double.IsFinite(quantile) || quantile < 0d || quantile > 1d) throw new ArgumentOutOfRangeException(nameof(quantile));
        return Percentile(data, quantile * 100d);
    }

    /// <summary>Returns the median and reorders the supplied working span.</summary>
    public static float Median(Span<float> data) => Percentile(data, 50d);

    /// <summary>Returns the requested percentile and reorders the supplied double-precision working span.</summary>
    public static double Percentile(Span<double> data, double percentile)
    {
        if (data.IsEmpty) throw new ArgumentException("Percentile is not defined for an empty sequence.", nameof(data));
        if (!double.IsFinite(percentile) || percentile < 0d || percentile > 100d) throw new ArgumentOutOfRangeException(nameof(percentile));
        data.Sort();
        double position = percentile / 100d * (data.Length - 1);
        int lower = (int)Math.Floor(position);
        int upper = (int)Math.Ceiling(position);
        if (lower == upper) return data[lower];
        return data[lower] + ((data[upper] - data[lower]) * (position - lower));
    }

    /// <summary>Returns the median and reorders the supplied double-precision working span.</summary>
    public static double Median(Span<double> data) => Percentile(data, 50d);

    /// <summary>Applies a Hann window in place using the selected backend for multiplication.</summary>
    public static float[] ApplyHannWindow(float[] values, ComputeOptions? options = null) =>
        ApplyWindow(values, static (index, length) => length <= 1 ? 1f : 0.5f * (1f - MathF.Cos(2f * MathF.PI * index / (length - 1))), options);

    /// <summary>Applies a Hamming window in place using the selected backend for multiplication.</summary>
    public static float[] ApplyHammingWindow(float[] values, ComputeOptions? options = null) =>
        ApplyWindow(values, static (index, length) => length <= 1 ? 1f : 0.54f - (0.46f * MathF.Cos(2f * MathF.PI * index / (length - 1))), options);

    /// <summary>Applies a Blackman window in place using the selected backend for multiplication.</summary>
    public static float[] ApplyBlackmanWindow(float[] values, ComputeOptions? options = null) =>
        ApplyWindow(values, static (index, length) =>
        {
            if (length <= 1) return 1f;
            float phase = 2f * MathF.PI * index / (length - 1);
            return 0.42f - (0.5f * MathF.Cos(phase)) + (0.08f * MathF.Cos(2f * phase));
        }, options);

    /// <summary>Applies a Hann window to an arbitrary span.</summary>
    public static void ApplyHannWindow(Span<float> values)
    {
        if (values.Length <= 1) return;
        for (int index = 0; index < values.Length; index++)
            values[index] *= 0.5f * (1f - MathF.Cos(2f * MathF.PI * index / (values.Length - 1)));
    }

    private static float[] ApplyWindow(float[] values, Func<int, int, float> createWeight, ComputeOptions? options)
    {
        ArgumentNullException.ThrowIfNull(values);
        var weights = GC.AllocateUninitializedArray<float>(values.Length);
        for (int index = 0; index < weights.Length; index++) weights[index] = createWeight(index, weights.Length);
        return ZipInPlace(values, weights, (value, weight) => value * weight, options);
    }
}
