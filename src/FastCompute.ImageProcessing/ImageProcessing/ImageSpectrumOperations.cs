using System.Numerics;

namespace FastCompute.ImageProcessing;

/// <summary>Provides image-domain preparation and radial spectrum measurements.</summary>
public static class ImageSpectrumOperations
{
    /// <summary>Extracts a centered region, removes DC, applies a separable 2D Hann window, and returns complex FFT input.</summary>
    public static Complex32[] PrepareSpectrumInput(
        ReadOnlySpan<float> source,
        int sourceWidth,
        int sourceHeight,
        int width,
        int height,
        ComputeOptions? options = null)
    {
        ValidateDimensions(source, sourceWidth, sourceHeight);
        if (width <= 0 || width > sourceWidth) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0 || height > sourceHeight) throw new ArgumentOutOfRangeException(nameof(height));
        int startX = (sourceWidth - width) / 2;
        int startY = (sourceHeight - height) / 2;
        var prepared = GC.AllocateUninitializedArray<float>(checked(width * height));
        for (int y = 0; y < height; y++)
            source.Slice(((startY + y) * sourceWidth) + startX, width).CopyTo(prepared.AsSpan(y * width, width));

        ComputeOptions effective = options ?? ComputeOptions.Default;
        float mean = (float)Compute.Mean(prepared, effective);
        Compute.RunInPlace(prepared, value => value - mean, effective);
        ApplyWindow2D(prepared, width, height, WindowFunction.Hann, effective);

        var result = GC.AllocateUninitializedArray<Complex32>(prepared.Length);
        Parallel.For(0, prepared.Length, new ParallelOptions
        {
            CancellationToken = effective.CancellationToken,
            MaxDegreeOfParallelism = effective.MaxDegreeOfParallelism ?? -1
        }, index => result[index] = new Complex32(prepared[index], 0f));
        return result;
    }

    /// <summary>Applies a separable two-dimensional signal window in place.</summary>
    public static float[] ApplyWindow2D(
        float[] values,
        int width,
        int height,
        WindowFunction window = WindowFunction.Hann,
        ComputeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(values);
        ValidateDimensions(values, width, height);
        if (!Enum.IsDefined(window)) throw new ArgumentOutOfRangeException(nameof(window));
        float[] xWindow = CreateWindow(width, window);
        float[] yWindow = CreateWindow(height, window);
        var weights = GC.AllocateUninitializedArray<float>(checked(width * height));
        Parallel.For(0, height, y =>
        {
            float vertical = yWindow[y];
            int offset = y * width;
            for (int x = 0; x < width; x++) weights[offset + x] = xWindow[x] * vertical;
        });
        return Compute.ZipInPlace(values, weights, (value, weight) => value * weight, options);
    }

    /// <summary>Calculates normalized radial mean power and returns exact radial band energies.</summary>
    public static float[] CalculateRadialSpectrum(
        ReadOnlySpan<float> powerSpectrum,
        int width,
        int height,
        int binCount,
        out FrequencyBandEnergy bandEnergy,
        float lowBoundary = 0.15f,
        float middleBoundary = 0.5f,
        ComputeOptions? options = null)
    {
        ValidateDimensions(powerSpectrum, width, height);
        if (binCount <= 0) throw new ArgumentOutOfRangeException(nameof(binCount));
        if (!float.IsFinite(lowBoundary) || !float.IsFinite(middleBoundary) || lowBoundary < 0f || middleBoundary <= lowBoundary || middleBoundary > 1f)
            throw new ArgumentOutOfRangeException(nameof(lowBoundary), "Frequency boundaries must satisfy 0 <= low < middle <= 1.");

        ComputeOptions effective = options ?? ComputeOptions.Default;
        (double[] sums, int[] counts, double[] bands) = Accumulate(powerSpectrum, width, height, binCount, lowBoundary, middleBoundary, effective);
        var result = new float[binCount];
        double total = 0d;
        for (int bin = 0; bin < binCount; bin++)
        {
            if (counts[bin] > 0) result[bin] = (float)(sums[bin] / counts[bin]);
            total += result[bin];
        }
        if (total > 1e-30d)
        {
            float inverse = (float)(1d / total);
            Compute.RunInPlace(result, value => value * inverse, effective);
        }
        bandEnergy = new FrequencyBandEnergy(bands[0], bands[1], bands[2]);
        return result;
    }

    /// <summary>Calculates exact energy in low, middle, and high radial image-frequency bands.</summary>
    public static FrequencyBandEnergy CalculateFrequencyBandEnergy(
        ReadOnlySpan<float> powerSpectrum,
        int width,
        int height,
        float lowBoundary = 0.15f,
        float middleBoundary = 0.5f,
        ComputeOptions? options = null)
    {
        _ = CalculateRadialSpectrum(powerSpectrum, width, height, 1, out FrequencyBandEnergy energy, lowBoundary, middleBoundary, options);
        return energy;
    }

    /// <summary>Measures local power peaks against a square neighbourhood mean.</summary>
    public static SpectrumPeakMetrics CalculatePeakMetrics(
        ReadOnlySpan<float> powerSpectrum,
        int width,
        int height,
        float strongPeakThreshold = 8f,
        ComputeOptions? options = null)
    {
        ValidateDimensions(powerSpectrum, width, height);
        if (!float.IsFinite(strongPeakThreshold) || strongPeakThreshold <= 0f)
            throw new ArgumentOutOfRangeException(nameof(strongPeakThreshold));
        float[] input = powerSpectrum[..checked(width * height)].ToArray();
        float[] kernel = Enumerable.Repeat(1f / 24f, 25).ToArray();
        kernel[12] = 0f;
        float[] neighbourhood = Compute.Convolve2D(input, width, height, kernel, 5, 5, options: options);
        float[] ratios = Compute.Zip(input, neighbourhood, (value, localMean) => value / ComputeMath.Max(localMean, 1e-30f), options);
        float maximum = ratios.Length == 0 ? 0f : Compute.Max(ratios, options);
        float[] strong = Compute.Threshold(ratios, strongPeakThreshold, options);
        int count = (int)Compute.Sum(strong, options);
        return new SpectrumPeakMetrics(MathF.Min(maximum, 1000f), count);
    }

    private static (double[] Sums, int[] Counts, double[] Bands) Accumulate(ReadOnlySpan<float> power, int width, int height, int binCount, float lowBoundary, float middleBoundary, ComputeOptions options)
    {
        (float[] Sums, int[] Counts, float[] Bands)? gpu = ImageGpuExecutor.TryRadialSpectrum(power, width, height, binCount, lowBoundary, middleBoundary, options);
        if (gpu is { } gpuResult)
            return (gpuResult.Sums.Select(value => (double)value).ToArray(), gpuResult.Counts, gpuResult.Bands.Select(value => (double)value).ToArray());

        float[] input = power[..checked(width * height)].ToArray();
        ComputeBackendKind backend = ResolveCpuBackend(options, input.Length);
        if (backend == ComputeBackendKind.ParallelCpu)
            return AccumulateParallel(input, width, height, binCount, lowBoundary, middleBoundary, options);
        if (backend == ComputeBackendKind.Simd)
            return AccumulateSimd(input, width, height, binCount, lowBoundary, middleBoundary, options.CancellationToken);
        var accumulator = new SpectrumAccumulator(binCount);
        AccumulateRows(input, width, height, 0, height, lowBoundary, middleBoundary, accumulator, options.CancellationToken);
        return (accumulator.Sums, accumulator.Counts, accumulator.Bands);
    }

    private static (double[] Sums, int[] Counts, double[] Bands) AccumulateParallel(float[] power, int width, int height, int binCount, float lowBoundary, float middleBoundary, ComputeOptions options)
    {
        var total = new SpectrumAccumulator(binCount);
        object gate = new();
        Parallel.For(0, height, new ParallelOptions
        {
            CancellationToken = options.CancellationToken,
            MaxDegreeOfParallelism = options.MaxDegreeOfParallelism ?? -1
        }, () => new SpectrumAccumulator(binCount), (y, _, local) =>
        {
            AccumulateRows(power, width, height, y, y + 1, lowBoundary, middleBoundary, local, options.CancellationToken);
            return local;
        }, local =>
        {
            lock (gate) total.Add(local);
        });
        return (total.Sums, total.Counts, total.Bands);
    }

    private static (double[] Sums, int[] Counts, double[] Bands) AccumulateSimd(float[] power, int width, int height, int binCount, float lowBoundary, float middleBoundary, CancellationToken token)
    {
        var result = new SpectrumAccumulator(binCount);
        float maximum = MathF.Sqrt(((width / 2f) * (width / 2f)) + ((height / 2f) * (height / 2f)));
        float[] xSquared = Enumerable.Range(0, width)
            .Select(x => { int frequency = x <= width / 2 ? x : x - width; return (float)(frequency * frequency); })
            .ToArray();
        int lanes = Vector<float>.Count;
        Span<float> radii = stackalloc float[lanes];
        Span<float> values = stackalloc float[lanes];
        for (int y = 0; y < height; y++)
        {
            token.ThrowIfCancellationRequested();
            int frequencyY = y <= height / 2 ? y : y - height;
            var ySquared = new Vector<float>(frequencyY * frequencyY);
            int x = 0;
            int vectorEnd = width - (width % lanes);
            for (; x < vectorEnd; x += lanes)
            {
                Vector.SquareRoot(new Vector<float>(xSquared, x) + ySquared).CopyTo(radii);
                new Vector<float>(power, (y * width) + x).CopyTo(values);
                for (int lane = 0; lane < lanes; lane++)
                    result.AddSample(values[lane], radii[lane] / maximum, binCount, lowBoundary, middleBoundary);
            }
            for (; x < width; x++)
            {
                float radius = MathF.Sqrt(xSquared[x] + (frequencyY * frequencyY)) / maximum;
                result.AddSample(power[(y * width) + x], radius, binCount, lowBoundary, middleBoundary);
            }
        }
        return (result.Sums, result.Counts, result.Bands);
    }

    private static void AccumulateRows(float[] power, int width, int height, int startY, int endY, float lowBoundary, float middleBoundary, SpectrumAccumulator result, CancellationToken token)
    {
        double maximum = Math.Sqrt(((width / 2d) * (width / 2d)) + ((height / 2d) * (height / 2d)));
        for (int y = startY; y < endY; y++)
        {
            token.ThrowIfCancellationRequested();
            int frequencyY = y <= height / 2 ? y : y - height;
            for (int x = 0; x < width; x++)
            {
                int frequencyX = x <= width / 2 ? x : x - width;
                float radius = (float)(Math.Sqrt((frequencyX * frequencyX) + (frequencyY * frequencyY)) / maximum);
                result.AddSample(power[(y * width) + x], radius, result.Sums.Length, lowBoundary, middleBoundary);
            }
        }
    }

    private static float[] CreateWindow(int length, WindowFunction window)
    {
        var values = new float[length];
        Array.Fill(values, 1f);
        switch (window)
        {
            case WindowFunction.Hann: Compute.ApplyHannWindow(values.AsSpan()); break;
            case WindowFunction.Hamming: Compute.ApplyHammingWindow(values); break;
            case WindowFunction.Blackman: Compute.ApplyBlackmanWindow(values); break;
        }
        return values;
    }

    private static ComputeBackendKind ResolveCpuBackend(ComputeOptions options, int length)
    {
        if (options.Backend == ComputeBackendKind.Gpu) throw new ComputeBackendUnavailableException(ComputeBackendKind.Gpu);
        if (options.Backend == ComputeBackendKind.Simd)
        {
            if (!Vector.IsHardwareAccelerated) throw new ComputeBackendUnavailableException(ComputeBackendKind.Simd);
            return ComputeBackendKind.Simd;
        }
        if (options.Backend != ComputeBackendKind.Auto) return options.Backend;
        if (length >= options.Thresholds.ParallelThreshold) return ComputeBackendKind.ParallelCpu;
        if (length >= options.Thresholds.SimdThreshold && Vector.IsHardwareAccelerated) return ComputeBackendKind.Simd;
        return ComputeBackendKind.Scalar;
    }

    private static void ValidateDimensions(ReadOnlySpan<float> source, int width, int height)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if (source.Length < checked(width * height)) throw new ArgumentException("Source is shorter than the declared dimensions.", nameof(source));
    }

    private sealed class SpectrumAccumulator(int binCount)
    {
        internal double[] Sums { get; } = new double[binCount];
        internal int[] Counts { get; } = new int[binCount];
        internal double[] Bands { get; } = new double[3];

        internal void AddSample(float value, float radius, int bins, float lowBoundary, float middleBoundary)
        {
            int bin = Math.Min(bins - 1, (int)(radius * bins));
            Sums[bin] += value;
            Counts[bin]++;
            Bands[radius < lowBoundary ? 0 : radius < middleBoundary ? 1 : 2] += value;
        }

        internal void Add(SpectrumAccumulator other)
        {
            for (int index = 0; index < Sums.Length; index++)
            {
                Sums[index] += other.Sums[index];
                Counts[index] += other.Counts[index];
            }
            for (int index = 0; index < Bands.Length; index++) Bands[index] += other.Bands[index];
        }
    }
}
