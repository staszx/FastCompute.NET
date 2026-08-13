using System.Numerics;

namespace FastCompute.ImageProcessing;

/// <summary>Provides image-coordinate spatial measurements.</summary>
public static class ImageSpatialOperations
{
    /// <summary>Calculates local range contrast (maximum minus minimum) for every sample.</summary>
    public static float[] LocalContrast(
        ReadOnlySpan<float> source,
        int width,
        int height,
        int radius = 1,
        ComputeOptions? options = null)
    {
        Validate(source, width, height);
        if (radius < 0) throw new ArgumentOutOfRangeException(nameof(radius));
        if (radius == 0) return new float[source.Length];
        ComputeOptions effective = options ?? ComputeOptions.Default;
        float[]? gpu = ImageGpuExecutor.TryLocalContrast(source, width, height, radius, effective);
        if (gpu is not null) return gpu;

        float[] input = source[..checked(width * height)].ToArray();
        var result = GC.AllocateUninitializedArray<float>(input.Length);
        ComputeBackendKind backend = ResolveCpuBackend(effective, input.Length);
        if (backend == ComputeBackendKind.ParallelCpu)
        {
            Parallel.For(0, height, new ParallelOptions
            {
                CancellationToken = effective.CancellationToken,
                MaxDegreeOfParallelism = effective.MaxDegreeOfParallelism ?? -1
            }, y => LocalContrastScalarRow(input, result, width, height, radius, y, 0, width));
        }
        else if (backend == ComputeBackendKind.Simd)
        {
            LocalContrastSimd(input, result, width, height, radius, effective.CancellationToken);
        }
        else
        {
            for (int y = 0; y < height; y++)
            {
                effective.CancellationToken.ThrowIfCancellationRequested();
                LocalContrastScalarRow(input, result, width, height, radius, y, 0, width);
            }
        }
        return result;
    }

    /// <summary>Calculates entropy for normalized grayscale values using equal-width bins.</summary>
    public static double Entropy(ReadOnlySpan<float> source, int binCount = 64, ComputeOptions? options = null)
    {
        if (binCount <= 0) throw new ArgumentOutOfRangeException(nameof(binCount));
        if (source.IsEmpty) return 0d;
        int[] histogram = Compute.Histogram(source.ToArray(), binCount, 0f, 1f, options);
        return Compute.ShannonEntropy(histogram);
    }

    /// <summary>Calculates a local Shannon-entropy map for normalized grayscale values.</summary>
    public static float[] LocalEntropy(
        ReadOnlySpan<float> source,
        int width,
        int height,
        int radius = 1,
        int binCount = 64,
        ComputeOptions? options = null)
    {
        Validate(source, width, height);
        if (radius < 0) throw new ArgumentOutOfRangeException(nameof(radius));
        if (binCount <= 0) throw new ArgumentOutOfRangeException(nameof(binCount));
        ComputeOptions effective = options ?? ComputeOptions.Default;
        if (effective.Backend == ComputeBackendKind.Simd)
            throw new ComputeBackendNotSupportedException(ComputeBackendKind.Simd, "local window entropy", "Scalar, ParallelCpu, Gpu, Auto");
        float[]? gpu = ImageGpuExecutor.TryLocalEntropy(source, width, height, radius, binCount, effective);
        if (gpu is not null) return gpu;
        float[] input = source[..checked(width * height)].ToArray();
        var result = GC.AllocateUninitializedArray<float>(input.Length);
        if (effective.Backend == ComputeBackendKind.ParallelCpu ||
            (effective.Backend == ComputeBackendKind.Auto && input.Length >= effective.Thresholds.ParallelThreshold))
        {
            Parallel.For(0, height, new ParallelOptions
            {
                CancellationToken = effective.CancellationToken,
                MaxDegreeOfParallelism = effective.MaxDegreeOfParallelism ?? -1
            }, y => CalculateEntropyRow(input, result, width, height, radius, binCount, y));
        }
        else
        {
            for (int y = 0; y < height; y++)
            {
                effective.CancellationToken.ThrowIfCancellationRequested();
                CalculateEntropyRow(input, result, width, height, radius, binCount, y);
            }
        }
        return result;
    }

    private static void CalculateEntropyRow(float[] source, float[] destination, int width, int height, int radius, int binCount, int y)
    {
        int startY = Math.Max(0, y - radius);
        int endY = Math.Min(height - 1, y + radius);
        var histogram = new int[binCount];
        for (int x = 0; x < width; x++)
        {
            Array.Clear(histogram);
            int startX = Math.Max(0, x - radius);
            int endX = Math.Min(width - 1, x + radius);
            for (int currentY = startY; currentY <= endY; currentY++)
            for (int currentX = startX; currentX <= endX; currentX++)
            {
                float value = Math.Clamp(source[(currentY * width) + currentX], 0f, 1f);
                histogram[Math.Min(binCount - 1, (int)(value * binCount))]++;
            }
            destination[(y * width) + x] = (float)Compute.ShannonEntropy(histogram);
        }
    }

    private static void LocalContrastSimd(float[] source, float[] destination, int width, int height, int radius, CancellationToken token)
    {
        int lanes = Vector<float>.Count;
        for (int y = 0; y < height; y++)
        {
            token.ThrowIfCancellationRequested();
            LocalContrastScalarRow(source, destination, width, height, radius, y, 0, Math.Min(radius, width));
            int x = radius;
            int interiorEnd = Math.Max(radius, width - radius);
            int vectorEnd = interiorEnd - ((interiorEnd - x) % lanes);
            for (; x < vectorEnd; x += lanes)
            {
                Vector<float> minimum = new(float.MaxValue);
                Vector<float> maximum = new(float.MinValue);
                for (int yy = y - radius; yy <= y + radius; yy++)
                {
                    int sourceY = Math.Clamp(yy, 0, height - 1);
                    for (int xx = -radius; xx <= radius; xx++)
                    {
                        Vector<float> values = new(source, (sourceY * width) + x + xx);
                        minimum = Vector.Min(minimum, values);
                        maximum = Vector.Max(maximum, values);
                    }
                }
                (maximum - minimum).CopyTo(destination, (y * width) + x);
            }
            LocalContrastScalarRow(source, destination, width, height, radius, y, x, width);
        }
    }

    private static void LocalContrastScalarRow(float[] source, float[] destination, int width, int height, int radius, int y, int startX, int endX)
    {
        int startY = Math.Max(0, y - radius);
        int endY = Math.Min(height - 1, y + radius);
        for (int x = startX; x < endX; x++)
        {
            float minimum = float.MaxValue;
            float maximum = float.MinValue;
            int neighbourhoodStartX = Math.Max(0, x - radius);
            int neighbourhoodEndX = Math.Min(width - 1, x + radius);
            for (int currentY = startY; currentY <= endY; currentY++)
            for (int currentX = neighbourhoodStartX; currentX <= neighbourhoodEndX; currentX++)
            {
                float value = source[(currentY * width) + currentX];
                minimum = MathF.Min(minimum, value);
                maximum = MathF.Max(maximum, value);
            }
            destination[(y * width) + x] = maximum - minimum;
        }
    }

    private static ComputeBackendKind ResolveCpuBackend(ComputeOptions options, int length)
    {
        if (options.Backend == ComputeBackendKind.Gpu)
            throw new ComputeBackendUnavailableException(ComputeBackendKind.Gpu);
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

    private static void Validate(ReadOnlySpan<float> source, int width, int height)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if (source.Length < checked(width * height)) throw new ArgumentException("Source is shorter than the image dimensions.", nameof(source));
    }
}
