using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace FastCompute.ImageProcessing;

/// <summary>Provides deterministic image resampling operations.</summary>
public static class ImageResampler
{
    /// <summary>Resizes a floating-point grayscale image using bilinear interpolation.</summary>
    public static Image<GrayF32> Resize(
        this Image<GrayF32> source,
        int width,
        int height,
        CancellationToken cancellationToken = default,
        ComputeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        float[] resized = Resize(
            MemoryMarshal.Cast<GrayF32, float>(source.Pixels.Span),
            source.Width,
            source.Height,
            width,
            height,
            cancellationToken,
            options);
        return Image<GrayF32>.Load(MemoryMarshal.Cast<float, GrayF32>(resized).ToArray(), width, height, source.Encoding);
    }

    /// <summary>Resizes a row-major floating-point buffer using bilinear interpolation.</summary>
    public static float[] Resize(
        ReadOnlySpan<float> source,
        int sourceWidth,
        int sourceHeight,
        int destinationWidth,
        int destinationHeight,
        CancellationToken cancellationToken = default,
        ComputeOptions? options = null)
    {
        ValidateResize(source, sourceWidth, sourceHeight, destinationWidth, destinationHeight);
        ComputeOptions effective = options ?? ComputeOptions.Default;
        if (sourceWidth == destinationWidth && sourceHeight == destinationHeight) return source[..checked(sourceWidth * sourceHeight)].ToArray();
        float[]? gpu = ImageGpuExecutor.TryResize(source[..checked(sourceWidth * sourceHeight)], sourceWidth, sourceHeight, destinationWidth, destinationHeight, effective, cancellationToken);
        if (gpu is not null) return gpu;
        float[] input = source[..checked(sourceWidth * sourceHeight)].ToArray();
        var output = GC.AllocateUninitializedArray<float>(checked(destinationWidth * destinationHeight));
        ComputeBackendKind backend = ResolveResizeBackend(effective, output.Length);
        if (backend == ComputeBackendKind.ParallelCpu)
        {
            Parallel.For(0, destinationHeight, new ParallelOptions
            {
                CancellationToken = cancellationToken,
                MaxDegreeOfParallelism = effective.MaxDegreeOfParallelism ?? -1
            }, y => ResizeRow(input, output, sourceWidth, sourceHeight, destinationWidth, destinationHeight, y, simd: false));
        }
        else
        {
            bool simd = backend == ComputeBackendKind.Simd;
            for (int y = 0; y < destinationHeight; y++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                ResizeRow(input, output, sourceWidth, sourceHeight, destinationWidth, destinationHeight, y, simd);
            }
        }
        return output;
    }

    /// <summary>Downsamples floating-point grayscale using deterministic area averaging.</summary>
    public static Image<GrayF32> Downsample(
        this Image<GrayF32> source,
        float scale,
        CancellationToken cancellationToken = default,
        ComputeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (!float.IsFinite(scale) || scale <= 0f || scale > 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(scale), "Scale must be finite and in the (0, 1] range.");
        }

        int width = Math.Max(1, (int)MathF.Round(source.Width * scale));
        int height = Math.Max(1, (int)MathF.Round(source.Height * scale));
        return Downsample(source, width, height, cancellationToken, options);
    }

    /// <summary>Downsamples floating-point grayscale to explicit dimensions using area averaging.</summary>
    public static Image<GrayF32> Downsample(
        this Image<GrayF32> source,
        int width,
        int height,
        CancellationToken cancellationToken = default,
        ComputeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        var output = GC.AllocateUninitializedArray<GrayF32>(checked(width * height));
        Downsample(
            MemoryMarshal.Cast<GrayF32, float>(source.Pixels.Span),
            MemoryMarshal.Cast<GrayF32, float>(output),
            source.Width,
            source.Height,
            width,
            height,
            cancellationToken,
            options);
        return Image<GrayF32>.Load(output, width, height, source.Encoding);
    }

    /// <summary>
    /// Downsamples contiguous floating-point pixels. Exact 2× and 4× reductions
    /// use dedicated SIMD kernels; other ratios use deterministic non-overlapping
    /// area bins.
    /// </summary>
    public static void Downsample(
        ReadOnlySpan<float> source,
        Span<float> destination,
        int sourceWidth,
        int sourceHeight,
        int destinationWidth,
        int destinationHeight,
        CancellationToken cancellationToken = default,
        ComputeOptions? options = null)
    {
        Validate(source, destination, sourceWidth, sourceHeight, destinationWidth, destinationHeight);
        if (source.Overlaps(destination))
        {
            throw new ArgumentException("Source and destination buffers must not overlap.", nameof(destination));
        }

        if (sourceWidth == destinationWidth && sourceHeight == destinationHeight)
        {
            source.CopyTo(destination);
            return;
        }

        ComputeOptions effectiveOptions = options ?? ComputeOptions.Default;
        float[]? gpuResult = ImageGpuExecutor.TryDownsample(
            source[..checked(sourceWidth * sourceHeight)],
            sourceWidth,
            sourceHeight,
            destinationWidth,
            destinationHeight,
            effectiveOptions,
            cancellationToken);
        if (gpuResult is not null)
        {
            gpuResult.CopyTo(destination);
            return;
        }
        ComputeBackendKind backend = ResolveResizeBackend(effectiveOptions, checked(destinationWidth * destinationHeight));
        if (backend == ComputeBackendKind.Scalar)
        {
            DownsampleAreaBinsScalar(source, destination, sourceWidth, sourceHeight, destinationWidth, destinationHeight, cancellationToken);
            return;
        }
        if (backend == ComputeBackendKind.ParallelCpu)
        {
            DownsampleAreaBinsParallel(source, destination, sourceWidth, sourceHeight, destinationWidth, destinationHeight, cancellationToken, effectiveOptions.MaxDegreeOfParallelism);
            return;
        }
        if (sourceWidth == destinationWidth * 2 && sourceHeight == destinationHeight * 2)
        {
            DownsampleHalf(source, destination, sourceWidth, sourceHeight, cancellationToken);
            return;
        }
        if (sourceWidth == destinationWidth * 4 && sourceHeight == destinationHeight * 4)
        {
            int halfWidth = sourceWidth / 2;
            int halfHeight = sourceHeight / 2;
            float[] temporary = ArrayPool<float>.Shared.Rent(checked(halfWidth * halfHeight));
            try
            {
                Span<float> half = temporary.AsSpan(0, halfWidth * halfHeight);
                DownsampleHalf(source, half, sourceWidth, sourceHeight, cancellationToken);
                DownsampleHalf(half, destination, halfWidth, halfHeight, cancellationToken);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(temporary);
            }
            return;
        }

        DownsampleAreaBins(source, destination, sourceWidth, sourceHeight, destinationWidth, destinationHeight, cancellationToken);
    }

    private static void DownsampleHalf(ReadOnlySpan<float> source, Span<float> destination, int sourceWidth, int sourceHeight, CancellationToken cancellationToken)
    {
        int destinationWidth = sourceWidth / 2;
        int destinationHeight = sourceHeight / 2;
        for (int y = 0; y < destinationHeight; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ReadOnlySpan<float> top = source.Slice((y * 2) * sourceWidth, sourceWidth);
            ReadOnlySpan<float> bottom = source.Slice(((y * 2) + 1) * sourceWidth, sourceWidth);
            Span<float> output = destination.Slice(y * destinationWidth, destinationWidth);
            int sourceX = 0;
            int destinationX = 0;
            if (Sse3.IsSupported)
            {
                Vector128<float> scale = Vector128.Create(0.25f);
                ref float topReference = ref MemoryMarshal.GetReference(top);
                ref float bottomReference = ref MemoryMarshal.GetReference(bottom);
                for (; sourceX <= sourceWidth - 8; sourceX += 8, destinationX += 4)
                {
                    Vector128<float> first = Sse.Add(
                        Vector128.LoadUnsafe(ref topReference, (nuint)sourceX),
                        Vector128.LoadUnsafe(ref bottomReference, (nuint)sourceX));
                    Vector128<float> second = Sse.Add(
                        Vector128.LoadUnsafe(ref topReference, (nuint)(sourceX + 4)),
                        Vector128.LoadUnsafe(ref bottomReference, (nuint)(sourceX + 4)));
                    Vector128<float> firstPairs = Sse3.HorizontalAdd(first, first);
                    Vector128<float> secondPairs = Sse3.HorizontalAdd(second, second);
                    Vector128<float> packed = Sse.Shuffle(firstPairs, secondPairs, 0x44);
                    Sse.Multiply(packed, scale).CopyTo(output.Slice(destinationX, 4));
                }
            }
            for (; destinationX < destinationWidth; destinationX++, sourceX += 2)
                output[destinationX] = (top[sourceX] + top[sourceX + 1] + bottom[sourceX] + bottom[sourceX + 1]) * 0.25f;
        }
    }

    private static void DownsampleAreaBins(ReadOnlySpan<float> source, Span<float> destination, int sourceWidth, int sourceHeight, int destinationWidth, int destinationHeight, CancellationToken cancellationToken)
    {
        for (int destinationY = 0; destinationY < destinationHeight; destinationY++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int sourceY0 = destinationY * sourceHeight / destinationHeight;
            int sourceY1 = Math.Max(sourceY0 + 1, (destinationY + 1) * sourceHeight / destinationHeight);
            for (int destinationX = 0; destinationX < destinationWidth; destinationX++)
            {
                int sourceX0 = destinationX * sourceWidth / destinationWidth;
                int sourceX1 = Math.Max(sourceX0 + 1, (destinationX + 1) * sourceWidth / destinationWidth);
                var vectorSum = Vector<float>.Zero;
                float scalarSum = 0f;
                int count = 0;
                for (int sourceY = sourceY0; sourceY < sourceY1; sourceY++)
                {
                    int offset = sourceY * sourceWidth;
                    int sourceX = sourceX0;
                    int vectorizedEnd = sourceX1 - ((sourceX1 - sourceX0) % Vector<float>.Count);
                    for (; sourceX < vectorizedEnd; sourceX += Vector<float>.Count)
                        vectorSum += new Vector<float>(source.Slice(offset + sourceX, Vector<float>.Count));
                    for (; sourceX < sourceX1; sourceX++) scalarSum += source[offset + sourceX];
                    count += sourceX1 - sourceX0;
                }
                destination[(destinationY * destinationWidth) + destinationX] =
                    (Vector.Sum(vectorSum) + scalarSum) / count;
            }
        }
    }

    private static void DownsampleAreaBinsScalar(ReadOnlySpan<float> source, Span<float> destination, int sourceWidth, int sourceHeight, int destinationWidth, int destinationHeight, CancellationToken token)
    {
        for (int destinationY = 0; destinationY < destinationHeight; destinationY++)
        {
            token.ThrowIfCancellationRequested();
            int sourceY0 = destinationY * sourceHeight / destinationHeight;
            int sourceY1 = Math.Max(sourceY0 + 1, (destinationY + 1) * sourceHeight / destinationHeight);
            for (int destinationX = 0; destinationX < destinationWidth; destinationX++)
            {
                int sourceX0 = destinationX * sourceWidth / destinationWidth;
                int sourceX1 = Math.Max(sourceX0 + 1, (destinationX + 1) * sourceWidth / destinationWidth);
                double sum = 0d;
                int count = 0;
                for (int sourceY = sourceY0; sourceY < sourceY1; sourceY++)
                for (int sourceX = sourceX0; sourceX < sourceX1; sourceX++)
                {
                    sum += source[(sourceY * sourceWidth) + sourceX];
                    count++;
                }
                destination[(destinationY * destinationWidth) + destinationX] = (float)(sum / count);
            }
        }
    }

    private static void DownsampleAreaBinsParallel(ReadOnlySpan<float> source, Span<float> destination, int sourceWidth, int sourceHeight, int destinationWidth, int destinationHeight, CancellationToken token, int? maxDegree)
    {
        float[] input = source[..checked(sourceWidth * sourceHeight)].ToArray();
        var output = GC.AllocateUninitializedArray<float>(checked(destinationWidth * destinationHeight));
        Parallel.For(0, destinationHeight, new ParallelOptions
        {
            CancellationToken = token,
            MaxDegreeOfParallelism = maxDegree ?? -1
        }, destinationY =>
        {
            int sourceY0 = destinationY * sourceHeight / destinationHeight;
            int sourceY1 = Math.Max(sourceY0 + 1, (destinationY + 1) * sourceHeight / destinationHeight);
            for (int destinationX = 0; destinationX < destinationWidth; destinationX++)
            {
                int sourceX0 = destinationX * sourceWidth / destinationWidth;
                int sourceX1 = Math.Max(sourceX0 + 1, (destinationX + 1) * sourceWidth / destinationWidth);
                double sum = 0d;
                int count = 0;
                for (int sourceY = sourceY0; sourceY < sourceY1; sourceY++)
                for (int sourceX = sourceX0; sourceX < sourceX1; sourceX++)
                {
                    sum += input[(sourceY * sourceWidth) + sourceX];
                    count++;
                }
                output[(destinationY * destinationWidth) + destinationX] = (float)(sum / count);
            }
        });
        output.CopyTo(destination);
    }

    private static void Validate(ReadOnlySpan<float> source, Span<float> destination, int sourceWidth, int sourceHeight, int destinationWidth, int destinationHeight)
    {
        if (sourceWidth <= 0) throw new ArgumentOutOfRangeException(nameof(sourceWidth));
        if (sourceHeight <= 0) throw new ArgumentOutOfRangeException(nameof(sourceHeight));
        if (destinationWidth <= 0 || destinationWidth > sourceWidth) throw new ArgumentOutOfRangeException(nameof(destinationWidth));
        if (destinationHeight <= 0 || destinationHeight > sourceHeight) throw new ArgumentOutOfRangeException(nameof(destinationHeight));
        if (source.Length < checked(sourceWidth * sourceHeight)) throw new ArgumentException("Source is shorter than its dimensions.", nameof(source));
        if (destination.Length < checked(destinationWidth * destinationHeight)) throw new ArgumentException("Destination is shorter than its dimensions.", nameof(destination));
    }

    private static void ResizeRow(float[] source, float[] destination, int sourceWidth, int sourceHeight, int destinationWidth, int destinationHeight, int destinationY, bool simd)
    {
        float sourceY = destinationHeight == 1 ? 0f : destinationY * (sourceHeight - 1f) / (destinationHeight - 1f);
        int y0 = (int)MathF.Floor(sourceY);
        int y1 = Math.Min(sourceHeight - 1, y0 + 1);
        float vertical = sourceY - y0;
        int x = 0;
        if (simd)
        {
            int lanes = Vector<float>.Count;
            Span<float> topValues = stackalloc float[lanes];
            Span<float> bottomValues = stackalloc float[lanes];
            int vectorEnd = destinationWidth - (destinationWidth % lanes);
            var verticalVector = new Vector<float>(vertical);
            for (; x < vectorEnd; x += lanes)
            {
                for (int lane = 0; lane < lanes; lane++)
                {
                    float sourceX = destinationWidth == 1 ? 0f : (x + lane) * (sourceWidth - 1f) / (destinationWidth - 1f);
                    int x0 = (int)MathF.Floor(sourceX);
                    int x1 = Math.Min(sourceWidth - 1, x0 + 1);
                    float horizontal = sourceX - x0;
                    topValues[lane] = source[(y0 * sourceWidth) + x0] + ((source[(y0 * sourceWidth) + x1] - source[(y0 * sourceWidth) + x0]) * horizontal);
                    bottomValues[lane] = source[(y1 * sourceWidth) + x0] + ((source[(y1 * sourceWidth) + x1] - source[(y1 * sourceWidth) + x0]) * horizontal);
                }
                Vector<float> top = new(topValues);
                Vector<float> bottom = new(bottomValues);
                (top + ((bottom - top) * verticalVector)).CopyTo(destination, (destinationY * destinationWidth) + x);
            }
        }
        for (; x < destinationWidth; x++)
        {
            float sourceX = destinationWidth == 1 ? 0f : x * (sourceWidth - 1f) / (destinationWidth - 1f);
            int x0 = (int)MathF.Floor(sourceX);
            int x1 = Math.Min(sourceWidth - 1, x0 + 1);
            float horizontal = sourceX - x0;
            float top = source[(y0 * sourceWidth) + x0] + ((source[(y0 * sourceWidth) + x1] - source[(y0 * sourceWidth) + x0]) * horizontal);
            float bottom = source[(y1 * sourceWidth) + x0] + ((source[(y1 * sourceWidth) + x1] - source[(y1 * sourceWidth) + x0]) * horizontal);
            destination[(destinationY * destinationWidth) + x] = top + ((bottom - top) * vertical);
        }
    }

    private static ComputeBackendKind ResolveResizeBackend(ComputeOptions options, int length)
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

    private static void ValidateResize(ReadOnlySpan<float> source, int sourceWidth, int sourceHeight, int destinationWidth, int destinationHeight)
    {
        if (sourceWidth <= 0) throw new ArgumentOutOfRangeException(nameof(sourceWidth));
        if (sourceHeight <= 0) throw new ArgumentOutOfRangeException(nameof(sourceHeight));
        if (destinationWidth <= 0) throw new ArgumentOutOfRangeException(nameof(destinationWidth));
        if (destinationHeight <= 0) throw new ArgumentOutOfRangeException(nameof(destinationHeight));
        if (source.Length < checked(sourceWidth * sourceHeight)) throw new ArgumentException("Source is shorter than its dimensions.", nameof(source));
    }
}
