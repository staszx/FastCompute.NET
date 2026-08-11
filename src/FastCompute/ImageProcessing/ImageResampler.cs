using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace FastCompute.ImageProcessing;

/// <summary>Provides deterministic image resampling operations.</summary>
public static class ImageResampler
{
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

    private static void Validate(ReadOnlySpan<float> source, Span<float> destination, int sourceWidth, int sourceHeight, int destinationWidth, int destinationHeight)
    {
        if (sourceWidth <= 0) throw new ArgumentOutOfRangeException(nameof(sourceWidth));
        if (sourceHeight <= 0) throw new ArgumentOutOfRangeException(nameof(sourceHeight));
        if (destinationWidth <= 0 || destinationWidth > sourceWidth) throw new ArgumentOutOfRangeException(nameof(destinationWidth));
        if (destinationHeight <= 0 || destinationHeight > sourceHeight) throw new ArgumentOutOfRangeException(nameof(destinationHeight));
        if (source.Length < checked(sourceWidth * sourceHeight)) throw new ArgumentException("Source is shorter than its dimensions.", nameof(source));
        if (destination.Length < checked(destinationWidth * destinationHeight)) throw new ArgumentException("Destination is shorter than its dimensions.", nameof(destination));
    }
}
