using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;

namespace FastCompute.ImageProcessing;

/// <summary>Provides reusable operations over floating-point grayscale images.</summary>
public static class GrayImageOperations
{
    /// <summary>
    /// Applies a separable box blur. Radius one uses a native SIMD three-tap
    /// kernel; larger radii use linear-time sliding windows.
    /// </summary>
    public static Image<GrayF32> BoxBlur(
        this Image<GrayF32> source,
        int radius = 1,
        CancellationToken cancellationToken = default,
        ComputeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        var output = GC.AllocateUninitializedArray<GrayF32>(source.Length);
        BoxBlur(
            MemoryMarshal.Cast<GrayF32, float>(source.Pixels.Span),
            MemoryMarshal.Cast<GrayF32, float>(output),
            source.Width,
            source.Height,
            radius,
            cancellationToken,
            options);
        return Image<GrayF32>.Load(
            output,
            source.Width,
            source.Height,
            source.Encoding);
    }

    /// <summary>
    /// Applies a separable box blur directly to contiguous floating-point image
    /// buffers. Source and destination may overlap.
    /// </summary>
    public static void BoxBlur(
        ReadOnlySpan<float> source,
        Span<float> destination,
        int width,
        int height,
        int radius = 1,
        CancellationToken cancellationToken = default,
        ComputeOptions? options = null)
    {
        ValidateImageBuffers(source, destination, width, height);
        if (radius < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(radius));
        }

        int length = checked(width * height);
        if (radius == 0)
        {
            source[..length].CopyTo(destination);
            return;
        }

        ComputeOptions effectiveOptions = options ?? ComputeOptions.Default;
        float[]? gpuResult = ImageGpuExecutor.TryBoxBlur(
            source[..length],
            width,
            height,
            radius,
            effectiveOptions,
            cancellationToken);
        if (gpuResult is not null)
        {
            gpuResult.CopyTo(destination);
            return;
        }

        float[] temporary = ArrayPool<float>.Shared.Rent(length);
        try
        {
            Span<float> horizontal = temporary.AsSpan(0, length);
            if (radius == 1)
            {
                BlurThreeTapHorizontal(source, horizontal, width, height, cancellationToken);
                BlurThreeTapVertical(horizontal, destination, width, height, cancellationToken);
            }
            else
            {
                BlurSlidingHorizontal(source, horizontal, width, height, radius, cancellationToken);
                BlurSlidingVertical(horizontal, destination, width, height, radius, cancellationToken);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(temporary);
        }
    }

    /// <summary>Subtracts two equally sized grayscale images.</summary>
    public static Image<GrayF32> Subtract(
        this Image<GrayF32> left,
        Image<GrayF32> right,
        CancellationToken cancellationToken = default,
        ComputeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        if (left.Width != right.Width || left.Height != right.Height)
        {
            throw new ArgumentException("Images must have equal dimensions.", nameof(right));
        }
        if (left.Encoding != right.Encoding)
        {
            throw new ArgumentException("Images must use the same color encoding.", nameof(right));
        }

        var output = GC.AllocateUninitializedArray<GrayF32>(left.Length);
        Subtract(
            MemoryMarshal.Cast<GrayF32, float>(left.Pixels.Span),
            MemoryMarshal.Cast<GrayF32, float>(right.Pixels.Span),
            MemoryMarshal.Cast<GrayF32, float>(output),
            cancellationToken,
            options);
        return Image<GrayF32>.Load(output, left.Width, left.Height, left.Encoding);
    }

    /// <summary>Subtracts contiguous floating-point buffers using native SIMD where available.</summary>
    public static void Subtract(
        ReadOnlySpan<float> left,
        ReadOnlySpan<float> right,
        Span<float> destination,
        CancellationToken cancellationToken = default,
        ComputeOptions? options = null)
    {
        if (left.Length != right.Length || destination.Length < left.Length)
        {
            throw new ArgumentException("Buffers must have matching lengths.");
        }

        ComputeOptions effectiveOptions = options ?? ComputeOptions.Default;
        float[]? gpuResult = ImageGpuExecutor.TrySubtract(left, right, effectiveOptions, cancellationToken);
        if (gpuResult is not null)
        {
            gpuResult.CopyTo(destination);
            return;
        }

        int index = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int lanes = Vector<float>.Count;
            int vectorizedLength = left.Length - (left.Length % lanes);
            for (; index < vectorizedLength; index += lanes)
            {
                if ((index & 0xFFFF) == 0) cancellationToken.ThrowIfCancellationRequested();
                (new Vector<float>(left.Slice(index, lanes)) - new Vector<float>(right.Slice(index, lanes)))
                    .CopyTo(destination.Slice(index, lanes));
            }
        }
        for (; index < left.Length; index++) destination[index] = left[index] - right[index];
    }

    private static void BlurThreeTapHorizontal(ReadOnlySpan<float> source, Span<float> destination, int width, int height, CancellationToken cancellationToken)
    {
        float third = 1f / 3f;
        for (int y = 0; y < height; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ReadOnlySpan<float> input = source.Slice(y * width, width);
            Span<float> output = destination.Slice(y * width, width);
            if (width == 1) { output[0] = input[0]; continue; }
            output[0] = (input[0] + input[1]) * 0.5f;
            int x = 1;
            if (Vector.IsHardwareAccelerated)
            {
                int lanes = Vector<float>.Count;
                var scale = new Vector<float>(third);
                for (; x <= width - lanes - 1; x += lanes)
                {
                    Vector<float> sum = new Vector<float>(input.Slice(x - 1, lanes)) +
                        new Vector<float>(input.Slice(x, lanes)) +
                        new Vector<float>(input.Slice(x + 1, lanes));
                    (sum * scale).CopyTo(output.Slice(x, lanes));
                }
            }
            for (; x < width - 1; x++) output[x] = (input[x - 1] + input[x] + input[x + 1]) * third;
            output[^1] = (input[^2] + input[^1]) * 0.5f;
        }
    }

    private static void BlurThreeTapVertical(ReadOnlySpan<float> source, Span<float> destination, int width, int height, CancellationToken cancellationToken)
    {
        if (height == 1) { source[..width].CopyTo(destination); return; }
        CombineRows(source[..width], source.Slice(width, width), default, destination[..width], 2);
        for (int y = 1; y < height - 1; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            CombineRows(
                source.Slice((y - 1) * width, width),
                source.Slice(y * width, width),
                source.Slice((y + 1) * width, width),
                destination.Slice(y * width, width),
                3);
        }
        CombineRows(
            source.Slice((height - 2) * width, width),
            source.Slice((height - 1) * width, width),
            default,
            destination.Slice((height - 1) * width, width),
            2);
    }

    private static void CombineRows(ReadOnlySpan<float> first, ReadOnlySpan<float> second, ReadOnlySpan<float> third, Span<float> destination, int divisor)
    {
        int x = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int lanes = Vector<float>.Count;
            int vectorizedLength = destination.Length - (destination.Length % lanes);
            var scale = new Vector<float>(1f / divisor);
            for (; x < vectorizedLength; x += lanes)
            {
                Vector<float> sum = new Vector<float>(first.Slice(x, lanes)) + new Vector<float>(second.Slice(x, lanes));
                if (divisor == 3) sum += new Vector<float>(third.Slice(x, lanes));
                (sum * scale).CopyTo(destination.Slice(x, lanes));
            }
        }
        for (; x < destination.Length; x++)
        {
            float sum = first[x] + second[x] + (divisor == 3 ? third[x] : 0);
            destination[x] = sum / divisor;
        }
    }

    private static void BlurSlidingHorizontal(ReadOnlySpan<float> source, Span<float> destination, int width, int height, int radius, CancellationToken cancellationToken)
    {
        for (int y = 0; y < height; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int offset = y * width;
            int end = Math.Min(width - 1, radius);
            double sum = 0;
            for (int x = 0; x <= end; x++) sum += source[offset + x];
            for (int x = 0; x < width; x++)
            {
                int start = Math.Max(0, x - radius);
                end = Math.Min(width - 1, x + radius);
                destination[offset + x] = (float)(sum / (end - start + 1));
                int remove = x - radius;
                int add = x + radius + 1;
                if (remove >= 0) sum -= source[offset + remove];
                if (add < width) sum += source[offset + add];
            }
        }
    }

    private static void BlurSlidingVertical(ReadOnlySpan<float> source, Span<float> destination, int width, int height, int radius, CancellationToken cancellationToken)
    {
        float[] sumsBuffer = ArrayPool<float>.Shared.Rent(width);
        try
        {
            Span<float> sums = sumsBuffer.AsSpan(0, width);
            sums.Clear();
            int initialEnd = Math.Min(height - 1, radius);
            for (int y = 0; y <= initialEnd; y++) AddRow(sums, source.Slice(y * width, width), 1f);
            for (int y = 0; y < height; y++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int start = Math.Max(0, y - radius);
                int end = Math.Min(height - 1, y + radius);
                ScaleRow(sums, destination.Slice(y * width, width), 1f / (end - start + 1));
                int remove = y - radius;
                int add = y + radius + 1;
                if (remove >= 0) AddRow(sums, source.Slice(remove * width, width), -1f);
                if (add < height) AddRow(sums, source.Slice(add * width, width), 1f);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(sumsBuffer);
        }
    }

    private static void AddRow(Span<float> target, ReadOnlySpan<float> row, float multiplier)
    {
        int x = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int lanes = Vector<float>.Count;
            int length = target.Length - (target.Length % lanes);
            var factor = new Vector<float>(multiplier);
            for (; x < length; x += lanes)
            {
                (new Vector<float>(target.Slice(x, lanes)) + (new Vector<float>(row.Slice(x, lanes)) * factor))
                    .CopyTo(target.Slice(x, lanes));
            }
        }
        for (; x < target.Length; x++) target[x] += row[x] * multiplier;
    }

    private static void ScaleRow(ReadOnlySpan<float> source, Span<float> destination, float multiplier)
    {
        int x = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int lanes = Vector<float>.Count;
            int length = source.Length - (source.Length % lanes);
            var factor = new Vector<float>(multiplier);
            for (; x < length; x += lanes)
                (new Vector<float>(source.Slice(x, lanes)) * factor).CopyTo(destination.Slice(x, lanes));
        }
        for (; x < source.Length; x++) destination[x] = source[x] * multiplier;
    }

    private static void ValidateImageBuffers(ReadOnlySpan<float> source, Span<float> destination, int width, int height)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        int length = checked(width * height);
        if (source.Length < length) throw new ArgumentException("Source is shorter than the image dimensions.", nameof(source));
        if (destination.Length < length) throw new ArgumentException("Destination is shorter than the image dimensions.", nameof(destination));
    }
}
