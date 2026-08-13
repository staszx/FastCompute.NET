using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace FastCompute.Backends;

internal static class FourierCpuExecutor
{
    internal static void Transform1D(Complex32[] data, FourierDirection direction, ComputeBackendKind backend, ComputeOptions options)
    {
        BitReverse(data);
        TransformStages(data, direction, backend, options);
        if (direction == FourierDirection.Inverse) Scale(data, 1f / data.Length, backend, options.CancellationToken);
    }

    internal static void Transform2D(Complex32[] data, int width, int height, FourierDirection direction, ComputeBackendKind backend, ComputeOptions options)
    {
        var parallelOptions = new ParallelOptions
        {
            CancellationToken = options.CancellationToken,
            MaxDegreeOfParallelism = options.MaxDegreeOfParallelism ?? -1
        };
        if (backend == ComputeBackendKind.ParallelCpu)
        {
            Parallel.For(0, height, parallelOptions, y => TransformUnnormalized(data.AsSpan(y * width, width), direction, simd: false));
            Parallel.For(0, width, parallelOptions, x => TransformColumn(data, width, height, x, direction, simd: false));
        }
        else
        {
            bool simd = backend == ComputeBackendKind.Simd;
            for (int y = 0; y < height; y++)
            {
                options.CancellationToken.ThrowIfCancellationRequested();
                TransformUnnormalized(data.AsSpan(y * width, width), direction, simd);
            }
            var column = new Complex32[height];
            for (int x = 0; x < width; x++)
            {
                options.CancellationToken.ThrowIfCancellationRequested();
                for (int y = 0; y < height; y++) column[y] = data[(y * width) + x];
                TransformUnnormalized(column, direction, simd);
                for (int y = 0; y < height; y++) data[(y * width) + x] = column[y];
            }
        }
        if (direction == FourierDirection.Inverse) Scale(data, 1f / data.Length, backend, options.CancellationToken);
    }

    private static void TransformColumn(Complex32[] data, int width, int height, int x, FourierDirection direction, bool simd)
    {
        var column = new Complex32[height];
        for (int y = 0; y < height; y++) column[y] = data[(y * width) + x];
        TransformUnnormalized(column, direction, simd);
        for (int y = 0; y < height; y++) data[(y * width) + x] = column[y];
    }

    private static void TransformUnnormalized(Span<Complex32> values, FourierDirection direction, bool simd)
    {
        BitReverse(values);
        for (int size = 2; size <= values.Length; size <<= 1)
        {
            int half = size >> 1;
            float angle = (direction == FourierDirection.Forward ? -2f : 2f) * MathF.PI / size;
            for (int start = 0; start < values.Length; start += size)
                ButterflyBlock(values, start, half, angle, simd);
        }
    }

    private static void TransformStages(Complex32[] data, FourierDirection direction, ComputeBackendKind backend, ComputeOptions options)
    {
        for (int size = 2; size <= data.Length; size <<= 1)
        {
            options.CancellationToken.ThrowIfCancellationRequested();
            int stageSize = size;
            int half = size >> 1;
            float angle = (direction == FourierDirection.Forward ? -2f : 2f) * MathF.PI / size;
            if (backend == ComputeBackendKind.ParallelCpu)
            {
                Parallel.For(0, data.Length / size, new ParallelOptions
                {
                    CancellationToken = options.CancellationToken,
                    MaxDegreeOfParallelism = options.MaxDegreeOfParallelism ?? -1
                }, block => ButterflyBlock(data, block * stageSize, half, angle, simd: false));
            }
            else
            {
                bool simd = backend == ComputeBackendKind.Simd;
                for (int start = 0; start < data.Length; start += size)
                    ButterflyBlock(data, start, half, angle, simd);
            }
        }
    }

    private static void ButterflyBlock(Span<Complex32> values, int start, int half, float angle, bool simd)
    {
        int j = 0;
        var step = new Complex32(MathF.Cos(angle), MathF.Sin(angle));
        var factor = new Complex32(1f, 0f);
        if (simd && Avx.IsSupported)
        {
            Span<float> floats = MemoryMarshal.Cast<Complex32, float>(values);
            ref float reference = ref MemoryMarshal.GetReference(floats);
            for (; j <= half - 4; j += 4)
            {
                Vector256<float> even = Vector256.LoadUnsafe(ref reference, (nuint)((start + j) * 2));
                Vector256<float> odd = Vector256.LoadUnsafe(ref reference, (nuint)((start + half + j) * 2));
                Vector256<float> twiddle = CreateTwiddles(factor, step, out factor);
                Vector256<float> real = Avx.Permute(twiddle, 0xA0);
                Vector256<float> imaginary = Avx.Permute(twiddle, 0xF5);
                Vector256<float> product = Avx.AddSubtract(
                    Avx.Multiply(odd, real),
                    Avx.Multiply(Avx.Permute(odd, 0xB1), imaginary));
                Avx.Add(even, product).StoreUnsafe(ref reference, (nuint)((start + j) * 2));
                Avx.Subtract(even, product).StoreUnsafe(ref reference, (nuint)((start + half + j) * 2));
            }
        }
        for (; j < half; j++)
        {
            Complex32 even = values[start + j];
            Complex32 odd = values[start + half + j] * factor;
            values[start + j] = even + odd;
            values[start + half + j] = even - odd;
            factor *= step;
        }
    }

    private static Vector256<float> CreateTwiddles(Complex32 first, Complex32 step, out Complex32 next)
    {
        Complex32 second = first * step;
        Complex32 third = second * step;
        Complex32 fourth = third * step;
        next = fourth * step;
        return Vector256.Create(
            first.Real, first.Imaginary,
            second.Real, second.Imaginary,
            third.Real, third.Imaginary,
            fourth.Real, fourth.Imaginary);
    }

    private static void BitReverse(Span<Complex32> values)
    {
        for (int i = 1, j = 0; i < values.Length; i++)
        {
            int bit = values.Length >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) (values[i], values[j]) = (values[j], values[i]);
        }
    }

    private static void Scale(Complex32[] values, float scale, ComputeBackendKind backend, CancellationToken cancellationToken)
    {
        if (backend == ComputeBackendKind.ParallelCpu)
        {
            Parallel.For(0, values.Length, index => values[index] = values[index] * scale);
            return;
        }
        int index = 0;
        if (backend == ComputeBackendKind.Simd && Avx.IsSupported)
        {
            Span<float> floats = MemoryMarshal.Cast<Complex32, float>(values);
            ref float reference = ref MemoryMarshal.GetReference(floats);
            Vector256<float> factor = Vector256.Create(scale);
            for (; index <= floats.Length - 8; index += 8)
                Avx.Multiply(Vector256.LoadUnsafe(ref reference, (nuint)index), factor).StoreUnsafe(ref reference, (nuint)index);
            index /= 2;
        }
        for (; index < values.Length; index++) values[index] = values[index] * scale;
        cancellationToken.ThrowIfCancellationRequested();
    }
}
