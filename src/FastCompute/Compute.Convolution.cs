using System.Collections.Concurrent;
using System.Numerics;
using FastCompute.Backends.Gpu;

namespace FastCompute;

public static partial class Compute
{
    /// <summary>Convolves a one-dimensional floating-point buffer.</summary>
    public static float[] Convolve1D(
        ReadOnlySpan<float> source,
        ReadOnlySpan<float> kernel,
        ConvolutionBoundary boundary = ConvolutionBoundary.Clamp,
        ComputeOptions? options = null)
    {
        ValidateKernel(kernel, nameof(kernel));
        float[] result = GC.AllocateUninitializedArray<float>(source.Length);
        Convolve1D(source, kernel, result, boundary, options);
        return result;
    }

    /// <summary>Convolves a one-dimensional floating-point buffer into an existing destination.</summary>
    public static void Convolve1D(
        ReadOnlySpan<float> source,
        ReadOnlySpan<float> kernel,
        Span<float> destination,
        ConvolutionBoundary boundary = ConvolutionBoundary.Clamp,
        ComputeOptions? options = null)
    {
        ValidateKernel(kernel, nameof(kernel));
        if (destination.Length < source.Length) throw new ArgumentException("Destination is shorter than source.", nameof(destination));
        if (source.Overlaps(destination)) throw new ArgumentException("Source and destination must not overlap.", nameof(destination));
        if (source.IsEmpty) return;

        ComputeOptions effective = options ?? ComputeOptions.Default;
        ValidateConvolutionOptions(effective);
        ComputeBackendKind backend = ResolveConvolutionBackend(effective, source.Length);
        float[] input = source.ToArray();
        float[] weights = kernel.ToArray();
        float[] output;
        if (backend == ComputeBackendKind.Gpu)
        {
            effective.CancellationToken.ThrowIfCancellationRequested();
            output = GpuComputeBackend.ResolveContext(effective).ExecuteConvolution1D(input, weights, boundary);
        }
        else
        {
            output = GC.AllocateUninitializedArray<float>(input.Length);
            if (backend == ComputeBackendKind.ParallelCpu)
                Convolve1DParallel(input, weights, output, boundary, effective);
            else if (backend == ComputeBackendKind.Simd)
                Convolve1DSimd(input, weights, output, boundary, effective.CancellationToken);
            else
                Convolve1DScalar(input, weights, output, boundary, 0, input.Length, effective.CancellationToken);
        }
        output.CopyTo(destination);
    }

    /// <summary>Convolves a row-major two-dimensional floating-point buffer.</summary>
    public static float[] Convolve2D(
        ReadOnlySpan<float> source,
        int width,
        int height,
        ReadOnlySpan<float> kernel,
        int kernelWidth,
        int kernelHeight,
        ConvolutionBoundary boundary = ConvolutionBoundary.Clamp,
        ComputeOptions? options = null)
    {
        var result = GC.AllocateUninitializedArray<float>(ValidateDimensions(source, width, height));
        Convolve2D(source, result, width, height, kernel, kernelWidth, kernelHeight, boundary, options);
        return result;
    }

    /// <summary>Convolves a row-major two-dimensional buffer into an existing destination.</summary>
    public static void Convolve2D(
        ReadOnlySpan<float> source,
        Span<float> destination,
        int width,
        int height,
        ReadOnlySpan<float> kernel,
        int kernelWidth,
        int kernelHeight,
        ConvolutionBoundary boundary = ConvolutionBoundary.Clamp,
        ComputeOptions? options = null)
    {
        int length = ValidateDimensions(source, width, height);
        ValidateKernelDimensions(kernel, kernelWidth, kernelHeight);
        if (destination.Length < length) throw new ArgumentException("Destination is shorter than the declared dimensions.", nameof(destination));
        if (source[..length].Overlaps(destination)) throw new ArgumentException("Source and destination must not overlap.", nameof(destination));

        ComputeOptions effective = options ?? ComputeOptions.Default;
        ValidateConvolutionOptions(effective);
        ComputeBackendKind backend = ResolveConvolutionBackend(effective, length);
        float[] input = source[..length].ToArray();
        float[] weights = kernel.ToArray();
        float[] output;
        if (backend == ComputeBackendKind.Gpu)
        {
            effective.CancellationToken.ThrowIfCancellationRequested();
            output = GpuComputeBackend.ResolveContext(effective).ExecuteConvolution2D(input, width, height, weights, kernelWidth, kernelHeight, boundary);
        }
        else
        {
            output = GC.AllocateUninitializedArray<float>(length);
            if (backend == ComputeBackendKind.ParallelCpu)
                Convolve2DParallel(input, output, width, height, weights, kernelWidth, kernelHeight, boundary, effective);
            else if (backend == ComputeBackendKind.Simd)
                Convolve2DSimd(input, output, width, height, weights, kernelWidth, kernelHeight, boundary, effective.CancellationToken);
            else
                Convolve2DScalar(input, output, width, height, weights, kernelWidth, kernelHeight, boundary, 0, height, effective.CancellationToken);
        }
        output.CopyTo(destination);
    }

    private static void Convolve1DParallel(float[] source, float[] kernel, float[] destination, ConvolutionBoundary boundary, ComputeOptions options)
    {
        var parallelOptions = new ParallelOptions { CancellationToken = options.CancellationToken, MaxDegreeOfParallelism = options.MaxDegreeOfParallelism ?? -1 };
        Parallel.ForEach(Partitioner.Create(0, source.Length), parallelOptions, range =>
            Convolve1DScalar(source, kernel, destination, boundary, range.Item1, range.Item2, options.CancellationToken));
    }

    private static void Convolve1DSimd(float[] source, float[] kernel, float[] destination, ConvolutionBoundary boundary, CancellationToken token)
    {
        int radius = kernel.Length / 2;
        Convolve1DScalar(source, kernel, destination, boundary, 0, Math.Min(radius, source.Length), token);
        int end = Math.Max(radius, source.Length - radius);
        int index = radius;
        int lanes = Vector<float>.Count;
        int vectorEnd = end - ((end - index) % lanes);
        for (; index < vectorEnd; index += lanes)
        {
            if ((index & 0xFFFF) == 0) token.ThrowIfCancellationRequested();
            Vector<float> sum = Vector<float>.Zero;
            for (int k = 0; k < kernel.Length; k++)
                sum += new Vector<float>(source, index + k - radius) * new Vector<float>(kernel[k]);
            sum.CopyTo(destination, index);
        }
        Convolve1DScalar(source, kernel, destination, boundary, index, source.Length, token);
    }

    private static void Convolve1DScalar(float[] source, float[] kernel, float[] destination, ConvolutionBoundary boundary, int start, int end, CancellationToken token)
    {
        int radius = kernel.Length / 2;
        for (int index = start; index < end; index++)
        {
            if ((index & 0xFFFF) == 0) token.ThrowIfCancellationRequested();
            float sum = 0f;
            for (int k = 0; k < kernel.Length; k++)
            {
                int sourceIndex = index + k - radius;
                if (boundary == ConvolutionBoundary.Clamp) sourceIndex = Math.Clamp(sourceIndex, 0, source.Length - 1);
                else if ((uint)sourceIndex >= (uint)source.Length) continue;
                sum += source[sourceIndex] * kernel[k];
            }
            destination[index] = sum;
        }
    }

    private static void Convolve2DParallel(float[] source, float[] destination, int width, int height, float[] kernel, int kernelWidth, int kernelHeight, ConvolutionBoundary boundary, ComputeOptions options)
    {
        var parallelOptions = new ParallelOptions { CancellationToken = options.CancellationToken, MaxDegreeOfParallelism = options.MaxDegreeOfParallelism ?? -1 };
        Parallel.For(0, height, parallelOptions, y =>
            Convolve2DScalar(source, destination, width, height, kernel, kernelWidth, kernelHeight, boundary, y, y + 1, options.CancellationToken));
    }

    private static void Convolve2DSimd(float[] source, float[] destination, int width, int height, float[] kernel, int kernelWidth, int kernelHeight, ConvolutionBoundary boundary, CancellationToken token)
    {
        int radiusX = kernelWidth / 2;
        int radiusY = kernelHeight / 2;
        for (int y = 0; y < height; y++)
        {
            token.ThrowIfCancellationRequested();
            if (y < radiusY || y >= height - radiusY || width <= radiusX * 2)
            {
                Convolve2DScalar(source, destination, width, height, kernel, kernelWidth, kernelHeight, boundary, y, y + 1, token);
                continue;
            }
            Convolve2DScalarRange(source, destination, width, height, kernel, kernelWidth, kernelHeight, boundary, y, 0, radiusX);
            int x = radiusX;
            int interiorEnd = width - radiusX;
            int lanes = Vector<float>.Count;
            int vectorEnd = interiorEnd - ((interiorEnd - x) % lanes);
            for (; x < vectorEnd; x += lanes)
            {
                Vector<float> sum = Vector<float>.Zero;
                for (int ky = 0; ky < kernelHeight; ky++)
                for (int kx = 0; kx < kernelWidth; kx++)
                    sum += new Vector<float>(source, ((y + ky - radiusY) * width) + x + kx - radiusX) * new Vector<float>(kernel[(ky * kernelWidth) + kx]);
                sum.CopyTo(destination, (y * width) + x);
            }
            Convolve2DScalarRange(source, destination, width, height, kernel, kernelWidth, kernelHeight, boundary, y, x, width);
        }
    }

    private static void Convolve2DScalar(float[] source, float[] destination, int width, int height, float[] kernel, int kernelWidth, int kernelHeight, ConvolutionBoundary boundary, int startY, int endY, CancellationToken token)
    {
        for (int y = startY; y < endY; y++)
        {
            token.ThrowIfCancellationRequested();
            Convolve2DScalarRange(source, destination, width, height, kernel, kernelWidth, kernelHeight, boundary, y, 0, width);
        }
    }

    private static void Convolve2DScalarRange(float[] source, float[] destination, int width, int height, float[] kernel, int kernelWidth, int kernelHeight, ConvolutionBoundary boundary, int y, int startX, int endX)
    {
        int radiusX = kernelWidth / 2;
        int radiusY = kernelHeight / 2;
        for (int x = startX; x < endX; x++)
        {
            float sum = 0f;
            for (int ky = 0; ky < kernelHeight; ky++)
            for (int kx = 0; kx < kernelWidth; kx++)
            {
                int sourceX = x + kx - radiusX;
                int sourceY = y + ky - radiusY;
                if (boundary == ConvolutionBoundary.Clamp)
                {
                    sourceX = Math.Clamp(sourceX, 0, width - 1);
                    sourceY = Math.Clamp(sourceY, 0, height - 1);
                }
                else if ((uint)sourceX >= (uint)width || (uint)sourceY >= (uint)height) continue;
                sum += source[(sourceY * width) + sourceX] * kernel[(ky * kernelWidth) + kx];
            }
            destination[(y * width) + x] = sum;
        }
    }

    private static ComputeBackendKind ResolveConvolutionBackend(ComputeOptions options, int length)
    {
        if (options.Backend == ComputeBackendKind.Gpu) return ComputeBackendKind.Gpu;
        if (options.Backend == ComputeBackendKind.Simd)
        {
            if (!Vector.IsHardwareAccelerated) throw new ComputeBackendUnavailableException(ComputeBackendKind.Simd);
            return ComputeBackendKind.Simd;
        }
        if (options.Backend != ComputeBackendKind.Auto) return options.Backend;
        if (length >= options.Thresholds.GpuHeavyThreshold && (options.GpuContext is not null || GpuComputeBackend.HasHardwareAccelerator)) return ComputeBackendKind.Gpu;
        if (length >= options.Thresholds.ParallelThreshold) return ComputeBackendKind.ParallelCpu;
        if (length >= options.Thresholds.SimdThreshold && Vector.IsHardwareAccelerated) return ComputeBackendKind.Simd;
        return ComputeBackendKind.Scalar;
    }

    private static void ValidateConvolutionOptions(ComputeOptions options)
    {
        ArgumentNullException.ThrowIfNull(options.Thresholds);
        if (options.GpuContext is not null && options.PreferredGpuAcceleratorIndex is not null)
            throw new ArgumentException("GpuContext and PreferredGpuAcceleratorIndex cannot be used together.", nameof(options));
    }

    private static void ValidateKernel(ReadOnlySpan<float> kernel, string parameterName)
    {
        if (kernel.IsEmpty || (kernel.Length & 1) == 0) throw new ArgumentException("Kernel length must be positive and odd.", parameterName);
    }

    private static void ValidateKernelDimensions(ReadOnlySpan<float> kernel, int width, int height)
    {
        if (width <= 0 || (width & 1) == 0) throw new ArgumentOutOfRangeException(nameof(width), "Kernel width must be positive and odd.");
        if (height <= 0 || (height & 1) == 0) throw new ArgumentOutOfRangeException(nameof(height), "Kernel height must be positive and odd.");
        if (kernel.Length != checked(width * height)) throw new ArgumentException("Kernel length does not match its dimensions.", nameof(kernel));
    }

    private static int ValidateDimensions(ReadOnlySpan<float> source, int width, int height)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        int length = checked(width * height);
        if (source.Length < length) throw new ArgumentException("Source is shorter than the declared dimensions.", nameof(source));
        return length;
    }
}
