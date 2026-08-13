using System.Numerics;
using System.Runtime.Intrinsics.X86;
using FastCompute.Backends;
using FastCompute.Backends.Gpu;

namespace FastCompute;

public static partial class Compute
{
    /// <summary>Calculates the inverse discrete Fourier transform.</summary>
    public static Complex32[] InverseFft(Complex32[] source, ComputeOptions? options = null) =>
        Fft(source, FourierDirection.Inverse, options);

    /// <summary>Calculates the inverse discrete Fourier transform in place.</summary>
    public static Complex32[] InverseFftInPlace(Complex32[] source, ComputeOptions? options = null) =>
        FftInPlace(source, FourierDirection.Inverse, options);

    /// <summary>Calculates the inverse two-dimensional discrete Fourier transform.</summary>
    public static Complex32[] InverseFft2D(Complex32[] source, int width, int height, ComputeOptions? options = null) =>
        Fft2D(source, width, height, FourierDirection.Inverse, options);

    /// <summary>Calculates the inverse two-dimensional discrete Fourier transform in place.</summary>
    public static Complex32[] InverseFft2DInPlace(Complex32[] source, int width, int height, ComputeOptions? options = null) =>
        Fft2DInPlace(source, width, height, FourierDirection.Inverse, options);

    /// <summary>Computes a radix-2 complex FFT and returns a new array.</summary>
    public static Complex32[] Fft(Complex32[] source, FourierDirection direction = FourierDirection.Forward, ComputeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        Complex32[] result = (Complex32[])source.Clone();
        return FftInPlace(result, direction, options);
    }

    /// <summary>Computes a radix-2 complex FFT in place.</summary>
    public static Complex32[] FftInPlace(Complex32[] source, FourierDirection direction = FourierDirection.Forward, ComputeOptions? options = null)
    {
        ValidateFourier(source, source.Length, 1, direction, options, out ComputeOptions effective);
        if (source.Length <= 1) return source;
        ComputeBackendKind backend = ResolveFourierBackend(source.Length, effective);
        if (backend == ComputeBackendKind.Gpu)
            GpuComputeBackend.ResolveContext(effective).ExecuteFft(source, source.Length, 1, direction);
        else
            FourierCpuExecutor.Transform1D(source, direction, backend, effective);
        return source;
    }

    /// <summary>Computes a row-major two-dimensional radix-2 complex FFT and returns a new array.</summary>
    public static Complex32[] Fft2D(Complex32[] source, int width, int height, FourierDirection direction = FourierDirection.Forward, ComputeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        Complex32[] result = (Complex32[])source.Clone();
        return Fft2DInPlace(result, width, height, direction, options);
    }

    /// <summary>Computes a row-major two-dimensional radix-2 complex FFT in place.</summary>
    public static Complex32[] Fft2DInPlace(Complex32[] source, int width, int height, FourierDirection direction = FourierDirection.Forward, ComputeOptions? options = null)
    {
        ValidateFourier(source, width, height, direction, options, out ComputeOptions effective);
        if (source.Length <= 1) return source;
        ComputeBackendKind backend = ResolveFourierBackend(source.Length, effective);
        if (backend == ComputeBackendKind.Gpu)
            GpuComputeBackend.ResolveContext(effective).ExecuteFft(source, width, height, direction);
        else
            FourierCpuExecutor.Transform2D(source, width, height, direction, backend, effective);
        return source;
    }

    private static void ValidateFourier(Complex32[] source, int width, int height, FourierDirection direction, ComputeOptions? options, out ComputeOptions effective)
    {
        if (!Enum.IsDefined(direction)) throw new ArgumentOutOfRangeException(nameof(direction));
        if (width <= 0 || height <= 0 || !BitOperations.IsPow2((uint)width) || !BitOperations.IsPow2((uint)height) || source.Length != checked(width * height))
            throw new ArgumentException("FFT dimensions must be positive powers of two matching the source length.");
        effective = options ?? ComputeOptions.Default;
        ArgumentNullException.ThrowIfNull(effective.Thresholds);
        if (effective.MaxDegreeOfParallelism is <= 0) throw new ArgumentOutOfRangeException(nameof(options));
        if (effective.GpuContext is not null && effective.PreferredGpuAcceleratorIndex is not null) throw new ArgumentException("GpuContext and PreferredGpuAcceleratorIndex cannot be combined.", nameof(options));
        effective.CancellationToken.ThrowIfCancellationRequested();
    }

    private static ComputeBackendKind ResolveFourierBackend(int length, ComputeOptions options)
    {
        if (options.Backend != ComputeBackendKind.Auto)
        {
            if (options.Backend == ComputeBackendKind.Simd && !Avx.IsSupported)
                throw new ComputeBackendNotSupportedException(ComputeBackendKind.Simd, "FFT on the current CPU", "Scalar, ParallelCpu, Gpu");
            return options.Backend;
        }
        if (length >= options.Thresholds.GpuHeavyThreshold && (options.GpuContext is not null || GpuComputeBackend.HasHardwareAccelerator)) return ComputeBackendKind.Gpu;
        if (length >= options.Thresholds.ParallelThreshold && Environment.ProcessorCount > 1) return ComputeBackendKind.ParallelCpu;
        return Avx.IsSupported && length >= options.Thresholds.SimdThreshold ? ComputeBackendKind.Simd : ComputeBackendKind.Scalar;
    }
}
