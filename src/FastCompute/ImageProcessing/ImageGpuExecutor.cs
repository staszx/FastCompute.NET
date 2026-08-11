using System.Runtime.InteropServices;
using FastCompute.Backends.Gpu;

namespace FastCompute.ImageProcessing;

internal static class ImageGpuExecutor
{
    internal static bool ShouldUseGpu(ComputeOptions options, int length)
    {
        ValidateOptions(options);
        if (options.Backend == ComputeBackendKind.Gpu) return true;
        if (options.Backend != ComputeBackendKind.Auto) return false;
        return length >= options.Thresholds.GpuSimpleThreshold &&
            (options.GpuContext is not null || GpuComputeBackend.HasHardwareAccelerator);
    }

    internal static bool TryConvert<TSource, TDestination>(
        ReadOnlySpan<TSource> source,
        ColorEncoding sourceEncoding,
        ColorEncoding destinationEncoding,
        ComputeOptions options,
        out TDestination[] destination)
        where TSource : unmanaged
        where TDestination : unmanaged
    {
        destination = [];
        if (!ShouldUseGpu(options, source.Length)) return false;

        (bool sourceFloat, int sourceComponents) = GetFormat<TSource>();
        (bool destinationFloat, int destinationComponents) = GetFormat<TDestination>();
        if (source.Length == 0) return true;
        options.CancellationToken.ThrowIfCancellationRequested();
        try
        {
            ComputeContext context = GpuComputeBackend.ResolveContext(options);
            if (sourceFloat)
            {
                float[] input = MemoryMarshal.Cast<TSource, float>(source).ToArray();
                if (destinationFloat)
                {
                    float[] output = context.ExecuteImageConversion(input, source.Length, sourceComponents, destinationComponents, (int)sourceEncoding, (int)destinationEncoding, floatDestination: true);
                    destination = MemoryMarshal.Cast<float, TDestination>(output).ToArray();
                }
                else
                {
                    byte[] output = context.ExecuteImageConversion(input, source.Length, sourceComponents, destinationComponents, (int)sourceEncoding, (int)destinationEncoding);
                    destination = MemoryMarshal.Cast<byte, TDestination>(output).ToArray();
                }
            }
            else
            {
                byte[] input = MemoryMarshal.AsBytes(source).ToArray();
                if (destinationFloat)
                {
                    float[] output = context.ExecuteImageConversion(input, source.Length, sourceComponents, destinationComponents, (int)sourceEncoding, (int)destinationEncoding, floatDestination: true);
                    destination = MemoryMarshal.Cast<float, TDestination>(output).ToArray();
                }
                else
                {
                    byte[] output = context.ExecuteImageConversion(input, source.Length, sourceComponents, destinationComponents, (int)sourceEncoding, (int)destinationEncoding);
                    destination = MemoryMarshal.Cast<byte, TDestination>(output).ToArray();
                }
            }
            options.CancellationToken.ThrowIfCancellationRequested();
            return true;
        }
        catch when (options.Backend == ComputeBackendKind.Auto && options.AllowFallback)
        {
            destination = [];
            return false;
        }
    }

    internal static float[]? TrySubtract(ReadOnlySpan<float> left, ReadOnlySpan<float> right, ComputeOptions options, CancellationToken cancellationToken)
    {
        if (left.Length == 0) return [];
        if (!ShouldUseGpu(options, left.Length)) return null;
        float[] leftArray = left.ToArray();
        float[] rightArray = right.ToArray();
        return Execute(options, cancellationToken, context => context.ExecuteImageSubtract(leftArray, rightArray));
    }

    internal static float[]? TryBoxBlur(ReadOnlySpan<float> source, int width, int height, int radius, ComputeOptions options, CancellationToken cancellationToken)
    {
        if (!ShouldUseGpu(options, source.Length)) return null;
        float[] sourceArray = source.ToArray();
        return Execute(options, cancellationToken, context => context.ExecuteImageBoxBlur(sourceArray, width, height, radius));
    }

    internal static float[]? TryDownsample(ReadOnlySpan<float> source, int sourceWidth, int sourceHeight, int destinationWidth, int destinationHeight, ComputeOptions options, CancellationToken cancellationToken)
    {
        if (!ShouldUseGpu(options, source.Length)) return null;
        float[] sourceArray = source.ToArray();
        return Execute(options, cancellationToken, context => context.ExecuteImageDownsample(sourceArray, sourceWidth, sourceHeight, destinationWidth, destinationHeight));
    }

    private static float[]? Execute(ComputeOptions options, CancellationToken cancellationToken, Func<ComputeContext, float[]> operation)
    {
        cancellationToken.ThrowIfCancellationRequested();
        options.CancellationToken.ThrowIfCancellationRequested();
        try
        {
            float[] result = operation(GpuComputeBackend.ResolveContext(options));
            cancellationToken.ThrowIfCancellationRequested();
            options.CancellationToken.ThrowIfCancellationRequested();
            return result;
        }
        catch when (options.Backend == ComputeBackendKind.Auto && options.AllowFallback)
        {
            return null;
        }
    }

    internal static (bool IsFloat, int ComponentCount) GetFormat<T>() where T : unmanaged
    {
        if (typeof(T) == typeof(Rgb)) return (true, 3);
        if (typeof(T) == typeof(GrayF32)) return (true, 1);
        if (typeof(T) == typeof(Rgb24)) return (false, 3);
        if (typeof(T) == typeof(Gray8)) return (false, 1);
        throw new NotSupportedException($"GPU pixel conversion does not support '{typeof(T).Name}'.");
    }

    private static void ValidateOptions(ComputeOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(options.Thresholds);
        if (options.GpuContext is not null && options.PreferredGpuAcceleratorIndex is not null)
            throw new ArgumentException("GpuContext and PreferredGpuAcceleratorIndex cannot be used together.", nameof(options));
    }
}
