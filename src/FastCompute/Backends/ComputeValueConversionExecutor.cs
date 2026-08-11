using System.Linq.Expressions;
using System.Numerics;
using System.Runtime.InteropServices;
using FastCompute.Backends.Gpu;
using FastCompute.Expressions;
using FastCompute.Gpu;

namespace FastCompute.Backends;

internal static class ComputeValueConversionExecutor
{
    internal static TDestination[] Transform<TSource, TDestination>(
        TSource[] source,
        Expression<Func<TSource, TDestination>> expression,
        ComputeOptions options)
        where TSource : unmanaged, IComputeValue<TSource>
        where TDestination : unmanaged, IComputeValue<TDestination>
    {
        Validate<TSource, TDestination>(source, expression, options);
        if (TSource.ComputeDescriptor.ComponentType ==
                ComputeComponentType.Float32 &&
            TDestination.ComputeDescriptor.ComponentType ==
                ComputeComponentType.Float32)
        {
            return ComputeValueExecutor.Transform(source, expression, options);
        }

        if (source.Length == 0)
        {
            return [];
        }

        bool nativeByte = TSource.ComputeDescriptor.ComponentType == ComputeComponentType.Byte &&
            TDestination.ComputeDescriptor.ComponentType == ComputeComponentType.Byte;
        ComputeBackendKind backend = ResolveBackend(source.Length, options, nativeByte);
        if (backend == ComputeBackendKind.Gpu &&
            TSource.ComputeDescriptor.ComponentType == ComputeComponentType.Byte &&
            TDestination.ComputeDescriptor.ComponentType == ComputeComponentType.Byte)
        {
            try
            {
                ByteComputeProgram byteProgram = ByteComputeParser.ParseMap(expression, TSource.ComputeDescriptor, TDestination.ComputeDescriptor);
                byte[] output = GpuComputeBackend.ResolveContext(options).ExecuteByteCompositeMap(MemoryMarshal.AsBytes(source.AsSpan()).ToArray(), source.Length, TSource.ComputeDescriptor.ComponentCount, ByteCompositeGpuProgramCompiler.Compile(byteProgram));
                return MemoryMarshal.Cast<byte, TDestination>(output).ToArray();
            }
            catch when (options.Backend == ComputeBackendKind.Auto && options.AllowFallback)
            {
                backend = ResolveCpuBackend(source.Length, options);
            }
        }
        Func<TSource, TDestination> operation = expression.Compile();
        if (backend == ComputeBackendKind.Simd &&
            TSource.ComputeDescriptor.ComponentType == ComputeComponentType.Byte &&
            TDestination.ComputeDescriptor.ComponentType == ComputeComponentType.Byte)
        {
            return ByteComputeSimdExecutor.Transform(
                source,
                ByteComputeParser.ParseMap(expression, TSource.ComputeDescriptor, TDestination.ComputeDescriptor),
                operation,
                options.CancellationToken);
        }
        var destination = GC.AllocateUninitializedArray<TDestination>(
            source.Length);
        Execute(source, destination, operation, backend, options);
        return destination;
    }

    internal static float[] Project<TSource>(
        TSource[] source,
        Expression<Func<TSource, float>> expression,
        ComputeOptions options)
        where TSource : unmanaged, IComputeValue<TSource>
    {
        ArgumentNullException.ThrowIfNull(expression);
        ValidateOptions(source, options);
        _ = TSource.ComputeDescriptor;
        if (source.Length == 0)
        {
            return [];
        }

        ComputeBackendKind backend = ResolveBackend(
            source.Length,
            options,
            TSource.ComputeDescriptor.ComponentType == ComputeComponentType.Byte);
        if (backend == ComputeBackendKind.Gpu &&
            TSource.ComputeDescriptor.ComponentType == ComputeComponentType.Byte)
        {
            try
            {
                ByteComputeProgram byteProgram = ByteComputeParser.ParseProjection(expression, TSource.ComputeDescriptor);
                return GpuComputeBackend.ResolveContext(options).ExecuteByteCompositeProjection(MemoryMarshal.AsBytes(source.AsSpan()).ToArray(), source.Length, TSource.ComputeDescriptor.ComponentCount, ByteCompositeGpuProgramCompiler.Compile(byteProgram));
            }
            catch when (options.Backend == ComputeBackendKind.Auto && options.AllowFallback)
            {
                backend = ResolveCpuBackend(source.Length, options);
            }
        }
        Func<TSource, float> operation = expression.Compile();
        if (backend == ComputeBackendKind.Simd &&
            TSource.ComputeDescriptor.ComponentType == ComputeComponentType.Byte)
        {
            return ByteComputeSimdExecutor.Project(
                source,
                ByteComputeParser.ParseProjection(expression, TSource.ComputeDescriptor),
                operation,
                options.CancellationToken);
        }
        var destination = GC.AllocateUninitializedArray<float>(source.Length);
        Execute(source, destination, operation, backend, options);
        return destination;
    }

    internal static TSource[] Map<TSource>(
        TSource[] source,
        Expression<Func<TSource, TSource>> expression,
        ComputeOptions options,
        bool inPlace)
        where TSource : unmanaged, IComputeValue<TSource>
    {
        Validate<TSource, TSource>(source, expression, options);
        if (source.Length == 0)
        {
            return inPlace ? source : [];
        }

        ComputeBackendKind backend = ResolveBackend(
            source.Length,
            options,
            TSource.ComputeDescriptor.ComponentType == ComputeComponentType.Byte);
        if (backend == ComputeBackendKind.Gpu &&
            TSource.ComputeDescriptor.ComponentType == ComputeComponentType.Byte)
        {
            try
            {
                ByteComputeProgram byteProgram = ByteComputeParser.ParseMap(expression, TSource.ComputeDescriptor, TSource.ComputeDescriptor);
                byte[] output = GpuComputeBackend.ResolveContext(options).ExecuteByteCompositeMap(MemoryMarshal.AsBytes(source.AsSpan()).ToArray(), source.Length, TSource.ComputeDescriptor.ComponentCount, ByteCompositeGpuProgramCompiler.Compile(byteProgram));
                TSource[] result = MemoryMarshal.Cast<byte, TSource>(output).ToArray();
                if (inPlace)
                {
                    result.CopyTo(source, 0);
                    return source;
                }
                return result;
            }
            catch when (options.Backend == ComputeBackendKind.Auto && options.AllowFallback)
            {
                backend = ResolveCpuBackend(source.Length, options);
            }
        }
        Func<TSource, TSource> operation = expression.Compile();
        if (backend == ComputeBackendKind.Simd &&
            TSource.ComputeDescriptor.ComponentType == ComputeComponentType.Byte)
        {
            return ByteComputeSimdExecutor.Map(
                source,
                ByteComputeParser.ParseMap(expression, TSource.ComputeDescriptor, TSource.ComputeDescriptor),
                operation,
                inPlace,
                options.CancellationToken);
        }
        TSource[] destination = inPlace
            ? source
            : GC.AllocateUninitializedArray<TSource>(source.Length);
        Execute(source, destination, operation, backend, options);
        return destination;
    }

    private static void Execute<TSource, TDestination>(
        TSource[] source,
        TDestination[] destination,
        Func<TSource, TDestination> operation,
        ComputeBackendKind backend,
        ComputeOptions options)
    {
        switch (backend)
        {
            case ComputeBackendKind.Scalar:
                for (int index = 0; index < source.Length; index++)
                {
                    if ((index & 0xFFF) == 0)
                    {
                        options.CancellationToken.ThrowIfCancellationRequested();
                    }

                    destination[index] = operation(source[index]);
                }

                return;
            case ComputeBackendKind.ParallelCpu:
                Parallel.For(
                    0,
                    source.Length,
                    new ParallelOptions
                    {
                        CancellationToken = options.CancellationToken,
                        MaxDegreeOfParallelism =
                            options.MaxDegreeOfParallelism ?? -1
                    },
                    index => destination[index] = operation(source[index]));
                return;
            case ComputeBackendKind.Simd:
            case ComputeBackendKind.Gpu:
                throw new ComputeBackendNotSupportedException(
                    backend,
                    "mixed component-type transformations",
                    "Scalar, ParallelCpu");
            default:
                throw new ArgumentOutOfRangeException(nameof(backend));
        }
    }

    private static ComputeBackendKind ResolveBackend(
        int length,
        ComputeOptions options,
        bool supportsNativeByte)
    {
        if (options.Backend != ComputeBackendKind.Auto)
        {
            return options.Backend;
        }

        if (supportsNativeByte && length >= options.Thresholds.GpuSimpleThreshold &&
            (options.GpuContext is not null || GpuComputeBackend.HasHardwareAccelerator))
            return ComputeBackendKind.Gpu;

        return supportsNativeByte
            ? ResolveCpuBackend(length, options)
            : length >= options.Thresholds.ParallelThreshold && Environment.ProcessorCount > 1
                ? ComputeBackendKind.ParallelCpu
                : ComputeBackendKind.Scalar;
    }

    private static ComputeBackendKind ResolveCpuBackend(int length, ComputeOptions options)
    {
        if (length >= options.Thresholds.ParallelThreshold && Environment.ProcessorCount > 1)
            return ComputeBackendKind.ParallelCpu;
        return Vector.IsHardwareAccelerated && length >= options.Thresholds.SimdThreshold
            ? ComputeBackendKind.Simd
            : ComputeBackendKind.Scalar;
    }

    private static void Validate<TSource, TDestination>(
        TSource[] source,
        LambdaExpression expression,
        ComputeOptions options)
        where TSource : unmanaged, IComputeValue<TSource>
        where TDestination : unmanaged, IComputeValue<TDestination>
    {
        ArgumentNullException.ThrowIfNull(expression);
        ValidateOptions(source, options);
        _ = TSource.ComputeDescriptor;
        _ = TDestination.ComputeDescriptor;
    }

    private static void ValidateOptions<TSource>(
        TSource[] source,
        ComputeOptions options)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(options.Thresholds);
        if (options.MaxDegreeOfParallelism is <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                "MaxDegreeOfParallelism must be positive when specified.");
        }

        options.CancellationToken.ThrowIfCancellationRequested();
    }
}
