using System.Linq.Expressions;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using FastCompute.Expressions;
using FastCompute.Backends.Gpu;
using FastCompute.Gpu;

namespace FastCompute.Backends;

internal static class ComputeValueExecutor
{
    internal static TDestination[] Transform<TSource, TDestination>(
        TSource[] source,
        Expression<Func<TSource, TDestination>> expression,
        ComputeOptions options)
        where TSource : unmanaged, IComputeValue<TSource>
        where TDestination : unmanaged, IComputeValue<TDestination>
    {
        Validate(source, options);
        ComputeValueDescriptor<TSource> sourceDescriptor =
            TSource.ComputeDescriptor;
        ComputeValueDescriptor<TDestination> destinationDescriptor =
            TDestination.ComputeDescriptor;
        if (sourceDescriptor.ComponentType != ComputeComponentType.Float32 ||
            destinationDescriptor.ComponentType != ComputeComponentType.Float32)
        {
            return ComputeValueConversionExecutor.Transform(
                source,
                expression,
                options);
        }

        ComputeValueExpressionProgram program =
            ComputeValueExpressionParser.ParseMap(
                expression,
                sourceDescriptor,
                destinationDescriptor);
        if (source.Length == 0)
        {
            return [];
        }

        ComputeBackendKind backend = ResolveBackend(source.Length, options);
        if (backend == ComputeBackendKind.Gpu)
        {
            try
            {
                float[] output = GpuComputeBackend.ResolveContext(options)
                    .ExecuteCompositeValue(
                        MemoryMarshal.Cast<TSource, float>(source).ToArray(),
                        source.Length,
                        sourceDescriptor.ComponentCount,
                        CompositeGpuProgramCompiler.Compile(program));
                return MemoryMarshal.Cast<float, TDestination>(output).ToArray();
            }
            catch when (options.Backend == ComputeBackendKind.Auto && options.AllowFallback)
            {
                backend = ResolveCpuBackend(source.Length, options);
            }
        }

        var destination = GC.AllocateUninitializedArray<TDestination>(
            source.Length);
        switch (backend)
        {
            case ComputeBackendKind.Scalar:
                TransformScalar(source, destination, expression, options);
                break;
            case ComputeBackendKind.ParallelCpu:
                TransformParallel(source, destination, expression, options);
                break;
            case ComputeBackendKind.Simd:
                TransformSimd(
                    source,
                    destination,
                    expression,
                    program,
                    sourceDescriptor,
                    destinationDescriptor,
                    options);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(options));
        }

        return destination;
    }

    internal static float[] Project<T>(
        T[] source,
        Expression<Func<T, float>> expression,
        ComputeOptions options)
        where T : unmanaged, IComputeValue<T>
    {
        Validate(source, options);
        if (T.ComputeDescriptor.ComponentType != ComputeComponentType.Float32)
        {
            return ComputeValueConversionExecutor.Project(
                source,
                expression,
                options);
        }

        ComputeValueExpressionProgram program =
            ComputeValueExpressionParser.ParseProjection(
                expression,
                T.ComputeDescriptor);
        if (source.Length == 0)
        {
            return [];
        }

        ComputeBackendKind backend = ResolveBackend(source.Length, options);
        if (backend == ComputeBackendKind.Gpu)
        {
            try
            {
                return ProjectGpu(source, program, options);
            }
            catch when (options.Backend == ComputeBackendKind.Auto && options.AllowFallback)
            {
                backend = ResolveCpuBackend(source.Length, options);
            }
        }
        return backend switch
        {
            ComputeBackendKind.Scalar => ProjectScalar(source, expression, options),
            ComputeBackendKind.ParallelCpu => ProjectParallel(source, expression, options),
            ComputeBackendKind.Simd => ProjectSimd(
                source,
                expression,
                program,
                options),
            _ => throw new ArgumentOutOfRangeException(nameof(options))
        };
    }

    internal static T[] Map<T>(
        T[] source,
        Expression<Func<T, T>> expression,
        ComputeOptions options,
        bool inPlace)
        where T : unmanaged, IComputeValue<T>
    {
        Validate(source, options);
        if (T.ComputeDescriptor.ComponentType != ComputeComponentType.Float32)
        {
            return ComputeValueConversionExecutor.Map(
                source,
                expression,
                options,
                inPlace);
        }

        ComputeValueExpressionProgram program =
            ComputeValueExpressionParser.ParseMap(expression, T.ComputeDescriptor);
        if (source.Length == 0)
        {
            return inPlace ? source : [];
        }

        ComputeBackendKind backend = ResolveBackend(source.Length, options);
        if (backend == ComputeBackendKind.Gpu)
        {
            try
            {
                return MapGpu(source, program, options, inPlace);
            }
            catch when (options.Backend == ComputeBackendKind.Auto && options.AllowFallback)
            {
                backend = ResolveCpuBackend(source.Length, options);
            }
        }

        T[] destination = inPlace
            ? source
            : GC.AllocateUninitializedArray<T>(source.Length);
        switch (backend)
        {
            case ComputeBackendKind.Scalar:
                MapScalar(source, destination, expression, options);
                break;
            case ComputeBackendKind.ParallelCpu:
                MapParallel(source, destination, expression, options);
                break;
            case ComputeBackendKind.Simd:
                MapSimd(source, destination, expression, program, options);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(options));
        }

        return destination;
    }

    private static float[] ProjectGpu<T>(
        T[] source,
        ComputeValueExpressionProgram expression,
        ComputeOptions options)
        where T : unmanaged, IComputeValue<T> =>
        GpuComputeBackend.ResolveContext(options).ExecuteCompositeValue(
            MemoryMarshal.Cast<T, float>(source).ToArray(),
            source.Length,
            T.ComputeDescriptor.ComponentCount,
            CompositeGpuProgramCompiler.Compile(expression));

    private static T[] MapGpu<T>(
        T[] source,
        ComputeValueExpressionProgram expression,
        ComputeOptions options,
        bool inPlace)
        where T : unmanaged, IComputeValue<T>
    {
        float[] output = GpuComputeBackend.ResolveContext(options)
            .ExecuteCompositeValue(
                MemoryMarshal.Cast<T, float>(source).ToArray(),
                source.Length,
                T.ComputeDescriptor.ComponentCount,
                CompositeGpuProgramCompiler.Compile(expression));
        if (inPlace)
        {
            MemoryMarshal.Cast<float, T>(output).CopyTo(source);
            return source;
        }

        return MemoryMarshal.Cast<float, T>(output).ToArray();
    }

    private static float[] ProjectScalar<T>(
        T[] source,
        Expression<Func<T, float>> expression,
        ComputeOptions options)
        where T : unmanaged, IComputeValue<T>
    {
        Func<T, float> operation = expression.Compile();
        float[] destination = GC.AllocateUninitializedArray<float>(source.Length);
        for (int index = 0; index < source.Length; index++)
        {
            CheckCancellation(index, options.CancellationToken);
            destination[index] = operation(source[index]);
        }

        return destination;
    }

    private static void TransformScalar<TSource, TDestination>(
        TSource[] source,
        TDestination[] destination,
        Expression<Func<TSource, TDestination>> expression,
        ComputeOptions options)
    {
        Func<TSource, TDestination> operation = expression.Compile();
        for (int index = 0; index < source.Length; index++)
        {
            CheckCancellation(index, options.CancellationToken);
            destination[index] = operation(source[index]);
        }
    }

    private static void TransformParallel<TSource, TDestination>(
        TSource[] source,
        TDestination[] destination,
        Expression<Func<TSource, TDestination>> expression,
        ComputeOptions options)
    {
        Func<TSource, TDestination> operation = expression.Compile();
        RunParallelRanges(
            source.Length,
            options,
            (start, end) =>
            {
                for (int index = start; index < end; index++)
                {
                    destination[index] = operation(source[index]);
                }
            });
    }

    private static void TransformSimd<TSource, TDestination>(
        TSource[] source,
        TDestination[] destination,
        Expression<Func<TSource, TDestination>> expression,
        ComputeValueExpressionProgram program,
        ComputeValueDescriptor<TSource> sourceDescriptor,
        ComputeValueDescriptor<TDestination> destinationDescriptor,
        ComputeOptions options)
        where TSource : unmanaged
        where TDestination : unmanaged
    {
        ReadOnlySpan<float> input = MemoryMarshal.Cast<TSource, float>(source);
        Span<float> output = MemoryMarshal.Cast<TDestination, float>(destination);
        int lanes = Vector<float>.Count;
        int vectorizedLength = source.Length - (source.Length % lanes);
        var components = new Vector<float>[sourceDescriptor.ComponentCount];
        Span<float> gathered =
            stackalloc float[sourceDescriptor.ComponentCount * lanes];
        Span<float> scattered =
            stackalloc float[destinationDescriptor.ComponentCount * lanes];
        for (int index = 0; index < vectorizedLength; index += lanes)
        {
            CheckCancellation(index, options.CancellationToken);
            Gather(
                input,
                index,
                sourceDescriptor.ComponentCount,
                components,
                gathered);
            for (int component = 0;
                 component < destinationDescriptor.ComponentCount;
                 component++)
            {
                Evaluate(program.Outputs[component], components)
                    .CopyTo(scattered.Slice(component * lanes, lanes));
            }

            Scatter(
                output,
                index,
                destinationDescriptor.ComponentCount,
                scattered,
                lanes);
        }

        Func<TSource, TDestination> scalar = expression.Compile();
        for (int index = vectorizedLength; index < source.Length; index++)
        {
            destination[index] = scalar(source[index]);
        }
    }

    private static float[] ProjectParallel<T>(
        T[] source,
        Expression<Func<T, float>> expression,
        ComputeOptions options)
        where T : unmanaged, IComputeValue<T>
    {
        Func<T, float> operation = expression.Compile();
        float[] destination = GC.AllocateUninitializedArray<float>(source.Length);
        RunParallelRanges(
            source.Length,
            options,
            (start, end) =>
            {
                for (int index = start; index < end; index++)
                {
                    destination[index] = operation(source[index]);
                }
            });
        return destination;
    }

    private static float[] ProjectSimd<T>(
        T[] source,
        Expression<Func<T, float>> expression,
        ComputeValueExpressionProgram program,
        ComputeOptions options)
        where T : unmanaged, IComputeValue<T>
    {
        ComputeValueDescriptor<T> descriptor = T.ComputeDescriptor;
        float[] destination = GC.AllocateUninitializedArray<float>(source.Length);
        ReadOnlySpan<float> values = MemoryMarshal.Cast<T, float>(source);
        int lanes = Vector<float>.Count;
        int vectorizedLength = source.Length - (source.Length % lanes);
        var components = new Vector<float>[descriptor.ComponentCount];
        Span<float> gathered = stackalloc float[descriptor.ComponentCount * lanes];

        for (int index = 0; index < vectorizedLength; index += lanes)
        {
            CheckCancellation(index, options.CancellationToken);
            Gather(values, index, descriptor.ComponentCount, components, gathered);
            Vector<float> result = Evaluate(program.Outputs[0], components);
            result.CopyTo(destination, index);
        }

        Func<T, float> scalar = expression.Compile();
        for (int index = vectorizedLength; index < source.Length; index++)
        {
            destination[index] = scalar(source[index]);
        }

        return destination;
    }

    private static void MapScalar<T>(
        T[] source,
        T[] destination,
        Expression<Func<T, T>> expression,
        ComputeOptions options)
        where T : unmanaged, IComputeValue<T>
    {
        Func<T, T> operation = expression.Compile();
        for (int index = 0; index < source.Length; index++)
        {
            CheckCancellation(index, options.CancellationToken);
            destination[index] = operation(source[index]);
        }
    }

    private static void MapParallel<T>(
        T[] source,
        T[] destination,
        Expression<Func<T, T>> expression,
        ComputeOptions options)
        where T : unmanaged, IComputeValue<T>
    {
        Func<T, T> operation = expression.Compile();
        RunParallelRanges(
            source.Length,
            options,
            (start, end) =>
            {
                for (int index = start; index < end; index++)
                {
                    destination[index] = operation(source[index]);
                }
            });
    }

    private static void MapSimd<T>(
        T[] source,
        T[] destination,
        Expression<Func<T, T>> expression,
        ComputeValueExpressionProgram program,
        ComputeOptions options)
        where T : unmanaged, IComputeValue<T>
    {
        ComputeValueDescriptor<T> descriptor = T.ComputeDescriptor;
        ReadOnlySpan<float> input = MemoryMarshal.Cast<T, float>(source);
        Span<float> output = MemoryMarshal.Cast<T, float>(destination);
        int lanes = Vector<float>.Count;
        int vectorizedLength = source.Length - (source.Length % lanes);
        var components = new Vector<float>[descriptor.ComponentCount];
        Span<float> gathered = stackalloc float[descriptor.ComponentCount * lanes];
        Span<float> scattered = stackalloc float[descriptor.ComponentCount * lanes];

        for (int index = 0; index < vectorizedLength; index += lanes)
        {
            CheckCancellation(index, options.CancellationToken);
            Gather(input, index, descriptor.ComponentCount, components, gathered);
            for (int component = 0; component < descriptor.ComponentCount; component++)
            {
                Evaluate(program.Outputs[component], components)
                    .CopyTo(scattered.Slice(component * lanes, lanes));
            }

            Scatter(output, index, descriptor.ComponentCount, scattered, lanes);
        }

        Func<T, T> scalar = expression.Compile();
        for (int index = vectorizedLength; index < source.Length; index++)
        {
            destination[index] = scalar(source[index]);
        }
    }

    private static void Gather(
        ReadOnlySpan<float> input,
        int pixelIndex,
        int componentCount,
        Vector<float>[] components,
        Span<float> gathered)
    {
        int lanes = Vector<float>.Count;
        int baseOffset = checked(pixelIndex * componentCount);
        if (componentCount == 1)
        {
            components[0] = new Vector<float>(input.Slice(baseOffset, lanes));
            return;
        }
        if (componentCount == 2 && lanes == 8 && Avx2.IsSupported)
        {
            PackedComponentKernels.DeinterleaveFloat2(
                input,
                pixelIndex,
                gathered.Slice(0, lanes),
                gathered.Slice(lanes, lanes));
            components[0] = new Vector<float>(gathered.Slice(0, lanes));
            components[1] = new Vector<float>(gathered.Slice(lanes, lanes));
            return;
        }
        if (componentCount == 3 && lanes == 8 && Avx2.IsSupported)
        {
            PackedComponentKernels.DeinterleaveFloat3(
                input,
                pixelIndex,
                gathered.Slice(0, lanes),
                gathered.Slice(lanes, lanes),
                gathered.Slice(lanes * 2, lanes));
            components[0] = new Vector<float>(gathered.Slice(0, lanes));
            components[1] = new Vector<float>(gathered.Slice(lanes, lanes));
            components[2] = new Vector<float>(gathered.Slice(lanes * 2, lanes));
            return;
        }
        for (int component = 0; component < componentCount; component++)
        {
            Span<float> componentValues = gathered.Slice(component * lanes, lanes);
            for (int lane = 0; lane < lanes; lane++)
            {
                componentValues[lane] =
                    input[baseOffset + (lane * componentCount) + component];
            }

            components[component] = new Vector<float>(componentValues);
        }
    }

    private static void Scatter(
        Span<float> output,
        int pixelIndex,
        int componentCount,
        ReadOnlySpan<float> components,
        int lanes)
    {
        int baseOffset = checked(pixelIndex * componentCount);
        if (componentCount == 1)
        {
            components[..lanes].CopyTo(output.Slice(baseOffset, lanes));
            return;
        }
        if (componentCount == 2 && lanes == 8 && Avx2.IsSupported)
        {
            PackedComponentKernels.InterleaveFloat2(
                components.Slice(0, lanes),
                components.Slice(lanes, lanes),
                output,
                pixelIndex);
            return;
        }
        if (componentCount == 3 && lanes == 8 && Avx2.IsSupported)
        {
            PackedComponentKernels.InterleaveFloat3(
                components.Slice(0, lanes),
                components.Slice(lanes, lanes),
                components.Slice(lanes * 2, lanes),
                output,
                pixelIndex);
            return;
        }
        for (int component = 0; component < componentCount; component++)
        {
            ReadOnlySpan<float> componentValues =
                components.Slice(component * lanes, lanes);
            for (int lane = 0; lane < lanes; lane++)
            {
                output[baseOffset + (lane * componentCount) + component] =
                    componentValues[lane];
            }
        }
    }

    private static Vector<float> Evaluate(
        ComputeValueExpressionNode node,
        IReadOnlyList<Vector<float>> components) => node switch
        {
            ComputeValueComponentNode component => components[component.Index],
            ComputeValueConstantNode constant => new Vector<float>(constant.Value),
            ComputeValueNegateNode negate => -Evaluate(negate.Operand, components),
            ComputeValueBinaryNode binary => Apply(
                binary.Operation,
                Evaluate(binary.Left, components),
                Evaluate(binary.Right, components)),
            _ => throw new NotSupportedException(
                $"Unknown composite expression node '{node.GetType().Name}'.")
        };

    private static Vector<float> Apply(
        ComputeValueBinaryOperation operation,
        Vector<float> left,
        Vector<float> right) => operation switch
        {
            ComputeValueBinaryOperation.Add => left + right,
            ComputeValueBinaryOperation.Subtract => left - right,
            ComputeValueBinaryOperation.Multiply => left * right,
            ComputeValueBinaryOperation.Divide => left / right,
            _ => throw new ArgumentOutOfRangeException(nameof(operation))
        };

    private static void RunParallelRanges(
        int length,
        ComputeOptions options,
        Action<int, int> operation)
    {
        if (length == 0)
        {
            return;
        }

        int workerCount = Math.Min(
            length,
            options.MaxDegreeOfParallelism ?? Environment.ProcessorCount);
        int rangeSize = (length + workerCount - 1) / workerCount;
        Parallel.For(
            0,
            workerCount,
            new ParallelOptions
            {
                CancellationToken = options.CancellationToken,
                MaxDegreeOfParallelism = workerCount
            },
            worker =>
            {
                int start = worker * rangeSize;
                int end = Math.Min(start + rangeSize, length);
                operation(start, end);
            });
    }

    private static ComputeBackendKind ResolveBackend(
        int length,
        ComputeOptions options)
    {
        if (options.Backend != ComputeBackendKind.Auto)
        {
            return options.Backend;
        }

        if (length >= options.Thresholds.GpuSimpleThreshold &&
            (options.GpuContext is not null || GpuComputeBackend.HasHardwareAccelerator))
        {
            return ComputeBackendKind.Gpu;
        }

        return ResolveCpuBackend(length, options);
    }

    private static ComputeBackendKind ResolveCpuBackend(
        int length,
        ComputeOptions options)
    {
        if (length >= options.Thresholds.ParallelThreshold &&
            Environment.ProcessorCount > 1)
        {
            return ComputeBackendKind.ParallelCpu;
        }

        return Vector.IsHardwareAccelerated &&
               length >= options.Thresholds.SimdThreshold
            ? ComputeBackendKind.Simd
            : ComputeBackendKind.Scalar;
    }

    private static void Validate<T>(T[] source, ComputeOptions options)
        where T : unmanaged, IComputeValue<T>
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

        if (options.GpuContext is not null &&
            options.PreferredGpuAcceleratorIndex is not null)
        {
            throw new ArgumentException(
                "GpuContext and PreferredGpuAcceleratorIndex cannot be used together.",
                nameof(options));
        }

        options.CancellationToken.ThrowIfCancellationRequested();
        _ = T.ComputeDescriptor;
    }

    private static void CheckCancellation(
        int index,
        CancellationToken cancellationToken)
    {
        if ((index & 0xFFF) == 0)
        {
            cancellationToken.ThrowIfCancellationRequested();
        }
    }
}
