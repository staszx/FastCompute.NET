using System.Linq.Expressions;
using System.Numerics;
using FastCompute.Backends.Cpu;
using FastCompute.Backends.Gpu;
using FastCompute.Backends.Simd;
using FastCompute.Expressions;

namespace FastCompute;

public static partial class Compute
{
    /// <summary>Applies a double or integer expression to every element.</summary>
    public static T[] Run<T>(
        T[] source,
        Expression<Func<T, T>> expression,
        ComputeOptions? options = null)
        where T : unmanaged, INumber<T> =>
        RunNumeric(source, expression, options);

    /// <summary>
    /// Applies a double or integer expression through the task-compatible API.
    /// </summary>
    public static Task<T[]> RunAsync<T>(
        T[] source,
        Expression<Func<T, T>> expression,
        ComputeOptions? options = null)
        where T : unmanaged, INumber<T> =>
        Task.FromResult(Run(source, expression, options));

    /// <summary>Applies a double or integer expression in place.</summary>
    public static T[] RunInPlace<T>(
        T[] source,
        Expression<Func<T, T>> expression,
        ComputeOptions? options = null)
        where T : unmanaged, INumber<T> =>
        RunNumericInPlace(source, expression, options);

    /// <summary>Zips two double or integer arrays.</summary>
    public static T[] Zip<T>(
        T[] left,
        T[] right,
        Expression<Func<T, T, T>> expression,
        ComputeOptions? options = null)
        where T : unmanaged, INumber<T> =>
        ZipNumeric(left, right, expression, options);

    /// <summary>Zips into the first double or integer array.</summary>
    public static T[] ZipInPlace<T>(
        T[] target,
        T[] right,
        Expression<Func<T, T, T>> expression,
        ComputeOptions? options = null)
        where T : unmanaged, INumber<T> =>
        ZipNumericInPlace(target, right, expression, options);

    /// <summary>Computes the sum of a double or integer array.</summary>
    public static T Sum<T>(
        T[] source,
        ComputeOptions? options = null)
        where T : unmanaged, INumber<T> =>
        ReduceNumeric(source, ComputeReductionKind.Sum, options);

    /// <summary>Computes the minimum of a double or integer array.</summary>
    public static T Min<T>(
        T[] source,
        ComputeOptions? options = null)
        where T : unmanaged, INumber<T> =>
        ReduceNumeric(source, ComputeReductionKind.Min, options);

    /// <summary>Computes the maximum of a double or integer array.</summary>
    public static T Max<T>(
        T[] source,
        ComputeOptions? options = null)
        where T : unmanaged, INumber<T> =>
        ReduceNumeric(source, ComputeReductionKind.Max, options);

    /// <summary>
    /// Computes the average of a double or integer array. Integer averages use
    /// normal truncating integer division.
    /// </summary>
    public static T Average<T>(
        T[] source,
        ComputeOptions? options = null)
        where T : unmanaged, INumber<T> =>
        ReduceNumeric(source, ComputeReductionKind.Average, options);

    internal static T ReduceMapped<T>(
        T[] source,
        Expression<Func<T, T>> expression,
        ComputeReductionKind reduction,
        ComputeOptions? options)
        where T : unmanaged, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(expression);
        ValidateNumericType<T>();
        ComputeOptions effective = ValidateOptions(options);
        effective.CancellationToken.ThrowIfCancellationRequested();
        NumericExpressionProgram<T> program =
            NumericExpressionParser.Parse<T>(expression);
        if (source.Length == 0)
        {
            if (reduction == ComputeReductionKind.Sum)
            {
                return T.Zero;
            }

            throw new InvalidOperationException(
                $"{reduction} is not defined for an empty array.");
        }

        ComputeBackendKind backend = ResolveNumericBackend(
            effective,
            program,
            source.Length);
        return backend switch
        {
            ComputeBackendKind.Scalar => NumericCpuExecutor.ReduceMappedScalar(
                source,
                program,
                reduction,
                effective.CancellationToken),
            ComputeBackendKind.ParallelCpu =>
                NumericCpuExecutor.ReduceMappedParallel(
                    source,
                    program,
                    reduction,
                    effective),
            ComputeBackendKind.Simd => NumericSimdExecutor.ReduceMapped(
                source,
                program,
                reduction,
                effective.CancellationToken),
            ComputeBackendKind.Gpu => ResolveNumericGpuContext(effective)
                .ExecuteNumericMappedReduction(
                    source,
                    program,
                    reduction,
                    effective),
            _ => throw new InvalidOperationException(
                $"Unexpected numeric backend '{backend}'.")
        };
    }

    internal static T ReduceZipped<T>(
        T[] left,
        T[] right,
        Expression<Func<T, T, T>> expression,
        ComputeReductionKind reduction,
        ComputeOptions? options)
        where T : unmanaged, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        ArgumentNullException.ThrowIfNull(expression);
        ValidateNumericType<T>();
        if (left.Length != right.Length)
        {
            throw new ArgumentException(
                $"Zip requires arrays of equal length, but received " +
                $"{left.Length} and {right.Length}.",
                nameof(right));
        }

        ComputeOptions effective = ValidateOptions(options);
        effective.CancellationToken.ThrowIfCancellationRequested();
        NumericExpressionProgram<T> program =
            NumericExpressionParser.Parse<T>(expression);
        if (left.Length == 0)
        {
            if (reduction == ComputeReductionKind.Sum)
            {
                return T.Zero;
            }

            throw new InvalidOperationException(
                $"{reduction} is not defined for an empty array.");
        }

        ComputeBackendKind backend = ResolveNumericBackend(
            effective,
            program,
            left.Length);
        return backend switch
        {
            ComputeBackendKind.Scalar => NumericCpuExecutor.ReduceZippedScalar(
                left,
                right,
                program,
                reduction,
                effective.CancellationToken),
            ComputeBackendKind.ParallelCpu =>
                NumericCpuExecutor.ReduceZippedParallel(
                    left,
                    right,
                    program,
                    reduction,
                    effective),
            ComputeBackendKind.Simd => NumericSimdExecutor.ReduceZipped(
                left,
                right,
                program,
                reduction,
                effective.CancellationToken),
            ComputeBackendKind.Gpu => ResolveNumericGpuContext(effective)
                .ExecuteNumericZippedReduction(
                    left,
                    right,
                    program,
                    reduction,
                    effective),
            _ => throw new InvalidOperationException(
                $"Unexpected numeric backend '{backend}'.")
        };
    }

    private static T[] RunNumeric<T>(
        T[] source,
        Expression<Func<T, T>> expression,
        ComputeOptions? options)
        where T : unmanaged, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(expression);
        ValidateNumericType<T>();
        ComputeOptions effective = ValidateOptions(options);
        effective.CancellationToken.ThrowIfCancellationRequested();
        NumericExpressionProgram<T> program =
            NumericExpressionParser.Parse<T>(expression);
        ComputeBackendKind backend = ResolveNumericBackend(
            effective,
            program,
            source.Length);
        return backend switch
        {
            ComputeBackendKind.Scalar =>
                NumericCpuExecutor.MapScalar(
                    source,
                    program,
                    effective.CancellationToken),
            ComputeBackendKind.ParallelCpu =>
                NumericCpuExecutor.MapParallel(
                    source,
                    program,
                    effective),
            ComputeBackendKind.Simd =>
                NumericSimdExecutor.Map(
                    source,
                    program,
                    effective.CancellationToken),
            ComputeBackendKind.Gpu =>
                ResolveNumericGpuContext(effective)
                    .ExecuteNumericMap(source, program, effective),
            _ => throw new InvalidOperationException(
                $"Unexpected numeric backend '{backend}'.")
        };
    }

    private static T[] RunNumericInPlace<T>(
        T[] source,
        Expression<Func<T, T>> expression,
        ComputeOptions? options)
        where T : unmanaged, INumber<T>
    {
        T[] result = RunNumeric(source, expression, options);
        result.CopyTo(source, 0);
        return source;
    }

    private static T[] ZipNumeric<T>(
        T[] left,
        T[] right,
        Expression<Func<T, T, T>> expression,
        ComputeOptions? options)
        where T : unmanaged, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        ArgumentNullException.ThrowIfNull(expression);
        ValidateNumericType<T>();
        if (left.Length != right.Length)
        {
            throw new ArgumentException(
                $"Zip requires arrays of equal length, but received " +
                $"{left.Length} and {right.Length}.",
                nameof(right));
        }

        ComputeOptions effective = ValidateOptions(options);
        effective.CancellationToken.ThrowIfCancellationRequested();
        NumericExpressionProgram<T> program =
            NumericExpressionParser.Parse<T>(expression);
        ComputeBackendKind backend = ResolveNumericBackend(
            effective,
            program,
            left.Length);
        return backend switch
        {
            ComputeBackendKind.Scalar =>
                NumericCpuExecutor.ZipScalar(
                    left,
                    right,
                    program,
                    effective.CancellationToken),
            ComputeBackendKind.ParallelCpu =>
                NumericCpuExecutor.ZipParallel(
                    left,
                    right,
                    program,
                    effective),
            ComputeBackendKind.Simd =>
                NumericSimdExecutor.Zip(
                    left,
                    right,
                    program,
                    effective.CancellationToken),
            ComputeBackendKind.Gpu =>
                ResolveNumericGpuContext(effective)
                    .ExecuteNumericZip(left, right, program, effective),
            _ => throw new InvalidOperationException(
                $"Unexpected numeric backend '{backend}'.")
        };
    }

    private static T[] ZipNumericInPlace<T>(
        T[] target,
        T[] right,
        Expression<Func<T, T, T>> expression,
        ComputeOptions? options)
        where T : unmanaged, INumber<T>
    {
        T[] result = ZipNumeric(target, right, expression, options);
        result.CopyTo(target, 0);
        return target;
    }

    private static T ReduceNumeric<T>(
        T[] source,
        ComputeReductionKind reduction,
        ComputeOptions? options)
        where T : unmanaged, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(source);
        ValidateNumericType<T>();
        if (source.Length == 0)
        {
            if (reduction == ComputeReductionKind.Sum)
            {
                return T.Zero;
            }

            throw new InvalidOperationException(
                $"{reduction} is not defined for an empty array.");
        }

        ComputeOptions effective = ValidateOptions(options);
        effective.CancellationToken.ThrowIfCancellationRequested();
        ComputeBackendKind backend = ResolveNumericReductionBackend(
            effective,
            source.Length);
        return backend switch
        {
            ComputeBackendKind.Scalar =>
                NumericCpuExecutor.ReduceScalar(
                    source,
                    reduction,
                    effective.CancellationToken),
            ComputeBackendKind.ParallelCpu =>
                NumericCpuExecutor.ReduceParallel(
                    source,
                    reduction,
                    effective),
            ComputeBackendKind.Simd =>
                NumericSimdExecutor.Reduce(
                    source,
                    reduction,
                    effective.CancellationToken),
            ComputeBackendKind.Gpu =>
                ResolveNumericGpuContext(effective)
                    .ExecuteNumericReduction(
                        source,
                        reduction,
                        effective),
            _ => throw new InvalidOperationException(
                $"Unexpected numeric backend '{backend}'.")
        };
    }

    private static ComputeBackendKind ResolveNumericBackend<T>(
        ComputeOptions options,
        NumericExpressionProgram<T> program,
        int length)
        where T : unmanaged, INumber<T>
    {
        if (options.Backend != ComputeBackendKind.Auto)
        {
            return ValidateExplicitNumericBackend(
                options.Backend,
                program);
        }

        if (ShouldUseNumericGpu(options, program, length))
        {
            return ComputeBackendKind.Gpu;
        }

        if (length >= options.Thresholds.ParallelThreshold)
        {
            return ComputeBackendKind.ParallelCpu;
        }

        if (length >= options.Thresholds.SimdThreshold &&
            NumericSimdExecutor.Supports(program))
        {
            return ComputeBackendKind.Simd;
        }

        return ComputeBackendKind.Scalar;
    }

    private static ComputeBackendKind ResolveNumericReductionBackend(
        ComputeOptions options,
        int length)
    {
        if (options.Backend != ComputeBackendKind.Auto)
        {
            if (options.Backend == ComputeBackendKind.Simd &&
                !Vector.IsHardwareAccelerated)
            {
                throw new ComputeBackendUnavailableException(
                    ComputeBackendKind.Simd);
            }

            return options.Backend;
        }

        if (length >= options.Thresholds.ParallelThreshold)
        {
            return ComputeBackendKind.ParallelCpu;
        }

        return length >= options.Thresholds.SimdThreshold &&
               Vector.IsHardwareAccelerated
            ? ComputeBackendKind.Simd
            : ComputeBackendKind.Scalar;
    }

    private static ComputeBackendKind ValidateExplicitNumericBackend<T>(
        ComputeBackendKind backend,
        NumericExpressionProgram<T> program)
        where T : unmanaged, INumber<T>
    {
        if (backend == ComputeBackendKind.Simd &&
            !NumericSimdExecutor.Supports(program))
        {
            if (!NumericSimdExecutor.IsAvailable<T>())
            {
                throw new ComputeBackendUnavailableException(backend);
            }

            throw new ComputeBackendNotSupportedException(
                backend,
                $"the requested {typeof(T).Name} expression",
                "Scalar, ParallelCpu, or Gpu");
        }

        return backend;
    }

    private static bool ShouldUseNumericGpu<T>(
        ComputeOptions options,
        NumericExpressionProgram<T> program,
        int length)
        where T : unmanaged
    {
        bool heavy = program.Instructions.Any(
            instruction => instruction.OpCode is
                NumericOpCode.Sqrt or
                NumericOpCode.Sin or
                NumericOpCode.Cos or
                NumericOpCode.Tan or
                NumericOpCode.Exp or
                NumericOpCode.Log or
                NumericOpCode.Log10 or
                NumericOpCode.Pow);
        int threshold = heavy
            ? options.Thresholds.GpuHeavyThreshold
            : options.Thresholds.GpuSimpleThreshold;
        if (length < threshold)
        {
            return false;
        }

        if (options.GpuContext is not null)
        {
            return options.GpuContext.MemoryLocation ==
                ComputeMemoryLocation.Device;
        }

        return GpuComputeBackend.Instance.IsAvailable;
    }

    private static ComputeContext ResolveNumericGpuContext(
        ComputeOptions options) =>
        GpuComputeBackend.ResolveContext(
            CreateExecutionContext(options, collectDiagnostics: false));

    private static void ValidateNumericType<T>()
        where T : unmanaged
    {
        if (typeof(T) != typeof(double) && typeof(T) != typeof(int))
        {
            throw new NotSupportedException(
                $"Typed numeric execution supports double and int, not " +
                $"'{typeof(T).Name}'.");
        }
    }
}
