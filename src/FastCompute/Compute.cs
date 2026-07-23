using System.Diagnostics;
using System.Linq.Expressions;
using FastCompute.Backends;
using FastCompute.Backends.ParallelCpu;
using FastCompute.Backends.Scalar;
using FastCompute.Backends.Gpu;
using FastCompute.Backends.Simd;
using FastCompute.Diagnostics;
using FastCompute.Expressions;

namespace FastCompute;

/// <summary>
/// Provides one-shot array computation operations.
/// </summary>
public static class Compute
{
    /// <summary>
    /// Applies an expression to every element of an array.
    /// </summary>
    /// <param name="source">The input array.</param>
    /// <param name="expression">The expression applied to each element.</param>
    /// <param name="options">Optional execution settings.</param>
    /// <returns>A new array containing the computed values.</returns>
    public static float[] Run(
        float[] source,
        Expression<Func<float, float>> expression,
        ComputeOptions? options = null) =>
        RunCore(source, expression, options, collectDiagnostics: false, out _);

    /// <summary>
    /// Applies an expression and returns the result together with execution diagnostics.
    /// </summary>
    /// <param name="source">The input array.</param>
    /// <param name="expression">The expression applied to each element.</param>
    /// <param name="options">Optional execution settings.</param>
    /// <returns>The computed array and collected diagnostics.</returns>
    public static ComputeResult<float[]> RunWithDiagnostics(
        float[] source,
        Expression<Func<float, float>> expression,
        ComputeOptions? options = null)
    {
        float[] value = RunCore(
            source,
            expression,
            options,
            collectDiagnostics: true,
            out ComputeDiagnostics? diagnostics);

        return new ComputeResult<float[]>(value, diagnostics!);
    }

    /// <summary>
    /// Applies a binary expression to corresponding elements of two arrays.
    /// </summary>
    /// <param name="left">The first input array.</param>
    /// <param name="right">The second input array.</param>
    /// <param name="expression">The expression applied to each pair of elements.</param>
    /// <param name="options">Optional execution settings.</param>
    /// <returns>A new array containing the computed values.</returns>
    /// <exception cref="ArgumentException">The input arrays have different lengths.</exception>
    public static float[] Zip(
        float[] left,
        float[] right,
        Expression<Func<float, float, float>> expression,
        ComputeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        ArgumentNullException.ThrowIfNull(expression);

        if (left.Length != right.Length)
        {
            throw new ArgumentException(
                $"Zip requires arrays of equal length, but received {left.Length} and {right.Length}.",
                nameof(right));
        }

        ComputeOptions effectiveOptions = ValidateOptions(options);
        effectiveOptions.CancellationToken.ThrowIfCancellationRequested();
        ComputeExpressionPlan plan = CreatePlan(expression, effectiveOptions);
        IComputeBackend backend = ResolveBackend(effectiveOptions, plan, left.Length);
        var context = CreateExecutionContext(effectiveOptions, collectDiagnostics: false);

        return backend.ExecuteZip(left, right, plan, context).Value;
    }

    /// <summary>
    /// Computes the sum of all elements in an array.
    /// </summary>
    /// <param name="source">The input array.</param>
    /// <param name="options">Optional execution settings.</param>
    /// <returns>The element sum, or zero when the array is empty.</returns>
    public static float Sum(float[] source, ComputeOptions? options = null)
        => Reduce(source, ComputeReductionKind.Sum, options);

    /// <summary>Computes the minimum element in an array.</summary>
    /// <exception cref="InvalidOperationException">The input array is empty.</exception>
    public static float Min(float[] source, ComputeOptions? options = null)
        => Reduce(source, ComputeReductionKind.Min, options);

    /// <summary>Computes the maximum element in an array.</summary>
    /// <exception cref="InvalidOperationException">The input array is empty.</exception>
    public static float Max(float[] source, ComputeOptions? options = null)
        => Reduce(source, ComputeReductionKind.Max, options);

    /// <summary>Computes the arithmetic mean of an array.</summary>
    /// <exception cref="InvalidOperationException">The input array is empty.</exception>
    public static float Average(float[] source, ComputeOptions? options = null)
        => Reduce(source, ComputeReductionKind.Average, options);

    private static float Reduce(
        float[] source,
        ComputeReductionKind reduction,
        ComputeOptions? options)
    {
        ArgumentNullException.ThrowIfNull(source);

        if (source.Length == 0 && reduction != ComputeReductionKind.Sum)
        {
            throw new InvalidOperationException(
                $"Cannot compute {reduction} for an empty array.");
        }

        ComputeOptions effectiveOptions = ValidateOptions(options);
        effectiveOptions.CancellationToken.ThrowIfCancellationRequested();
        IComputeBackend backend = ResolveBackend(effectiveOptions, plan: null, source.Length);
        var context = CreateExecutionContext(effectiveOptions, collectDiagnostics: false);
        return backend.Reduce(source, reduction, context).Value;
    }

    private static float[] RunCore(
        float[] source,
        Expression<Func<float, float>> expression,
        ComputeOptions? options,
        bool collectDiagnostics,
        out ComputeDiagnostics? diagnostics)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(expression);

        long planningStarted = collectDiagnostics ? Stopwatch.GetTimestamp() : 0L;
        ComputeOptions effectiveOptions = ValidateOptions(options);
        effectiveOptions.CancellationToken.ThrowIfCancellationRequested();
        ComputeExpressionPlan plan = CreatePlan(expression, effectiveOptions);
        IComputeBackend backend = ResolveBackend(effectiveOptions, plan, source.Length);
        TimeSpan planningTime = collectDiagnostics
            ? Stopwatch.GetElapsedTime(planningStarted)
            : TimeSpan.Zero;

        var context = CreateExecutionContext(effectiveOptions, collectDiagnostics);
        ComputeBackendExecution<float[]> execution =
            backend.ExecuteMap(source, plan, context);

        diagnostics = collectDiagnostics
            ? new ComputeDiagnostics(
                backend.Kind,
                planningTime,
                execution.CompilationTime,
                execution.UploadTime,
                execution.ExecutionTime,
                execution.DownloadTime,
                execution.KernelCacheHit,
                execution.DeviceName)
            : null;

        return execution.Value;
    }

    private static ComputeOptions ValidateOptions(ComputeOptions? options)
    {
        ComputeOptions result = options ?? ComputeOptions.Default;

        if (result.MaxDegreeOfParallelism is <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                result.MaxDegreeOfParallelism,
                "MaxDegreeOfParallelism must be greater than zero.");
        }

        ArgumentNullException.ThrowIfNull(result.Thresholds);
        ValidateThreshold(result.Thresholds.SimdThreshold, nameof(ComputeThresholdOptions.SimdThreshold));
        ValidateThreshold(result.Thresholds.ParallelThreshold, nameof(ComputeThresholdOptions.ParallelThreshold));
        ValidateThreshold(result.Thresholds.GpuSimpleThreshold, nameof(ComputeThresholdOptions.GpuSimpleThreshold));
        ValidateThreshold(result.Thresholds.GpuMediumThreshold, nameof(ComputeThresholdOptions.GpuMediumThreshold));
        ValidateThreshold(result.Thresholds.GpuHeavyThreshold, nameof(ComputeThresholdOptions.GpuHeavyThreshold));

        return result;
    }

    private static void ValidateThreshold(int threshold, string propertyName)
    {
        if (threshold < 0)
        {
            throw new ArgumentOutOfRangeException(
                propertyName,
                threshold,
                "Compute thresholds cannot be negative.");
        }
    }

    private static ComputeExecutionContext CreateExecutionContext(
        ComputeOptions options,
        bool collectDiagnostics) =>
        new(
            options.CancellationToken,
            options.MaxDegreeOfParallelism,
            collectDiagnostics,
            options.GpuContext);

    private static ComputeExpressionPlan CreatePlan(
        LambdaExpression expression,
        ComputeOptions options)
    {
        ComputeExpressionPlan plan = ComputeExpressionParser.Parse(expression);

        return options.OptimizationMode switch
        {
            ComputeOptimizationMode.Strict => StrictComputeOptimizer.Optimize(plan),
            ComputeOptimizationMode.Fast => throw new NotSupportedException(
                "Fast optimization mode is reserved for a later implementation stage."),
            _ => throw new ArgumentOutOfRangeException(
                nameof(options),
                options.OptimizationMode,
                "Unknown optimization mode.")
        };
    }

    private static IComputeBackend ResolveBackend(
        ComputeOptions options,
        ComputeExpressionPlan? plan,
        int elementCount)
    {
        IComputeBackend backend = options.Backend switch
        {
            ComputeBackendKind.Auto => SelectAutomaticBackend(options, plan, elementCount),
            ComputeBackendKind.Scalar => ScalarComputeBackend.Instance,
            ComputeBackendKind.ParallelCpu => ParallelComputeBackend.Instance,
            ComputeBackendKind.Simd => SimdComputeBackend.Instance,
            ComputeBackendKind.Gpu => GpuComputeBackend.Instance,
            _ => throw new ComputeBackendUnavailableException(options.Backend)
        };

        if (!backend.IsAvailable || (plan is not null && !backend.Supports(plan)))
        {
            if (options.Backend == ComputeBackendKind.Auto && options.AllowFallback)
            {
                return ScalarComputeBackend.Instance;
            }

            throw new ComputeBackendUnavailableException(backend.Kind);
        }

        return backend;
    }

    private static IComputeBackend SelectAutomaticBackend(
        ComputeOptions options,
        ComputeExpressionPlan? plan,
        int elementCount)
    {
        int gpuThreshold =
            ComputeExpressionClassifier.GetGpuThreshold(plan, options.Thresholds);
        if (elementCount >= gpuThreshold &&
            (options.GpuContext is not null ||
             GpuComputeBackend.HasHardwareAccelerator))
        {
            return GpuComputeBackend.Instance;
        }

        bool simdSupported =
            SimdComputeBackend.Instance.IsAvailable &&
            (plan is null || SimdComputeBackend.Instance.Supports(plan));

        if (simdSupported &&
            elementCount >= options.Thresholds.SimdThreshold)
        {
            return SimdComputeBackend.Instance;
        }

        int availableParallelism =
            options.MaxDegreeOfParallelism ?? Environment.ProcessorCount;

        return availableParallelism > 1 &&
               elementCount >= options.Thresholds.ParallelThreshold
            ? ParallelComputeBackend.Instance
            : ScalarComputeBackend.Instance;
    }
}
