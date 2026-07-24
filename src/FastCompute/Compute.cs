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
    /// Applies an expression to every element and stores each result back into
    /// the source array.
    /// </summary>
    /// <remarks>
    /// If execution is cancelled or a backend operation fails, elements already
    /// processed are not rolled back.
    /// </remarks>
    /// <param name="source">The array to read and overwrite.</param>
    /// <param name="expression">The expression applied to each element.</param>
    /// <param name="options">Optional execution settings.</param>
    /// <returns>The same array instance supplied in <paramref name="source"/>.</returns>
    public static float[] RunInPlace(
        float[] source,
        Expression<Func<float, float>> expression,
        ComputeOptions? options = null) =>
        RunInPlaceCore(
            source,
            expression,
            options,
            collectDiagnostics: false,
            out _);

    /// <summary>
    /// Applies an expression in place and returns execution diagnostics.
    /// </summary>
    /// <remarks>
    /// If execution is cancelled or a backend operation fails, elements already
    /// processed are not rolled back.
    /// </remarks>
    /// <param name="source">The array to read and overwrite.</param>
    /// <param name="expression">The expression applied to each element.</param>
    /// <param name="options">Optional execution settings.</param>
    /// <returns>The source array and collected diagnostics.</returns>
    public static ComputeResult<float[]> RunInPlaceWithDiagnostics(
        float[] source,
        Expression<Func<float, float>> expression,
        ComputeOptions? options = null)
    {
        float[] value = RunInPlaceCore(
            source,
            expression,
            options,
            collectDiagnostics: true,
            out ComputeDiagnostics? diagnostics);

        return new ComputeResult<float[]>(value, diagnostics!);
    }

    /// <summary>
    /// Applies an arbitrary user delegate to every element using a CPU backend.
    /// </summary>
    /// <remarks>
    /// The delegate is executed directly and is not converted to the compute IR.
    /// Auto selects only Scalar or Parallel CPU for this operation.
    /// </remarks>
    /// <param name="source">The input array.</param>
    /// <param name="operation">The user operation applied to each element.</param>
    /// <param name="options">Optional execution settings.</param>
    /// <returns>A new array containing the computed values.</returns>
    public static float[] RunDelegate(
        float[] source,
        Func<float, float> operation,
        ComputeOptions? options = null) =>
        RunDelegateCore(
            source,
            operation,
            options,
            collectDiagnostics: false,
            out _);

    /// <summary>
    /// Applies an arbitrary user delegate using a CPU backend and returns
    /// execution diagnostics.
    /// </summary>
    /// <param name="source">The input array.</param>
    /// <param name="operation">The user operation applied to each element.</param>
    /// <param name="options">Optional execution settings.</param>
    /// <returns>The computed array and collected diagnostics.</returns>
    public static ComputeResult<float[]> RunDelegateWithDiagnostics(
        float[] source,
        Func<float, float> operation,
        ComputeOptions? options = null)
    {
        float[] value = RunDelegateCore(
            source,
            operation,
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
        IComputeBackend backend =
            ResolveBackend(effectiveOptions, plan, left.Length).Backend;
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
        IComputeBackend backend =
            ResolveBackend(effectiveOptions, plan: null, source.Length).Backend;
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
        BackendResolution resolution =
            ResolveBackend(effectiveOptions, plan, source.Length);
        IComputeBackend backend = resolution.Backend;
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
            {
                BackendSelectionReason = resolution.Reason,
                EstimatedGpuMemoryBytes = resolution.EstimatedGpuMemoryBytes,
                GpuMemoryBudgetBytes = resolution.GpuMemoryBudgetBytes
            }
            : null;

        return execution.Value;
    }

    private static float[] RunDelegateCore(
        float[] source,
        Func<float, float> operation,
        ComputeOptions? options,
        bool collectDiagnostics,
        out ComputeDiagnostics? diagnostics)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(operation);

        long planningStarted = collectDiagnostics ? Stopwatch.GetTimestamp() : 0L;
        ComputeOptions effectiveOptions = ValidateOptions(options);
        effectiveOptions.CancellationToken.ThrowIfCancellationRequested();
        BackendResolution resolution =
            ResolveDelegateBackend(effectiveOptions, source.Length);
        TimeSpan planningTime = collectDiagnostics
            ? Stopwatch.GetElapsedTime(planningStarted)
            : TimeSpan.Zero;
        var context =
            CreateExecutionContext(effectiveOptions, collectDiagnostics);
        ComputeBackendExecution<float[]> execution =
            resolution.Backend.Kind switch
            {
                ComputeBackendKind.Scalar =>
                    ScalarComputeBackend.Instance.ExecuteDelegateMap(
                        source,
                        operation,
                        context),
                ComputeBackendKind.ParallelCpu =>
                    ParallelComputeBackend.Instance.ExecuteDelegateMap(
                        source,
                        operation,
                        context),
                _ => throw new InvalidOperationException(
                    "Delegate backend resolution returned a non-CPU backend.")
            };

        diagnostics = collectDiagnostics
            ? new ComputeDiagnostics(
                resolution.Backend.Kind,
                planningTime,
                execution.CompilationTime,
                execution.UploadTime,
                execution.ExecutionTime,
                execution.DownloadTime,
                execution.KernelCacheHit,
                execution.DeviceName)
            {
                BackendSelectionReason = resolution.Reason
            }
            : null;

        return execution.Value;
    }

    private static float[] RunInPlaceCore(
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
        BackendResolution resolution =
            ResolveInPlaceBackend(effectiveOptions, plan, source.Length);
        TimeSpan planningTime = collectDiagnostics
            ? Stopwatch.GetElapsedTime(planningStarted)
            : TimeSpan.Zero;
        var context =
            CreateExecutionContext(effectiveOptions, collectDiagnostics);
        ComputeBackendExecution<float[]> execution =
            resolution.Backend.Kind switch
            {
                ComputeBackendKind.Scalar =>
                    ScalarComputeBackend.Instance.ExecuteMapInPlace(
                        source,
                        plan,
                        context),
                ComputeBackendKind.ParallelCpu =>
                    ParallelComputeBackend.Instance.ExecuteMapInPlace(
                        source,
                        plan,
                        context),
                ComputeBackendKind.Simd =>
                    SimdComputeBackend.Instance.ExecuteMapInPlace(
                        source,
                        plan,
                        context),
                _ => throw new InvalidOperationException(
                    "In-place backend resolution returned an unsupported backend.")
            };

        diagnostics = collectDiagnostics
            ? new ComputeDiagnostics(
                resolution.Backend.Kind,
                planningTime,
                execution.CompilationTime,
                execution.UploadTime,
                execution.ExecutionTime,
                execution.DownloadTime,
                execution.KernelCacheHit,
                execution.DeviceName)
            {
                BackendSelectionReason = resolution.Reason,
                IsInPlace = true
            }
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

        if (result.GpuMemoryBudgetBytes is <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                result.GpuMemoryBudgetBytes,
                "GpuMemoryBudgetBytes must be greater than zero.");
        }

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

    private static BackendResolution ResolveBackend(
        ComputeOptions options,
        ComputeExpressionPlan? plan,
        int elementCount)
    {
        BackendResolution resolution = options.Backend switch
        {
            ComputeBackendKind.Auto => SelectAutomaticBackend(options, plan, elementCount),
            ComputeBackendKind.Scalar => Explicit(ScalarComputeBackend.Instance),
            ComputeBackendKind.ParallelCpu => Explicit(ParallelComputeBackend.Instance),
            ComputeBackendKind.Simd => Explicit(SimdComputeBackend.Instance),
            ComputeBackendKind.Gpu => Explicit(GpuComputeBackend.Instance),
            _ => throw new ComputeBackendUnavailableException(options.Backend)
        };
        IComputeBackend backend = resolution.Backend;

        if (!backend.IsAvailable || (plan is not null && !backend.Supports(plan)))
        {
            if (options.Backend == ComputeBackendKind.Auto && options.AllowFallback)
            {
                return new BackendResolution(
                    ScalarComputeBackend.Instance,
                    $"{resolution.Reason} Selected backend was unavailable; " +
                    "Scalar fallback was used.",
                    resolution.EstimatedGpuMemoryBytes,
                    resolution.GpuMemoryBudgetBytes);
            }

            throw new ComputeBackendUnavailableException(backend.Kind);
        }

        return resolution;
    }

    private static BackendResolution ResolveDelegateBackend(
        ComputeOptions options,
        int elementCount) =>
        options.Backend switch
        {
            ComputeBackendKind.Auto =>
                SelectAutomaticDelegateBackend(options, elementCount),
            ComputeBackendKind.Scalar =>
                Explicit(ScalarComputeBackend.Instance),
            ComputeBackendKind.ParallelCpu =>
                Explicit(ParallelComputeBackend.Instance),
            ComputeBackendKind.Simd or ComputeBackendKind.Gpu =>
                throw new ComputeBackendNotSupportedException(
                    options.Backend,
                    "arbitrary user delegates",
                    $"{ComputeBackendKind.Scalar}, {ComputeBackendKind.ParallelCpu}"),
            _ => throw new ComputeBackendUnavailableException(options.Backend)
        };

    private static BackendResolution ResolveInPlaceBackend(
        ComputeOptions options,
        ComputeExpressionPlan plan,
        int elementCount)
    {
        BackendResolution resolution = options.Backend switch
        {
            ComputeBackendKind.Auto =>
                SelectAutomaticInPlaceBackend(options, plan, elementCount),
            ComputeBackendKind.Scalar =>
                Explicit(ScalarComputeBackend.Instance),
            ComputeBackendKind.ParallelCpu =>
                Explicit(ParallelComputeBackend.Instance),
            ComputeBackendKind.Simd =>
                Explicit(SimdComputeBackend.Instance),
            ComputeBackendKind.Gpu =>
                throw new ComputeBackendNotSupportedException(
                    options.Backend,
                    "in-place Map",
                    $"{ComputeBackendKind.Scalar}, " +
                    $"{ComputeBackendKind.ParallelCpu}, " +
                    $"{ComputeBackendKind.Simd}"),
            _ => throw new ComputeBackendUnavailableException(options.Backend)
        };

        if (!resolution.Backend.IsAvailable ||
            !resolution.Backend.Supports(plan))
        {
            throw new ComputeBackendUnavailableException(
                resolution.Backend.Kind);
        }

        return resolution;
    }

    private static BackendResolution SelectAutomaticInPlaceBackend(
        ComputeOptions options,
        ComputeExpressionPlan plan,
        int elementCount)
    {
        bool simdSupported =
            SimdComputeBackend.Instance.IsAvailable &&
            SimdComputeBackend.Instance.Supports(plan);

        if (simdSupported &&
            elementCount >= options.Thresholds.SimdThreshold)
        {
            return new BackendResolution(
                SimdComputeBackend.Instance,
                "GPU in-place execution is not implemented. SIMD selected " +
                "because the expression is supported and its threshold was reached.",
                null,
                null);
        }

        int availableParallelism =
            options.MaxDegreeOfParallelism ?? Environment.ProcessorCount;
        if (availableParallelism > 1 &&
            elementCount >= options.Thresholds.ParallelThreshold)
        {
            return new BackendResolution(
                ParallelComputeBackend.Instance,
                "GPU in-place execution is not implemented. Parallel CPU " +
                "selected because SIMD was unavailable or unsupported and the " +
                "parallel threshold was reached.",
                null,
                null);
        }

        return new BackendResolution(
            ScalarComputeBackend.Instance,
            "GPU in-place execution is not implemented. Scalar selected " +
            "because no accelerated CPU backend met its requirements.",
            null,
            null);
    }

    private static BackendResolution SelectAutomaticDelegateBackend(
        ComputeOptions options,
        int elementCount)
    {
        int availableParallelism =
            options.MaxDegreeOfParallelism ?? Environment.ProcessorCount;

        if (availableParallelism > 1 &&
            elementCount >= options.Thresholds.ParallelThreshold)
        {
            return new BackendResolution(
                ParallelComputeBackend.Instance,
                "Parallel CPU selected because arbitrary delegates are CPU-only " +
                "and the parallel threshold was reached.",
                null,
                null);
        }

        return new BackendResolution(
            ScalarComputeBackend.Instance,
            "Scalar selected because arbitrary delegates are CPU-only and the " +
            "parallel threshold or available parallelism requirement was not met.",
            null,
            null);
    }

    private static BackendResolution SelectAutomaticBackend(
        ComputeOptions options,
        ComputeExpressionPlan? plan,
        int elementCount)
    {
        ComputeExpressionComplexity complexity =
            ComputeExpressionClassifier.Classify(plan);
        int gpuThreshold =
            ComputeExpressionClassifier.GetGpuThreshold(plan, options.Thresholds);
        string gpuDecision;
        long? estimatedGpuMemoryBytes = null;
        long? gpuMemoryBudgetBytes = null;

        if (complexity == ComputeExpressionComplexity.Simple &&
            gpuThreshold == int.MaxValue)
        {
            gpuDecision =
                "GPU was not considered because automatic GPU selection for " +
                "CPU-resident simple expressions is disabled by default.";
        }
        else if (elementCount < gpuThreshold)
        {
            gpuDecision = $"GPU threshold {gpuThreshold} was not reached.";
        }
        else if (!GpuComputeBackend.TryGetAutomaticMemoryBudget(
                     options.GpuContext,
                     options.GpuMemoryBudgetBytes,
                     out long memoryBudget))
        {
            gpuDecision = "No hardware GPU accelerator is available.";
        }
        else
        {
            estimatedGpuMemoryBytes =
                EstimateGpuWorkingSetBytes(
                    plan?.ParameterCount ?? 1,
                    elementCount);
            gpuMemoryBudgetBytes = memoryBudget;

            if (estimatedGpuMemoryBytes <= memoryBudget)
            {
                return new BackendResolution(
                    GpuComputeBackend.Instance,
                    $"GPU selected for a {complexity.ToString().ToLowerInvariant()} " +
                    $"expression; estimated working set " +
                    $"{estimatedGpuMemoryBytes} bytes fits the " +
                    $"{memoryBudget}-byte budget.",
                    estimatedGpuMemoryBytes,
                    gpuMemoryBudgetBytes);
            }

            gpuDecision =
                $"GPU rejected because estimated working set " +
                $"{estimatedGpuMemoryBytes} bytes exceeds the " +
                $"{memoryBudget}-byte memory budget.";
        }

        bool simdSupported =
            SimdComputeBackend.Instance.IsAvailable &&
            (plan is null || SimdComputeBackend.Instance.Supports(plan));

        if (simdSupported &&
            elementCount >= options.Thresholds.SimdThreshold)
        {
            return new BackendResolution(
                SimdComputeBackend.Instance,
                $"{gpuDecision} SIMD selected because the expression is " +
                "supported and its threshold was reached.",
                estimatedGpuMemoryBytes,
                gpuMemoryBudgetBytes);
        }

        int availableParallelism =
            options.MaxDegreeOfParallelism ?? Environment.ProcessorCount;

        if (availableParallelism > 1 &&
            elementCount >= options.Thresholds.ParallelThreshold)
        {
            return new BackendResolution(
                ParallelComputeBackend.Instance,
                $"{gpuDecision} Parallel CPU selected because SIMD is " +
                "unavailable or unsupported and the parallel threshold was reached.",
                estimatedGpuMemoryBytes,
                gpuMemoryBudgetBytes);
        }

        return new BackendResolution(
            ScalarComputeBackend.Instance,
            $"{gpuDecision} Scalar selected because no accelerated CPU backend " +
            "met its availability and threshold requirements.",
            estimatedGpuMemoryBytes,
            gpuMemoryBudgetBytes);
    }

    internal static long EstimateGpuWorkingSetBytes(
        int parameterCount,
        int elementCount)
    {
        if (elementCount == 0)
        {
            return 0;
        }

        int fullLengthBuffers = parameterCount switch
        {
            2 => 3,
            1 => 2,
            _ => throw new ArgumentOutOfRangeException(nameof(parameterCount))
        };
        const long planningOverheadBytes = 1024 * 1024;

        return checked(
            (long)elementCount *
            sizeof(float) *
            fullLengthBuffers +
            planningOverheadBytes);
    }

    private static BackendResolution Explicit(IComputeBackend backend) =>
        new(
            backend,
            $"{backend.Kind} was explicitly requested.",
            null,
            null);

    private readonly record struct BackendResolution(
        IComputeBackend Backend,
        string Reason,
        long? EstimatedGpuMemoryBytes,
        long? GpuMemoryBudgetBytes);
}
