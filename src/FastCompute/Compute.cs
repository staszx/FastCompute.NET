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
public static partial class Compute
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
    /// Applies an expression and exposes the result through a task-compatible
    /// API. Cancellation is supplied through <see cref="ComputeOptions"/>.
    /// </summary>
    /// <remarks>
    /// Current backends expose synchronous completion primitives, so this
    /// method completes synchronously without scheduling unnecessary
    /// thread-pool work.
    /// </remarks>
    public static Task<float[]> RunAsync(
        float[] source,
        Expression<Func<float, float>> expression,
        ComputeOptions? options = null) =>
        Task.FromResult(Run(source, expression, options));

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
    /// Applies a binary expression and stores each result in
    /// <paramref name="target"/>.
    /// </summary>
    /// <remarks>
    /// If execution is cancelled or a backend operation fails, elements already
    /// processed are not rolled back.
    /// </remarks>
    /// <returns>The same array instance supplied in <paramref name="target"/>.</returns>
    public static float[] ZipInPlace(
        float[] target,
        float[] right,
        Expression<Func<float, float, float>> expression,
        ComputeOptions? options = null) =>
        ZipInPlaceCore(
            target,
            right,
            expression,
            options,
            collectDiagnostics: false,
            out _);

    /// <summary>
    /// Applies a binary expression in place and returns execution diagnostics.
    /// </summary>
    public static ComputeResult<float[]> ZipInPlaceWithDiagnostics(
        float[] target,
        float[] right,
        Expression<Func<float, float, float>> expression,
        ComputeOptions? options = null)
    {
        float[] value = ZipInPlaceCore(
            target,
            right,
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
        ComputeOptions? options = null) =>
        ZipCore(
            left,
            right,
            expression,
            options,
            collectDiagnostics: false,
            out _);

    /// <summary>
    /// Applies a binary expression and returns the result together with
    /// execution diagnostics.
    /// </summary>
    public static ComputeResult<float[]> ZipWithDiagnostics(
        float[] left,
        float[] right,
        Expression<Func<float, float, float>> expression,
        ComputeOptions? options = null)
    {
        float[] value = ZipCore(
            left,
            right,
            expression,
            options,
            collectDiagnostics: true,
            out ComputeDiagnostics? diagnostics);

        return new ComputeResult<float[]>(value, diagnostics!);
    }

    private static float[] ZipCore(
        float[] left,
        float[] right,
        Expression<Func<float, float, float>> expression,
        ComputeOptions? options,
        bool collectDiagnostics,
        out ComputeDiagnostics? diagnostics)
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

        long planningStarted = collectDiagnostics ? Stopwatch.GetTimestamp() : 0L;
        ComputeOptions effectiveOptions = ValidateOptions(options);
        effectiveOptions.CancellationToken.ThrowIfCancellationRequested();
        ComputeExpressionPlan plan = CreatePlan(expression, effectiveOptions);
        BackendResolution resolution =
            ResolveBackend(effectiveOptions, plan, left.Length);
        TimeSpan planningTime = collectDiagnostics
            ? Stopwatch.GetElapsedTime(planningStarted)
            : TimeSpan.Zero;
        var context = CreateExecutionContext(effectiveOptions, collectDiagnostics);
        ComputeBackendExecution<float[]> execution =
            resolution.Backend.ExecuteZip(left, right, plan, context);

        diagnostics = collectDiagnostics
            ? CreateDiagnostics(
                resolution,
                planningTime,
                execution,
                isInPlace: false)
            : null;

        return execution.Value;
    }

    /// <summary>
    /// Counts values in equally sized bins over the inclusive range
    /// [<paramref name="minimum"/>, <paramref name="maximum"/>].
    /// </summary>
    /// <remarks>
    /// Finite values outside the range are clamped to edge bins and NaN values
    /// are ignored. The maximum value belongs to the last bin. Use the
    /// overload accepting <see cref="HistogramOptions"/> to select another
    /// out-of-range behavior.
    /// </remarks>
    public static int[] Histogram(
        float[] source,
        int binCount,
        float minimum,
        float maximum,
        ComputeOptions? options = null) =>
        HistogramCore(
            source,
            binCount,
            minimum,
            maximum,
            new HistogramOptions(),
            options,
            collectDiagnostics: false,
            out _);

    /// <summary>
    /// Counts values in equally sized bins with explicit out-of-range behavior.
    /// </summary>
    public static int[] Histogram(
        float[] source,
        int binCount,
        float minimum,
        float maximum,
        HistogramOptions histogramOptions,
        ComputeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(histogramOptions);
        return HistogramCore(
            source,
            binCount,
            minimum,
            maximum,
            histogramOptions,
            options,
            collectDiagnostics: false,
            out _);
    }

    /// <summary>Builds a histogram and returns execution diagnostics.</summary>
    public static ComputeResult<int[]> HistogramWithDiagnostics(
        float[] source,
        int binCount,
        float minimum,
        float maximum,
        ComputeOptions? options = null)
    {
        int[] value = HistogramCore(
            source,
            binCount,
            minimum,
            maximum,
            new HistogramOptions(),
            options,
            collectDiagnostics: true,
            out ComputeDiagnostics? diagnostics);
        return new ComputeResult<int[]>(value, diagnostics!);
    }

    /// <summary>
    /// Builds a histogram with explicit out-of-range behavior and returns
    /// execution diagnostics.
    /// </summary>
    public static ComputeResult<int[]> HistogramWithDiagnostics(
        float[] source,
        int binCount,
        float minimum,
        float maximum,
        HistogramOptions histogramOptions,
        ComputeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(histogramOptions);
        int[] value = HistogramCore(
            source,
            binCount,
            minimum,
            maximum,
            histogramOptions,
            options,
            collectDiagnostics: true,
            out ComputeDiagnostics? diagnostics);
        return new ComputeResult<int[]>(value, diagnostics!);
    }

    private static int[] HistogramCore(
        float[] source,
        int binCount,
        float minimum,
        float maximum,
        HistogramOptions histogramOptions,
        ComputeOptions? options,
        bool collectDiagnostics,
        out ComputeDiagnostics? diagnostics)
    {
        ArgumentNullException.ThrowIfNull(source);
        ValidateHistogramArguments(binCount, minimum, maximum);
        ValidateHistogramOptions(histogramOptions);

        long planningStarted =
            collectDiagnostics ? Stopwatch.GetTimestamp() : 0L;
        ComputeOptions effectiveOptions = ValidateOptions(options);
        effectiveOptions.CancellationToken.ThrowIfCancellationRequested();
        BackendResolution resolution =
            ResolveHistogramBackend(
                effectiveOptions,
                source.Length,
                binCount);
        TimeSpan planningTime = collectDiagnostics
            ? Stopwatch.GetElapsedTime(planningStarted)
            : TimeSpan.Zero;
        ComputeExecutionContext context =
            CreateExecutionContext(effectiveOptions, collectDiagnostics);
        ComputeBackendExecution<int[]> execution =
            resolution.Backend.Kind switch
            {
                ComputeBackendKind.Scalar =>
                    ScalarComputeBackend.Instance.ExecuteHistogram(
                        source,
                        binCount,
                        minimum,
                        maximum,
                        histogramOptions.OutOfRangeMode,
                        context),
                ComputeBackendKind.ParallelCpu =>
                    ParallelComputeBackend.Instance.ExecuteHistogram(
                        source,
                        binCount,
                        minimum,
                        maximum,
                        histogramOptions.OutOfRangeMode,
                        context),
                ComputeBackendKind.Gpu =>
                    GpuComputeBackend.Instance.ExecuteHistogram(
                        source,
                        binCount,
                        minimum,
                        maximum,
                        histogramOptions.OutOfRangeMode,
                        context),
                _ => throw new InvalidOperationException(
                    "Histogram backend resolution returned an unsupported backend.")
            };

        diagnostics = collectDiagnostics
            ? CreateDiagnostics(
                resolution,
                planningTime,
                execution,
                isInPlace: false)
            : null;
        return execution.Value;
    }

    /// <summary>
    /// Computes the sum of all elements in an array.
    /// </summary>
    /// <param name="source">The input array.</param>
    /// <param name="options">Optional execution settings.</param>
    /// <returns>The element sum, or zero when the array is empty.</returns>
    public static float Sum(float[] source, ComputeOptions? options = null)
        => ReduceCore(
            source,
            ComputeReductionKind.Sum,
            options,
            collectDiagnostics: false,
            out _);

    /// <summary>Computes the sum and returns execution diagnostics.</summary>
    public static ComputeResult<float> SumWithDiagnostics(
        float[] source,
        ComputeOptions? options = null) =>
        ReduceWithDiagnostics(source, ComputeReductionKind.Sum, options);

    /// <summary>Computes the minimum element in an array.</summary>
    /// <exception cref="InvalidOperationException">The input array is empty.</exception>
    public static float Min(float[] source, ComputeOptions? options = null)
        => ReduceCore(
            source,
            ComputeReductionKind.Min,
            options,
            collectDiagnostics: false,
            out _);

    /// <summary>Computes the minimum and returns execution diagnostics.</summary>
    public static ComputeResult<float> MinWithDiagnostics(
        float[] source,
        ComputeOptions? options = null) =>
        ReduceWithDiagnostics(source, ComputeReductionKind.Min, options);

    /// <summary>Computes the maximum element in an array.</summary>
    /// <exception cref="InvalidOperationException">The input array is empty.</exception>
    public static float Max(float[] source, ComputeOptions? options = null)
        => ReduceCore(
            source,
            ComputeReductionKind.Max,
            options,
            collectDiagnostics: false,
            out _);

    /// <summary>Computes the maximum and returns execution diagnostics.</summary>
    public static ComputeResult<float> MaxWithDiagnostics(
        float[] source,
        ComputeOptions? options = null) =>
        ReduceWithDiagnostics(source, ComputeReductionKind.Max, options);

    /// <summary>Computes the arithmetic mean of an array.</summary>
    /// <exception cref="InvalidOperationException">The input array is empty.</exception>
    public static float Average(float[] source, ComputeOptions? options = null)
        => ReduceCore(
            source,
            ComputeReductionKind.Average,
            options,
            collectDiagnostics: false,
            out _);

    /// <summary>Computes the average and returns execution diagnostics.</summary>
    public static ComputeResult<float> AverageWithDiagnostics(
        float[] source,
        ComputeOptions? options = null) =>
        ReduceWithDiagnostics(source, ComputeReductionKind.Average, options);

    private static ComputeResult<float> ReduceWithDiagnostics(
        float[] source,
        ComputeReductionKind reduction,
        ComputeOptions? options)
    {
        float value = ReduceCore(
            source,
            reduction,
            options,
            collectDiagnostics: true,
            out ComputeDiagnostics? diagnostics);
        return new ComputeResult<float>(value, diagnostics!);
    }

    private static float ReduceCore(
        float[] source,
        ComputeReductionKind reduction,
        ComputeOptions? options,
        bool collectDiagnostics,
        out ComputeDiagnostics? diagnostics)
    {
        ArgumentNullException.ThrowIfNull(source);

        if (source.Length == 0 && reduction != ComputeReductionKind.Sum)
        {
            throw new InvalidOperationException(
                $"Cannot compute {reduction} for an empty array.");
        }

        long planningStarted =
            collectDiagnostics ? Stopwatch.GetTimestamp() : 0L;
        ComputeOptions effectiveOptions = ValidateOptions(options);
        effectiveOptions.CancellationToken.ThrowIfCancellationRequested();
        BackendResolution resolution =
            ResolveBackend(effectiveOptions, plan: null, source.Length);
        TimeSpan planningTime = collectDiagnostics
            ? Stopwatch.GetElapsedTime(planningStarted)
            : TimeSpan.Zero;
        var context = CreateExecutionContext(effectiveOptions, collectDiagnostics);
        ComputeBackendExecution<float> execution =
            resolution.Backend.Reduce(source, reduction, context);

        diagnostics = collectDiagnostics
            ? CreateDiagnostics(
                resolution,
                planningTime,
                execution,
                isInPlace: false)
            : null;

        return execution.Value;
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
            ? CreateDiagnostics(
                resolution,
                planningTime,
                execution,
                isInPlace: false)
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
            ResolveInPlaceBackend(
                effectiveOptions,
                plan,
                source.Length,
                fullLengthBufferCount: 1,
                operationName: "in-place Map");
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
                ComputeBackendKind.Gpu =>
                    GpuComputeBackend.Instance.ExecuteMapInPlace(
                        source,
                        plan,
                        context),
                _ => throw new InvalidOperationException(
                    "In-place backend resolution returned an unsupported backend.")
            };

        diagnostics = collectDiagnostics
            ? CreateDiagnostics(
                resolution,
                planningTime,
                execution,
                isInPlace: true)
            : null;

        return execution.Value;
    }

    private static float[] ZipInPlaceCore(
        float[] target,
        float[] right,
        Expression<Func<float, float, float>> expression,
        ComputeOptions? options,
        bool collectDiagnostics,
        out ComputeDiagnostics? diagnostics)
    {
        ArgumentNullException.ThrowIfNull(target);
        ArgumentNullException.ThrowIfNull(right);
        ArgumentNullException.ThrowIfNull(expression);

        if (target.Length != right.Length)
        {
            throw new ArgumentException(
                $"ZipInPlace requires arrays of equal length, but received " +
                $"{target.Length} and {right.Length}.",
                nameof(right));
        }

        long planningStarted =
            collectDiagnostics ? Stopwatch.GetTimestamp() : 0L;
        ComputeOptions effectiveOptions = ValidateOptions(options);
        effectiveOptions.CancellationToken.ThrowIfCancellationRequested();
        ComputeExpressionPlan plan = CreatePlan(expression, effectiveOptions);
        BackendResolution resolution =
            ResolveInPlaceBackend(
                effectiveOptions,
                plan,
                target.Length,
                fullLengthBufferCount: 2,
                operationName: "in-place Zip");
        TimeSpan planningTime = collectDiagnostics
            ? Stopwatch.GetElapsedTime(planningStarted)
            : TimeSpan.Zero;
        var context =
            CreateExecutionContext(effectiveOptions, collectDiagnostics);
        ComputeBackendExecution<float[]> execution =
            resolution.Backend.Kind switch
            {
                ComputeBackendKind.Scalar =>
                    ScalarComputeBackend.Instance.ExecuteZipInPlace(
                        target,
                        right,
                        plan,
                        context),
                ComputeBackendKind.ParallelCpu =>
                    ParallelComputeBackend.Instance.ExecuteZipInPlace(
                        target,
                        right,
                        plan,
                        context),
                ComputeBackendKind.Simd =>
                    SimdComputeBackend.Instance.ExecuteZipInPlace(
                        target,
                        right,
                        plan,
                        context),
                ComputeBackendKind.Gpu =>
                    GpuComputeBackend.Instance.ExecuteZipInPlace(
                        target,
                        right,
                        plan,
                        context),
                _ => throw new InvalidOperationException(
                    "In-place Zip backend resolution returned an unsupported backend.")
            };

        diagnostics = collectDiagnostics
            ? CreateDiagnostics(
                resolution,
                planningTime,
                execution,
                isInPlace: true)
            : null;

        return execution.Value;
    }

    private static ComputeDiagnostics CreateDiagnostics<T>(
        BackendResolution resolution,
        TimeSpan planningTime,
        ComputeBackendExecution<T> execution,
        bool isInPlace) =>
        new(
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
            EstimatedGpuMemoryBytes = resolution.EstimatedGpuMemoryBytes,
            GpuMemoryBudgetBytes = resolution.GpuMemoryBudgetBytes,
            IsInPlace = isInPlace,
            ChunkCount = execution.ChunkCount,
            ChunkElementCount = execution.ChunkElementCount,
            UploadedBytes = execution.UploadedBytes,
            DownloadedBytes = execution.DownloadedBytes,
            IsStreaming = execution.IsStreaming,
            StreamCount = execution.StreamCount
        };

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
        ValidateThreshold(result.Thresholds.GpuHistogramThreshold, nameof(ComputeThresholdOptions.GpuHistogramThreshold));

        if (result.GpuMemoryBudgetBytes is <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                result.GpuMemoryBudgetBytes,
                "GpuMemoryBudgetBytes must be greater than zero.");
        }

        if (result.GpuChunkElementCount is <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                result.GpuChunkElementCount,
                "GpuChunkElementCount must be greater than zero.");
        }

        if (result.PreferredGpuAcceleratorIndex is < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                result.PreferredGpuAcceleratorIndex,
                "PreferredGpuAcceleratorIndex cannot be negative.");
        }

        if (result.GpuContext is not null &&
            result.PreferredGpuAcceleratorIndex is not null)
        {
            throw new ArgumentException(
                "GpuContext and PreferredGpuAcceleratorIndex cannot be used together.",
                nameof(options));
        }

        return ComputeDefaults.Apply(result);
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

    private static void ValidateGpuStreamingRequest(
        ComputeOptions options,
        bool supportsStreaming)
    {
        if (!options.EnableGpuStreaming)
        {
            return;
        }

        if (options.Backend != ComputeBackendKind.Gpu)
        {
            throw new ArgumentException(
                "GPU streaming requires an explicitly selected GPU backend.",
                nameof(options));
        }

        if (!options.EnableGpuChunking)
        {
            throw new ArgumentException(
                "GPU streaming requires GPU chunking to be enabled.",
                nameof(options));
        }

        if (!supportsStreaming)
        {
            throw new NotSupportedException(
                "GPU streaming currently supports out-of-place unary Map only.");
        }
    }

    private static void ValidateHistogramArguments(
        int binCount,
        float minimum,
        float maximum)
    {
        if (binCount <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(binCount),
                binCount,
                "Histogram bin count must be greater than zero.");
        }

        if (!float.IsFinite(minimum))
        {
            throw new ArgumentOutOfRangeException(
                nameof(minimum),
                minimum,
                "Histogram minimum must be finite.");
        }

        if (!float.IsFinite(maximum) || maximum <= minimum)
        {
            throw new ArgumentOutOfRangeException(
                nameof(maximum),
                maximum,
                "Histogram maximum must be finite and greater than minimum.");
        }
    }

    private static void ValidateHistogramOptions(
        HistogramOptions histogramOptions)
    {
        if (!Enum.IsDefined(histogramOptions.OutOfRangeMode))
        {
            throw new ArgumentOutOfRangeException(
                nameof(histogramOptions),
                histogramOptions.OutOfRangeMode,
                "Histogram out-of-range mode is not defined.");
        }
    }

    private static ComputeExecutionContext CreateExecutionContext(
        ComputeOptions options,
        bool collectDiagnostics) =>
        new(
            options.CancellationToken,
            options.MaxDegreeOfParallelism,
            collectDiagnostics,
            options.GpuContext,
            options.PreferredGpuAcceleratorIndex,
            options.GpuMemoryBudgetBytes,
            options.EnableGpuChunking,
            options.GpuChunkElementCount,
            options.EnableGpuStreaming);

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
        ValidateGpuStreamingRequest(
            options,
            supportsStreaming: plan?.ParameterCount == 1);
        BackendResolution resolution = options.Backend switch
        {
            ComputeBackendKind.Auto => SelectAutomaticBackend(options, plan, elementCount),
            ComputeBackendKind.Scalar => Explicit(ScalarComputeBackend.Instance),
            ComputeBackendKind.ParallelCpu => Explicit(ParallelComputeBackend.Instance),
            ComputeBackendKind.Simd => Explicit(SimdComputeBackend.Instance),
            ComputeBackendKind.Gpu =>
                plan is null
                    ? ResolveExplicitGpuReduction(options, elementCount)
                    : ResolveExplicitGpu(options, plan, elementCount),
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

    private static BackendResolution ResolveExplicitGpuReduction(
        ComputeOptions options,
        int elementCount)
    {
        long gpuMemoryBudgetBytes =
            GpuComputeBackend.GetExplicitMemoryBudget(
                options.GpuContext,
                options.PreferredGpuAcceleratorIndex,
                options.GpuMemoryBudgetBytes);
        GpuChunkPlan chunkPlan = GpuChunkPlan.Create(
            elementCount,
            fullLengthBufferCount: 2,
            gpuMemoryBudgetBytes,
            options.EnableGpuChunking,
            options.GpuChunkElementCount);
        string executionKind =
            chunkPlan.IsChunked
                ? $"{chunkPlan.ChunkCount} sequential chunks of up to " +
                  $"{chunkPlan.ChunkElementCount} elements"
                : "one GPU allocation set";

        return new BackendResolution(
            GpuComputeBackend.Instance,
            $"GPU was explicitly requested for Reduction; execution uses " +
            $"{executionKind} within the {gpuMemoryBudgetBytes}-byte budget.",
            chunkPlan.FullWorkingSetBytes,
            gpuMemoryBudgetBytes);
    }

    private static BackendResolution ResolveExplicitGpu(
        ComputeOptions options,
        ComputeExpressionPlan plan,
        int elementCount)
    {
        long gpuMemoryBudgetBytes =
            GpuComputeBackend.GetExplicitMemoryBudget(
                options.GpuContext,
                options.PreferredGpuAcceleratorIndex,
                options.GpuMemoryBudgetBytes);
        GpuChunkPlan chunkPlan =
            plan.ParameterCount == 1
                ? GpuChunkPlan.CreateMap(
                    elementCount,
                    gpuMemoryBudgetBytes,
                    options.EnableGpuChunking,
                    options.GpuChunkElementCount,
                    options.EnableGpuStreaming)
                : GpuChunkPlan.Create(
                    elementCount,
                    fullLengthBufferCount: 3,
                    gpuMemoryBudgetBytes,
                    options.EnableGpuChunking,
                    options.GpuChunkElementCount);
        string executionKind =
            chunkPlan.IsChunked
                ? $"{chunkPlan.ChunkCount} " +
                  (options.EnableGpuStreaming
                      ? "double-buffered streaming"
                      : "sequential") +
                  " chunks of up to " +
                  $"{chunkPlan.ChunkElementCount} elements"
                : "one GPU allocation set";

        return new BackendResolution(
            GpuComputeBackend.Instance,
            $"GPU was explicitly requested; execution uses {executionKind} " +
            $"within the {gpuMemoryBudgetBytes}-byte budget.",
            chunkPlan.FullWorkingSetBytes,
            gpuMemoryBudgetBytes);
    }

    private static BackendResolution ResolveDelegateBackend(
        ComputeOptions options,
        int elementCount)
    {
        ValidateGpuStreamingRequest(options, supportsStreaming: false);
        return options.Backend switch
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
    }

    private static BackendResolution ResolveHistogramBackend(
        ComputeOptions options,
        int elementCount,
        int binCount)
    {
        ValidateGpuStreamingRequest(options, supportsStreaming: false);
        return options.Backend switch
        {
            ComputeBackendKind.Auto =>
                SelectAutomaticHistogramBackend(
                    options,
                    elementCount,
                    binCount),
            ComputeBackendKind.Scalar =>
                Explicit(ScalarComputeBackend.Instance),
            ComputeBackendKind.ParallelCpu =>
                Explicit(ParallelComputeBackend.Instance),
            ComputeBackendKind.Gpu =>
                ResolveExplicitGpuHistogram(
                    options,
                    elementCount,
                    binCount),
            ComputeBackendKind.Simd =>
                throw new ComputeBackendNotSupportedException(
                    ComputeBackendKind.Simd,
                    "Histogram",
                    $"{ComputeBackendKind.Scalar}, " +
                    $"{ComputeBackendKind.ParallelCpu}, " +
                    $"{ComputeBackendKind.Gpu}"),
            _ => throw new ComputeBackendUnavailableException(options.Backend)
        };
    }

    private static BackendResolution ResolveExplicitGpuHistogram(
        ComputeOptions options,
        int elementCount,
        int binCount)
    {
        long budgetBytes =
            GpuComputeBackend.GetExplicitMemoryBudget(
                options.GpuContext,
                options.PreferredGpuAcceleratorIndex,
                options.GpuMemoryBudgetBytes);
        GpuChunkPlan chunkPlan = CreateGpuHistogramChunkPlan(
            elementCount,
            binCount,
            budgetBytes,
            options);
        string executionKind =
            chunkPlan.IsChunked
                ? $"{chunkPlan.ChunkCount} sequential chunks of up to " +
                  $"{chunkPlan.ChunkElementCount} elements"
                : "one GPU input allocation";

        return new BackendResolution(
            GpuComputeBackend.Instance,
            $"GPU was explicitly requested for Histogram; execution uses " +
            $"{executionKind} within the {budgetBytes}-byte budget.",
            chunkPlan.FullWorkingSetBytes,
            budgetBytes);
    }

    private static BackendResolution SelectAutomaticHistogramBackend(
        ComputeOptions options,
        int elementCount,
        int binCount)
    {
        string gpuDecision;
        long? estimatedGpuMemoryBytes = null;
        long? gpuMemoryBudgetBytes = null;

        if (options.Thresholds.GpuHistogramThreshold == int.MaxValue)
        {
            gpuDecision =
                "Automatic GPU Histogram selection is disabled by default " +
                "for CPU-resident input.";
        }
        else if (elementCount < options.Thresholds.GpuHistogramThreshold)
        {
            gpuDecision =
                $"GPU threshold {options.Thresholds.GpuHistogramThreshold} " +
                "was not reached.";
        }
        else if (!GpuComputeBackend.TryGetAutomaticMemoryBudget(
                     options.GpuContext,
                     options.PreferredGpuAcceleratorIndex,
                     options.GpuMemoryBudgetBytes,
                     out long memoryBudget))
        {
            gpuDecision = GetUnavailableGpuReason(options);
        }
        else
        {
            estimatedGpuMemoryBytes =
                EstimateGpuHistogramWorkingSetBytes(elementCount, binCount);
            gpuMemoryBudgetBytes = memoryBudget;

            try
            {
                GpuChunkPlan chunkPlan = CreateGpuHistogramChunkPlan(
                    elementCount,
                    binCount,
                    memoryBudget,
                    options);
                string executionKind =
                    chunkPlan.IsChunked
                        ? $"chunked Histogram using {chunkPlan.ChunkCount} " +
                          $"chunks of up to " +
                          $"{chunkPlan.ChunkElementCount} elements"
                        : "Histogram whose full working set fits the budget";

                return new BackendResolution(
                    GpuComputeBackend.Instance,
                    $"GPU selected for {executionKind}.",
                    estimatedGpuMemoryBytes,
                    gpuMemoryBudgetBytes);
            }
            catch (ComputeGpuMemoryBudgetExceededException)
            {
                gpuDecision =
                    $"GPU rejected because the Histogram working set exceeds " +
                    $"the {memoryBudget}-byte memory budget and no configured " +
                    "chunk fits it.";
            }
        }

        int availableParallelism =
            options.MaxDegreeOfParallelism ?? Environment.ProcessorCount;
        if (availableParallelism > 1 &&
            elementCount >= options.Thresholds.ParallelThreshold)
        {
            return new BackendResolution(
                ParallelComputeBackend.Instance,
                $"{gpuDecision} Parallel CPU selected because its threshold " +
                "was reached.",
                estimatedGpuMemoryBytes,
                gpuMemoryBudgetBytes);
        }

        return new BackendResolution(
            ScalarComputeBackend.Instance,
            $"{gpuDecision} Scalar selected because the Parallel CPU threshold " +
            "or available parallelism requirement was not met.",
            estimatedGpuMemoryBytes,
            gpuMemoryBudgetBytes);
    }

    private static GpuChunkPlan CreateGpuHistogramChunkPlan(
        int elementCount,
        int binCount,
        long budgetBytes,
        ComputeOptions options) =>
        GpuChunkPlan.Create(
            elementCount,
            bytesPerElement: sizeof(float),
            fixedWorkingSetBytes: checked((long)binCount * sizeof(int)),
            budgetBytes,
            options.EnableGpuChunking,
            options.GpuChunkElementCount);

    private static BackendResolution ResolveInPlaceBackend(
        ComputeOptions options,
        ComputeExpressionPlan plan,
        int elementCount,
        int fullLengthBufferCount,
        string operationName)
    {
        ValidateGpuStreamingRequest(options, supportsStreaming: false);
        BackendResolution resolution = options.Backend switch
        {
            ComputeBackendKind.Auto =>
                SelectAutomaticInPlaceBackend(
                    options,
                    plan,
                    elementCount,
                    fullLengthBufferCount,
                    operationName),
            ComputeBackendKind.Scalar =>
                Explicit(ScalarComputeBackend.Instance),
            ComputeBackendKind.ParallelCpu =>
                Explicit(ParallelComputeBackend.Instance),
            ComputeBackendKind.Simd =>
                Explicit(SimdComputeBackend.Instance),
            ComputeBackendKind.Gpu =>
                ResolveExplicitGpuInPlace(
                    options,
                    elementCount,
                    fullLengthBufferCount,
                    operationName),
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

    private static BackendResolution ResolveExplicitGpuInPlace(
        ComputeOptions options,
        int elementCount,
        int fullLengthBufferCount,
        string operationName)
    {
        long gpuMemoryBudgetBytes =
            GpuComputeBackend.GetExplicitMemoryBudget(
                options.GpuContext,
                options.PreferredGpuAcceleratorIndex,
                options.GpuMemoryBudgetBytes);
        GpuChunkPlan chunkPlan = GpuChunkPlan.Create(
            elementCount,
            fullLengthBufferCount,
            gpuMemoryBudgetBytes,
            options.EnableGpuChunking,
            options.GpuChunkElementCount);
        string executionKind =
            chunkPlan.IsChunked
                ? $"{chunkPlan.ChunkCount} sequential chunks of up to " +
                  $"{chunkPlan.ChunkElementCount} elements"
                : "one GPU allocation set";

        return new BackendResolution(
            GpuComputeBackend.Instance,
            $"GPU was explicitly requested for {operationName}; execution " +
            $"uses {executionKind} within the " +
            $"{gpuMemoryBudgetBytes}-byte budget.",
            chunkPlan.FullWorkingSetBytes,
            gpuMemoryBudgetBytes);
    }

    private static BackendResolution SelectAutomaticInPlaceBackend(
        ComputeOptions options,
        ComputeExpressionPlan plan,
        int elementCount,
        int fullLengthBufferCount,
        string operationName)
    {
        ComputeExpressionComplexity complexity =
            ComputeExpressionClassifier.Classify(plan);
        int gpuThreshold =
            ComputeExpressionClassifier.GetGpuThreshold(
                plan,
                options.Thresholds);
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
                     options.PreferredGpuAcceleratorIndex,
                     options.GpuMemoryBudgetBytes,
                     out long memoryBudget))
        {
            gpuDecision = GetUnavailableGpuReason(options);
        }
        else
        {
            estimatedGpuMemoryBytes =
                GpuChunkPlan.EstimateWorkingSetBytes(
                    elementCount,
                    fullLengthBufferCount);
            gpuMemoryBudgetBytes = memoryBudget;

            if (estimatedGpuMemoryBytes <= memoryBudget)
            {
                return new BackendResolution(
                    GpuComputeBackend.Instance,
                    $"GPU selected for {operationName} and a " +
                    $"{complexity.ToString().ToLowerInvariant()} expression; " +
                    $"estimated working set {estimatedGpuMemoryBytes} bytes " +
                    $"fits the {memoryBudget}-byte budget.",
                    estimatedGpuMemoryBytes,
                    gpuMemoryBudgetBytes);
            }

            if (options.EnableGpuChunking)
            {
                try
                {
                    GpuChunkPlan chunkPlan = GpuChunkPlan.Create(
                        elementCount,
                        fullLengthBufferCount,
                        memoryBudget,
                        enableChunking: true,
                        options.GpuChunkElementCount);

                    return new BackendResolution(
                        GpuComputeBackend.Instance,
                        $"GPU selected for chunked {operationName} and a " +
                        $"{complexity.ToString().ToLowerInvariant()} expression; " +
                        $"the full {estimatedGpuMemoryBytes}-byte working set " +
                        $"exceeds the {memoryBudget}-byte budget, so execution " +
                        $"uses {chunkPlan.ChunkCount} sequential chunks of up to " +
                        $"{chunkPlan.ChunkElementCount} elements.",
                        estimatedGpuMemoryBytes,
                        gpuMemoryBudgetBytes);
                }
                catch (ComputeGpuMemoryBudgetExceededException)
                {
                    gpuDecision =
                        $"GPU rejected because the full estimated in-place " +
                        $"working set exceeds the {memoryBudget}-byte memory " +
                        "budget and no configured chunk fits it.";
                }
            }
            else
            {
                gpuDecision =
                    $"GPU rejected because estimated in-place working set " +
                    $"{estimatedGpuMemoryBytes} bytes exceeds the " +
                    $"{memoryBudget}-byte memory budget.";
            }
        }

        bool simdSupported =
            SimdComputeBackend.Instance.IsAvailable &&
            SimdComputeBackend.Instance.Supports(plan);

        if (simdSupported &&
            elementCount >= options.Thresholds.SimdThreshold)
        {
            return new BackendResolution(
                SimdComputeBackend.Instance,
                $"{gpuDecision} SIMD selected " +
                "because the expression is supported and its threshold was reached.",
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
                $"{gpuDecision} Parallel CPU " +
                "selected because SIMD was unavailable or unsupported and the " +
                "parallel threshold was reached.",
                estimatedGpuMemoryBytes,
                gpuMemoryBudgetBytes);
        }

        return new BackendResolution(
            ScalarComputeBackend.Instance,
            $"{gpuDecision} Scalar selected " +
            "because no accelerated CPU backend met its requirements.",
            estimatedGpuMemoryBytes,
            gpuMemoryBudgetBytes);
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
                     options.PreferredGpuAcceleratorIndex,
                     options.GpuMemoryBudgetBytes,
                     out long memoryBudget))
        {
            gpuDecision = GetUnavailableGpuReason(options);
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

            if (options.EnableGpuChunking)
            {
                int fullLengthBufferCount =
                    plan?.ParameterCount == 2 ? 3 : 2;
                try
                {
                    GpuChunkPlan chunkPlan = GpuChunkPlan.Create(
                        elementCount,
                        fullLengthBufferCount,
                        memoryBudget,
                        enableChunking: true,
                        options.GpuChunkElementCount);

                    return new BackendResolution(
                        GpuComputeBackend.Instance,
                        $"GPU selected for a chunked " +
                        $"{complexity.ToString().ToLowerInvariant()} expression; " +
                        $"the full {estimatedGpuMemoryBytes}-byte working set " +
                        $"exceeds the {memoryBudget}-byte budget, so execution uses " +
                        $"{chunkPlan.ChunkCount} sequential chunks of up to " +
                        $"{chunkPlan.ChunkElementCount} elements.",
                        estimatedGpuMemoryBytes,
                        gpuMemoryBudgetBytes);
                }
                catch (ComputeGpuMemoryBudgetExceededException)
                {
                    gpuDecision =
                        $"GPU rejected because the full estimated working set " +
                        $"exceeds the {memoryBudget}-byte memory budget and no " +
                        "configured chunk fits it.";
                }
            }
            else
            {
                gpuDecision =
                    $"GPU rejected because estimated working set " +
                    $"{estimatedGpuMemoryBytes} bytes exceeds the " +
                    $"{memoryBudget}-byte memory budget.";
            }
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
        return GpuChunkPlan.EstimateWorkingSetBytes(
            elementCount,
            fullLengthBuffers);
    }

    private static string GetUnavailableGpuReason(ComputeOptions options) =>
        options.PreferredGpuAcceleratorIndex is int index
            ? $"Preferred hardware GPU accelerator index {index} is unavailable."
            : "No hardware GPU accelerator is available.";

    internal static long EstimateGpuInPlaceWorkingSetBytes(
        int elementCount)
    {
        if (elementCount == 0)
        {
            return 0;
        }

        return checked(
            (long)elementCount *
            sizeof(float) +
            GpuChunkPlan.PlanningOverheadBytes);
    }

    internal static long EstimateGpuZipInPlaceWorkingSetBytes(
        int elementCount) =>
        GpuChunkPlan.EstimateWorkingSetBytes(
            elementCount,
            fullLengthBufferCount: 2);

    internal static long EstimateGpuHistogramWorkingSetBytes(
        int elementCount,
        int binCount) =>
        GpuChunkPlan.EstimateWorkingSetBytes(
            elementCount,
            bytesPerElement: sizeof(float),
            fixedWorkingSetBytes: checked((long)binCount * sizeof(int)));

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
