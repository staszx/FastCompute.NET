namespace FastCompute;

/// <summary>
/// Configures a compute operation.
/// </summary>
public sealed class ComputeOptions
{
    internal static ComputeOptions Default { get; } = new();

    /// <summary>
    /// Gets the requested backend. The default is <see cref="ComputeBackendKind.Auto"/>.
    /// </summary>
    public ComputeBackendKind Backend { get; init; } = ComputeBackendKind.Auto;

    /// <summary>
    /// Gets a value indicating whether automatic backend selection may fall back to
    /// another available backend.
    /// </summary>
    public bool AllowFallback { get; init; } = true;

    /// <summary>
    /// Gets the token used to cancel planning or execution.
    /// </summary>
    public CancellationToken CancellationToken { get; init; }

    /// <summary>
    /// Gets the maximum CPU parallelism.
    /// </summary>
    public int? MaxDegreeOfParallelism { get; init; }

    /// <summary>
    /// Gets a value indicating whether diagnostic-capable APIs should collect diagnostics.
    /// </summary>
    public bool EnableDiagnostics { get; init; }

    /// <summary>
    /// Gets the expression optimization mode.
    /// </summary>
    public ComputeOptimizationMode OptimizationMode { get; init; } = ComputeOptimizationMode.Strict;

    /// <summary>
    /// Gets the thresholds used by automatic backend selection.
    /// </summary>
    public ComputeThresholdOptions Thresholds { get; init; } = new();

    /// <summary>
    /// Gets an optional upper bound for memory used by one automatic GPU
    /// operation. The effective budget never exceeds the context safety limit.
    /// </summary>
    public long? GpuMemoryBudgetBytes { get; init; }

    /// <summary>
    /// Gets a value indicating whether Map and Zip may be split into sequential
    /// GPU chunks when the complete working set does not fit the memory budget.
    /// </summary>
    public bool EnableGpuChunking { get; init; } = true;

    /// <summary>
    /// Gets an optional maximum number of elements in one GPU Map or Zip chunk.
    /// Setting this value forces chunked execution when the input is larger.
    /// </summary>
    public int? GpuChunkElementCount { get; init; }

    /// <summary>
    /// Gets a value indicating whether an explicitly requested chunked GPU Map
    /// may overlap transfers and kernel execution using two accelerator streams.
    /// The default is false because streaming is workload-dependent.
    /// </summary>
    public bool EnableGpuStreaming { get; init; }

    /// <summary>
    /// Gets the preferred hardware GPU accelerator index. In Auto mode this
    /// selects which GPU is considered without forcing GPU execution. The
    /// context is created lazily only when the planner selects GPU.
    /// </summary>
    public int? PreferredGpuAcceleratorIndex { get; init; }

    /// <summary>
    /// Gets the reusable GPU context. When omitted for an explicit GPU operation,
    /// FastCompute uses its lazily created shared default context.
    /// This option cannot be combined with
    /// <see cref="PreferredGpuAcceleratorIndex"/>.
    /// </summary>
    public ComputeContext? GpuContext { get; init; }
}
