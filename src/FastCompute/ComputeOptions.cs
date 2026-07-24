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
    /// Gets the reusable GPU context. When omitted for an explicit GPU operation,
    /// FastCompute uses its lazily created shared default context.
    /// </summary>
    public ComputeContext? GpuContext { get; init; }
}
