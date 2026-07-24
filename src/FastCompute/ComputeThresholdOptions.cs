namespace FastCompute;

/// <summary>
/// Defines element-count thresholds used by automatic backend selection.
/// </summary>
public sealed class ComputeThresholdOptions
{
    /// <summary>Gets the minimum element count for automatic SIMD selection.</summary>
    public int SimdThreshold { get; init; } = 1_024;

    /// <summary>Gets the minimum element count for automatic parallel CPU selection.</summary>
    public int ParallelThreshold { get; init; } = 100_000;

    /// <summary>
    /// Gets the CPU-memory GPU threshold for simple expressions. The default
    /// disables automatic GPU selection because PCIe transfer normally dominates
    /// arithmetic-only one-shot operations. Set a lower value to opt in.
    /// </summary>
    public int GpuSimpleThreshold { get; init; } = int.MaxValue;

    /// <summary>Gets the CPU-memory GPU threshold for medium expressions.</summary>
    public int GpuMediumThreshold { get; init; } = 2_000_000;

    /// <summary>Gets the CPU-memory GPU threshold for heavy expressions.</summary>
    public int GpuHeavyThreshold { get; init; } = 300_000;
}
