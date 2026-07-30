namespace FastCompute;

/// <summary>
/// Configures creation of a reusable GPU compute context.
/// </summary>
public sealed class ComputeContextOptions
{
    /// <summary>The default maximum amount of idle accelerator memory retained.</summary>
    public const long DefaultMemoryPoolLimitBytes = 256L * 1024 * 1024;

    /// <summary>
    /// Gets the explicit accelerator index, or <see langword="null"/> to select
    /// the preferred non-CPU accelerator and fall back to ILGPU CPU.
    /// </summary>
    public int? AcceleratorIndex { get; init; }

    /// <summary>
    /// Gets the maximum number of bytes retained by unused transient buffers.
    /// A value of zero disables retention while still allowing active rentals.
    /// </summary>
    public long MemoryPoolLimitBytes { get; init; } =
        DefaultMemoryPoolLimitBytes;
}
