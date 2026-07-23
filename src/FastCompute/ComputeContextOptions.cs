namespace FastCompute;

/// <summary>
/// Configures creation of a reusable GPU compute context.
/// </summary>
public sealed class ComputeContextOptions
{
    /// <summary>
    /// Gets the explicit accelerator index, or <see langword="null"/> to select
    /// the preferred non-CPU accelerator and fall back to ILGPU CPU.
    /// </summary>
    public int? AcceleratorIndex { get; init; }
}
