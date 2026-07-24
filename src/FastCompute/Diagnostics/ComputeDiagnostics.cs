namespace FastCompute.Diagnostics;

/// <summary>
/// Describes how a compute operation was planned and executed.
/// </summary>
/// <param name="Backend">The backend that executed the operation.</param>
/// <param name="PlanningTime">Time spent parsing, optimizing, and selecting a backend.</param>
/// <param name="CompilationTime">Time spent compiling the backend operation.</param>
/// <param name="UploadTime">Time spent uploading inputs to a device.</param>
/// <param name="ExecutionTime">Time spent executing the operation.</param>
/// <param name="DownloadTime">Time spent downloading results from a device.</param>
/// <param name="KernelCacheHit">Whether a cached compiled kernel was used.</param>
/// <param name="DeviceName">The selected device name, when applicable.</param>
public sealed record ComputeDiagnostics(
    ComputeBackendKind Backend,
    TimeSpan PlanningTime,
    TimeSpan CompilationTime,
    TimeSpan UploadTime,
    TimeSpan ExecutionTime,
    TimeSpan DownloadTime,
    bool KernelCacheHit,
    string? DeviceName)
{
    /// <summary>Gets a human-readable explanation of backend selection.</summary>
    public string? BackendSelectionReason { get; init; }

    /// <summary>Gets the estimated GPU working set considered by Auto.</summary>
    public long? EstimatedGpuMemoryBytes { get; init; }

    /// <summary>Gets the effective GPU memory budget considered by Auto.</summary>
    public long? GpuMemoryBudgetBytes { get; init; }

    /// <summary>Gets a value indicating whether execution reused the input array.</summary>
    public bool IsInPlace { get; init; }
}
