namespace FastCompute;

/// <summary>
/// Describes the result of explicit GPU preparation.
/// </summary>
/// <param name="CacheHit">Whether both the expression plan and kernel were cached.</param>
/// <param name="PlanningTime">Time spent parsing and lowering the expression.</param>
/// <param name="CompilationTime">Time spent compiling the ILGPU kernel.</param>
/// <param name="Backend">The prepared backend.</param>
/// <param name="DeviceName">The target accelerator name.</param>
public sealed record ComputeCompilationResult(
    bool CacheHit,
    TimeSpan PlanningTime,
    TimeSpan CompilationTime,
    ComputeBackendKind Backend,
    string DeviceName);
