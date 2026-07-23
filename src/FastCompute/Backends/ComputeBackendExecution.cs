namespace FastCompute.Backends;

internal readonly record struct ComputeBackendExecution<T>(
    T Value,
    TimeSpan CompilationTime,
    TimeSpan ExecutionTime,
    TimeSpan UploadTime = default,
    TimeSpan DownloadTime = default,
    bool KernelCacheHit = false,
    string? DeviceName = null);

internal readonly record struct ComputeExecutionContext(
    CancellationToken CancellationToken,
    int? MaxDegreeOfParallelism,
    bool CollectDiagnostics,
    ComputeContext? GpuContext);
