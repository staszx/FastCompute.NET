namespace FastCompute.Backends;

internal readonly record struct ComputeBackendExecution<T>(
    T Value,
    TimeSpan CompilationTime,
    TimeSpan ExecutionTime,
    TimeSpan UploadTime = default,
    TimeSpan DownloadTime = default,
    bool KernelCacheHit = false,
    string? DeviceName = null,
    int ChunkCount = 0,
    int ChunkElementCount = 0,
    long UploadedBytes = 0,
    long DownloadedBytes = 0,
    bool IsStreaming = false,
    int StreamCount = 0);

internal readonly record struct ComputeExecutionContext(
    CancellationToken CancellationToken,
    int? MaxDegreeOfParallelism,
    bool CollectDiagnostics,
    ComputeContext? GpuContext,
    int? PreferredGpuAcceleratorIndex,
    long? GpuMemoryBudgetBytes,
    bool EnableGpuChunking,
    int? GpuChunkElementCount,
    bool EnableGpuStreaming);
