using System.Collections.Concurrent;
using System.Diagnostics;
using System.Linq.Expressions;
using FastCompute.Backends;
using FastCompute.Expressions;
using FastCompute.Gpu;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using IlGpuContext = ILGPU.Context;

namespace FastCompute;

/// <summary>
/// Owns an ILGPU accelerator, compiled kernels, and lowered expression plans.
/// </summary>
public sealed class ComputeContext : IDisposable
{
    private const long AutoMemoryBudgetNumerator = 3;
    private const long AutoMemoryBudgetDenominator = 4;

    private readonly IlGpuContext ilGpuContext;
    private readonly Accelerator accelerator;
    private readonly GpuFloatMemoryPool memoryPool;
    private readonly ConcurrentDictionary<string, GpuProgram> programs = new();
    private readonly ConcurrentDictionary<ComputeKernelKind, Lazy<CompiledKernel>> kernels = new();
    private long graphCopyOnWriteCount;
    private long graphInPlaceReuseCount;
    private int disposed;

    private ComputeContext(ComputeContextOptions options)
    {
        ilGpuContext = IlGpuContext.Create(
            builder => builder.AllAccelerators().EnableAlgorithms());

        try
        {
            Device device = SelectDevice(ilGpuContext, options.AcceleratorIndex);
            accelerator = device.CreateAccelerator(ilGpuContext);
            memoryPool = new GpuFloatMemoryPool(accelerator);
        }
        catch
        {
            ilGpuContext.Dispose();
            throw;
        }
    }

    /// <summary>Gets the selected accelerator name.</summary>
    public string DeviceName
    {
        get
        {
            ThrowIfDisposed();
            return accelerator.Name;
        }
    }

    /// <summary>Gets the total accelerator memory reported by ILGPU.</summary>
    public long DeviceMemorySize
    {
        get
        {
            ThrowIfDisposed();
            return accelerator.MemorySize;
        }
    }

    /// <summary>Gets a snapshot of transient device-buffer pool usage.</summary>
    public ComputeMemoryPoolStatistics MemoryPoolStatistics
    {
        get
        {
            ThrowIfDisposed();
            return memoryPool.Statistics;
        }
    }

    /// <summary>Creates a reusable context and selects an accelerator.</summary>
    public static ComputeContext Create(ComputeContextOptions? options = null) =>
        new(options ?? new ComputeContextOptions());

    /// <summary>Returns accelerators in the same order used by explicit selection.</summary>
    public static IReadOnlyList<ComputeDeviceInfo> GetAccelerators()
    {
        using IlGpuContext context =
            IlGpuContext.Create(
                builder => builder.AllAccelerators().EnableAlgorithms());

        return context.Devices
            .Select((device, index) =>
                new ComputeDeviceInfo(
                    index,
                    device.Name,
                    device.AcceleratorType.ToString()))
            .ToArray();
    }

    /// <summary>Uploads an unmanaged array and returns a GPU-resident buffer.</summary>
    public ComputeBuffer<T> Upload<T>(T[] source)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ThrowIfDisposed();
        ValidateElementType(typeof(T));
        if (source.Length == 0)
        {
            return new ComputeBuffer<T>(
                this,
                new BufferSourceNode<T>(this, length: 0));
        }

        MemoryBuffer1D<T, Stride1D.Dense> buffer =
            accelerator.Allocate1D(source);
        return new ComputeBuffer<T>(
            this,
            new BufferSourceNode<T>(this, buffer));
    }

    /// <summary>
    /// Parses an expression and synchronously compiles its map-kernel template.
    /// </summary>
    public ComputeCompilationResult Precompile<T>(Expression<Func<T, T>> expression)
        where T : unmanaged =>
        PrecompileCore(expression, ComputeKernelKind.Map, typeof(T));

    /// <summary>
    /// Parses an expression and synchronously compiles its zip-kernel template.
    /// </summary>
    public ComputeCompilationResult Precompile<T>(Expression<Func<T, T, T>> expression)
        where T : unmanaged =>
        PrecompileCore(expression, ComputeKernelKind.Zip, typeof(T));

    /// <summary>Synchronously compiles the GPU reduction template.</summary>
    public ComputeCompilationResult PrecompileReduction<T>(
        ComputeReductionKind reduction)
        where T : unmanaged =>
        PrecompileReductionCore(reduction, typeof(T));

    /// <summary>Synchronously compiles the GPU histogram template.</summary>
    public ComputeCompilationResult PrecompileHistogram<T>()
        where T : unmanaged =>
        PrecompileHistogramCore(typeof(T));

    /// <summary>
    /// Synchronously prepares several map, zip, reduction, or histogram kernels.
    /// </summary>
    public IReadOnlyList<ComputeCompilationResult> Precompile(
        params ComputeKernelDescriptor[] kernelsToPrepare)
    {
        ArgumentNullException.ThrowIfNull(kernelsToPrepare);
        ThrowIfDisposed();

        var results = new ComputeCompilationResult[kernelsToPrepare.Length];
        for (int index = 0; index < kernelsToPrepare.Length; index++)
        {
            ComputeKernelDescriptor descriptor =
                kernelsToPrepare[index] ??
                throw new ArgumentException("Kernel descriptors cannot contain null.", nameof(kernelsToPrepare));

            results[index] = descriptor.Kind switch
            {
                ComputeKernelKind.Reduction =>
                    PrecompileReductionCore(
                        descriptor.Reduction!.Value,
                        descriptor.ElementType),
                ComputeKernelKind.Histogram =>
                    PrecompileHistogramCore(descriptor.ElementType),
                _ => PrecompileCore(
                    descriptor.Expression!,
                    descriptor.Kind,
                    descriptor.ElementType)
            };
        }

        return results;
    }

    /// <summary>
    /// Synchronously compiles every GPU kernel template implemented in this version.
    /// </summary>
    public IReadOnlyList<ComputeCompilationResult> PrecompileAll()
    {
        ThrowIfDisposed();
        ComputeKernelKind[] kinds = Enum.GetValues<ComputeKernelKind>();
        var results = new ComputeCompilationResult[kinds.Length];

        for (int index = 0; index < kinds.Length; index++)
        {
            _ = GetOrCompileKernel(kinds[index], out KernelCompilation compilation);
            results[index] = new ComputeCompilationResult(
                compilation.CacheHit,
                TimeSpan.Zero,
                compilation.CompilationTime,
                ComputeBackendKind.Gpu,
                accelerator.Name);
        }

        return results;
    }

    /// <summary>
    /// Creates a reusable map operation that skips expression planning and kernel lookup.
    /// </summary>
    public PreparedCompute<T> Prepare<T>(Expression<Func<T, T>> expression)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(expression);
        ThrowIfDisposed();
        ValidateElementType(typeof(T));

        GpuProgram program = GetOrCreateProgram(expression, out _);
        CompiledKernel kernel = GetOrCompileKernel(ComputeKernelKind.Map, out _);
        return new PreparedCompute<T>(this, program, kernel);
    }

    /// <summary>Executes a float map expression on the selected accelerator.</summary>
    public float[] Run(float[] source, Expression<Func<float, float>> expression)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(expression);
        ThrowIfDisposed();

        GpuProgram program = GetOrCreateProgram(expression, out _);
        CompiledKernel kernel = GetOrCompileKernel(ComputeKernelKind.Map, out _);
        return ExecuteMap(source, program, kernel, CancellationToken.None);
    }

    /// <summary>
    /// Executes a float map expression on the selected accelerator and writes
    /// the result back into the source array.
    /// </summary>
    public float[] RunInPlace(
        float[] source,
        Expression<Func<float, float>> expression)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(expression);
        ThrowIfDisposed();

        GpuProgram program = GetOrCreateProgram(expression, out _);
        CompiledKernel kernel = GetOrCompileKernel(ComputeKernelKind.Map, out _);
        return ExecuteMapInPlaceOnDevice(
            source,
            program,
            kernel,
            CancellationToken.None,
            collectDiagnostics: false,
            TimeSpan.Zero,
            kernelCacheHit: true).Value;
    }

    /// <summary>Executes a float zip expression on the selected accelerator.</summary>
    public float[] Zip(
        float[] left,
        float[] right,
        Expression<Func<float, float, float>> expression)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        ArgumentNullException.ThrowIfNull(expression);
        ThrowIfDisposed();

        if (left.Length != right.Length)
        {
            throw new ArgumentException(
                $"Zip requires arrays of equal length, but received {left.Length} and {right.Length}.",
                nameof(right));
        }

        GpuProgram program = GetOrCreateProgram(expression, out _);
        CompiledKernel kernel = GetOrCompileKernel(ComputeKernelKind.Zip, out _);
        return ExecuteZip(left, right, program, kernel, CancellationToken.None);
    }

    /// <summary>
    /// Executes a float zip expression and writes the result into
    /// <paramref name="target"/>.
    /// </summary>
    public float[] ZipInPlace(
        float[] target,
        float[] right,
        Expression<Func<float, float, float>> expression)
    {
        ArgumentNullException.ThrowIfNull(target);
        ArgumentNullException.ThrowIfNull(right);
        ArgumentNullException.ThrowIfNull(expression);
        ThrowIfDisposed();

        if (target.Length != right.Length)
        {
            throw new ArgumentException(
                $"ZipInPlace requires arrays of equal length, but received " +
                $"{target.Length} and {right.Length}.",
                nameof(right));
        }

        GpuProgram program = GetOrCreateProgram(expression, out _);
        CompiledKernel kernel = GetOrCompileKernel(ComputeKernelKind.Zip, out _);
        return ExecuteZipInPlaceOnDevice(
            target,
            right,
            program,
            kernel,
            CancellationToken.None,
            collectDiagnostics: false,
            TimeSpan.Zero,
            kernelCacheHit: true).Value;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposed, 1) != 0)
        {
            return;
        }

        memoryPool.Dispose();
        accelerator.Dispose();
        ilGpuContext.Dispose();
    }

    internal float[] ExecuteMap(
        float[] source,
        GpuProgram program,
        CompiledKernel kernel,
        CancellationToken cancellationToken) =>
        ExecuteMapOnDevice(
            source,
            program,
            kernel,
            cancellationToken,
            collectDiagnostics: false,
            TimeSpan.Zero,
            kernelCacheHit: true).Value;

    internal ComputeBackendExecution<float[]> ExecuteMapPlan(
        float[] source,
        ComputeExpressionPlan plan,
        ComputeExecutionContext executionContext)
    {
        ThrowIfDisposed();
        GpuProgram program = GetOrCreateProgram(plan, out bool programCacheHit);
        CompiledKernel kernel = GetOrCompileKernel(
            ComputeKernelKind.Map,
            out KernelCompilation compilation);
        long budgetBytes =
            GetAutomaticMemoryBudget(
                executionContext.GpuMemoryBudgetBytes);
        GpuChunkPlan chunkPlan = GpuChunkPlan.CreateMap(
            source.Length,
            budgetBytes,
            executionContext.EnableGpuChunking,
            executionContext.GpuChunkElementCount,
            executionContext.EnableGpuStreaming);

        return chunkPlan.IsChunked
            ? executionContext.EnableGpuStreaming
                ? ExecuteMapDoubleBuffered(
                    source,
                    program,
                    kernel,
                    chunkPlan,
                    executionContext.CancellationToken,
                    executionContext.CollectDiagnostics,
                    compilation.CompilationTime,
                    programCacheHit && compilation.CacheHit)
                : ExecuteMapChunked(
                    source,
                    program,
                    kernel,
                    chunkPlan,
                    executionContext.CancellationToken,
                    executionContext.CollectDiagnostics,
                    compilation.CompilationTime,
                    programCacheHit && compilation.CacheHit)
            : ExecuteMapOnDevice(
                source,
                program,
                kernel,
                executionContext.CancellationToken,
                executionContext.CollectDiagnostics,
                compilation.CompilationTime,
                programCacheHit && compilation.CacheHit);
    }

    internal ComputeBackendExecution<float[]> ExecuteMapInPlacePlan(
        float[] source,
        ComputeExpressionPlan plan,
        ComputeExecutionContext executionContext)
    {
        ThrowIfDisposed();
        GpuProgram program = GetOrCreateProgram(plan, out bool programCacheHit);
        CompiledKernel kernel = GetOrCompileKernel(
            ComputeKernelKind.Map,
            out KernelCompilation compilation);
        long budgetBytes =
            GetAutomaticMemoryBudget(
                executionContext.GpuMemoryBudgetBytes);
        GpuChunkPlan chunkPlan = GpuChunkPlan.Create(
            source.Length,
            fullLengthBufferCount: 1,
            budgetBytes,
            executionContext.EnableGpuChunking,
            executionContext.GpuChunkElementCount);

        return chunkPlan.IsChunked
            ? ExecuteMapInPlaceChunked(
                source,
                program,
                kernel,
                chunkPlan,
                executionContext.CancellationToken,
                executionContext.CollectDiagnostics,
                compilation.CompilationTime,
                programCacheHit && compilation.CacheHit)
            : ExecuteMapInPlaceOnDevice(
                source,
                program,
                kernel,
                executionContext.CancellationToken,
                executionContext.CollectDiagnostics,
                compilation.CompilationTime,
                programCacheHit && compilation.CacheHit);
    }

    internal float[] ExecuteZip(
        float[] left,
        float[] right,
        GpuProgram program,
        CompiledKernel kernel,
        CancellationToken cancellationToken) =>
        ExecuteZipOnDevice(
            left,
            right,
            program,
            kernel,
            cancellationToken,
            collectDiagnostics: false,
            TimeSpan.Zero,
            kernelCacheHit: true).Value;

    internal ComputeBackendExecution<float[]> ExecuteZipPlan(
        float[] left,
        float[] right,
        ComputeExpressionPlan plan,
        ComputeExecutionContext executionContext)
    {
        ThrowIfDisposed();
        GpuProgram program = GetOrCreateProgram(plan, out bool programCacheHit);
        CompiledKernel kernel = GetOrCompileKernel(
            ComputeKernelKind.Zip,
            out KernelCompilation compilation);
        long budgetBytes =
            GetAutomaticMemoryBudget(
                executionContext.GpuMemoryBudgetBytes);
        GpuChunkPlan chunkPlan = GpuChunkPlan.Create(
            left.Length,
            fullLengthBufferCount: 3,
            budgetBytes,
            executionContext.EnableGpuChunking,
            executionContext.GpuChunkElementCount);

        return chunkPlan.IsChunked
            ? ExecuteZipChunked(
                left,
                right,
                program,
                kernel,
                chunkPlan,
                executionContext.CancellationToken,
                executionContext.CollectDiagnostics,
                compilation.CompilationTime,
                programCacheHit && compilation.CacheHit)
            : ExecuteZipOnDevice(
                left,
                right,
                program,
                kernel,
                executionContext.CancellationToken,
                executionContext.CollectDiagnostics,
                compilation.CompilationTime,
                programCacheHit && compilation.CacheHit);
    }

    internal ComputeBackendExecution<float[]> ExecuteZipInPlacePlan(
        float[] target,
        float[] right,
        ComputeExpressionPlan plan,
        ComputeExecutionContext executionContext)
    {
        ThrowIfDisposed();
        GpuProgram program = GetOrCreateProgram(plan, out bool programCacheHit);
        CompiledKernel kernel = GetOrCompileKernel(
            ComputeKernelKind.Zip,
            out KernelCompilation compilation);
        long budgetBytes =
            GetAutomaticMemoryBudget(
                executionContext.GpuMemoryBudgetBytes);
        GpuChunkPlan chunkPlan = GpuChunkPlan.Create(
            target.Length,
            fullLengthBufferCount: 2,
            budgetBytes,
            executionContext.EnableGpuChunking,
            executionContext.GpuChunkElementCount);

        return chunkPlan.IsChunked
            ? ExecuteZipInPlaceChunked(
                target,
                right,
                program,
                kernel,
                chunkPlan,
                executionContext.CancellationToken,
                executionContext.CollectDiagnostics,
                compilation.CompilationTime,
                programCacheHit && compilation.CacheHit)
            : ExecuteZipInPlaceOnDevice(
                target,
                right,
                program,
                kernel,
                executionContext.CancellationToken,
                executionContext.CollectDiagnostics,
                compilation.CompilationTime,
                programCacheHit && compilation.CacheHit);
    }

    internal ComputeBackendExecution<float> ExecuteReduction(
        float[] source,
        ComputeReductionKind reduction,
        ComputeExecutionContext executionContext)
    {
        ThrowIfDisposed();
        executionContext.CancellationToken.ThrowIfCancellationRequested();

        CompiledKernel kernel = GetOrCompileKernel(
            ComputeKernelKind.Reduction,
            out KernelCompilation compilation);
        long budgetBytes =
            GetAutomaticMemoryBudget(
                executionContext.GpuMemoryBudgetBytes);
        GpuChunkPlan chunkPlan = GpuChunkPlan.Create(
            source.Length,
            fullLengthBufferCount: 2,
            budgetBytes,
            executionContext.EnableGpuChunking,
            executionContext.GpuChunkElementCount);

        if (source.Length == 0)
        {
            return new ComputeBackendExecution<float>(
                0f,
                compilation.CompilationTime,
                TimeSpan.Zero,
                KernelCacheHit: compilation.CacheHit,
                DeviceName: accelerator.Name);
        }

        if (chunkPlan.IsChunked)
        {
            return ExecuteReductionChunked(
                source,
                reduction,
                kernel,
                chunkPlan,
                executionContext,
                compilation);
        }

        var leases = new List<GpuFloatMemoryPool.Lease>();

        try
        {
            long uploadStarted = StartTiming(executionContext.CollectDiagnostics);
            GpuFloatMemoryPool.Lease sourceLease =
                memoryPool.Rent(source.Length);
            leases.Add(sourceLease);
            MemoryBuffer1D<float, Stride1D.Dense> sourceBuffer =
                sourceLease.Buffer;
            sourceBuffer.CopyFromCPU(source);
            TimeSpan uploadTime =
                StopTiming(uploadStarted, executionContext.CollectDiagnostics);

            long executionStarted =
                StartTiming(executionContext.CollectDiagnostics);
            MemoryBuffer1D<float, Stride1D.Dense> current =
                ExecuteReductionPasses(
                    sourceBuffer,
                    source.Length,
                    reduction,
                    kernel,
                    leases);
            TimeSpan executionTime =
                StopTiming(executionStarted, executionContext.CollectDiagnostics);
            executionContext.CancellationToken.ThrowIfCancellationRequested();

            long downloadStarted =
                StartTiming(executionContext.CollectDiagnostics);
            float result = current.GetAsArray1D()[0];
            if (reduction == ComputeReductionKind.Average)
            {
                result /= source.Length;
            }

            TimeSpan downloadTime =
                StopTiming(downloadStarted, executionContext.CollectDiagnostics);

            return new ComputeBackendExecution<float>(
                result,
                compilation.CompilationTime,
                executionTime,
                uploadTime,
                downloadTime,
                compilation.CacheHit,
                accelerator.Name,
                ChunkCount: 1,
                ChunkElementCount: source.Length,
                UploadedBytes: checked((long)source.Length * sizeof(float)),
                DownloadedBytes: sizeof(float));
        }
        finally
        {
            ReturnLeases(leases);
        }
    }

    internal ComputeBackendExecution<int[]> ExecuteHistogram(
        float[] source,
        int binCount,
        float minimum,
        float maximum,
        ComputeExecutionContext executionContext)
    {
        ThrowIfDisposed();
        executionContext.CancellationToken.ThrowIfCancellationRequested();
        CompiledKernel kernel = GetOrCompileKernel(
            ComputeKernelKind.Histogram,
            out KernelCompilation compilation);
        long budgetBytes =
            GetAutomaticMemoryBudget(
                executionContext.GpuMemoryBudgetBytes);
        long histogramBytes = checked((long)binCount * sizeof(int));
        GpuChunkPlan chunkPlan = GpuChunkPlan.Create(
            source.Length,
            bytesPerElement: sizeof(float),
            fixedWorkingSetBytes: histogramBytes,
            budgetBytes,
            executionContext.EnableGpuChunking,
            executionContext.GpuChunkElementCount);

        if (source.Length == 0)
        {
            return new ComputeBackendExecution<int[]>(
                new int[binCount],
                compilation.CompilationTime,
                TimeSpan.Zero,
                KernelCacheHit: compilation.CacheHit,
                DeviceName: accelerator.Name);
        }

        TimeSpan uploadTime = TimeSpan.Zero;
        TimeSpan executionTime = TimeSpan.Zero;
        float scale = binCount / (maximum - minimum);
        using MemoryBuffer1D<int, Stride1D.Dense> histogramBuffer =
            accelerator.Allocate1D<int>(binCount);
        long clearStarted =
            StartTiming(executionContext.CollectDiagnostics);
        histogramBuffer.MemSetToZero();
        accelerator.Synchronize();
        executionTime +=
            StopTiming(clearStarted, executionContext.CollectDiagnostics);

        for (int offset = 0; offset < source.Length;)
        {
            executionContext.CancellationToken.ThrowIfCancellationRequested();
            int count =
                Math.Min(chunkPlan.ChunkElementCount, source.Length - offset);
            using GpuFloatMemoryPool.Lease sourceLease =
                memoryPool.Rent(count);
            MemoryBuffer1D<float, Stride1D.Dense> sourceBuffer =
                sourceLease.Buffer;

            long uploadStarted =
                StartTiming(executionContext.CollectDiagnostics);
            sourceBuffer.View.CopyFromCPU(
                accelerator.DefaultStream,
                source.AsSpan(offset, count));
            accelerator.Synchronize();
            uploadTime +=
                StopTiming(
                    uploadStarted,
                    executionContext.CollectDiagnostics);

            long executionStarted =
                StartTiming(executionContext.CollectDiagnostics);
            kernel.Histogram!(
                count,
                sourceBuffer.View,
                histogramBuffer.View,
                binCount,
                minimum,
                maximum,
                scale);
            accelerator.Synchronize();
            executionTime +=
                StopTiming(
                    executionStarted,
                    executionContext.CollectDiagnostics);
            offset += count;
        }

        executionContext.CancellationToken.ThrowIfCancellationRequested();
        long downloadStarted =
            StartTiming(executionContext.CollectDiagnostics);
        int[] result = histogramBuffer.GetAsArray1D();
        TimeSpan downloadTime =
            StopTiming(
                downloadStarted,
                executionContext.CollectDiagnostics);

        return new ComputeBackendExecution<int[]>(
            result,
            compilation.CompilationTime,
            executionTime,
            uploadTime,
            downloadTime,
            compilation.CacheHit,
            accelerator.Name,
            chunkPlan.ChunkCount,
            chunkPlan.ChunkElementCount,
            checked((long)source.Length * sizeof(float)),
            histogramBytes);
    }

    private ComputeBackendExecution<float> ExecuteReductionChunked(
        float[] source,
        ComputeReductionKind reduction,
        CompiledKernel kernel,
        GpuChunkPlan chunkPlan,
        ComputeExecutionContext executionContext,
        KernelCompilation compilation)
    {
        TimeSpan uploadTime = TimeSpan.Zero;
        TimeSpan executionTime = TimeSpan.Zero;
        TimeSpan downloadTime = TimeSpan.Zero;
        float combined = 0f;
        bool hasCombinedValue = false;
        var partialResult = new float[1];
        var leases = new List<GpuFloatMemoryPool.Lease>();

        for (int offset = 0; offset < source.Length;)
        {
            executionContext.CancellationToken.ThrowIfCancellationRequested();
            int count =
                Math.Min(chunkPlan.ChunkElementCount, source.Length - offset);

            try
            {
                long uploadStarted =
                    StartTiming(executionContext.CollectDiagnostics);
                GpuFloatMemoryPool.Lease sourceLease =
                    memoryPool.Rent(count);
                leases.Add(sourceLease);
                MemoryBuffer1D<float, Stride1D.Dense> sourceBuffer =
                    sourceLease.Buffer;
                sourceBuffer.View.CopyFromCPU(
                    accelerator.DefaultStream,
                    source.AsSpan(offset, count));
                accelerator.Synchronize();
                uploadTime +=
                    StopTiming(
                        uploadStarted,
                        executionContext.CollectDiagnostics);

                long executionStarted =
                    StartTiming(executionContext.CollectDiagnostics);
                MemoryBuffer1D<float, Stride1D.Dense> current =
                    ExecuteReductionPasses(
                        sourceBuffer,
                        count,
                        reduction,
                        kernel,
                        leases);
                executionTime +=
                    StopTiming(
                        executionStarted,
                        executionContext.CollectDiagnostics);
                executionContext.CancellationToken.ThrowIfCancellationRequested();

                long downloadStarted =
                    StartTiming(executionContext.CollectDiagnostics);
                current.View.CopyToCPU(
                    accelerator.DefaultStream,
                    partialResult.AsSpan());
                accelerator.Synchronize();
                float partial = partialResult[0];
                downloadTime +=
                    StopTiming(
                        downloadStarted,
                        executionContext.CollectDiagnostics);

                long combineStarted =
                    StartTiming(executionContext.CollectDiagnostics);
                combined = CombineReductionPartial(
                    combined,
                    partial,
                    reduction,
                    hasCombinedValue);
                hasCombinedValue = true;
                executionTime +=
                    StopTiming(
                        combineStarted,
                        executionContext.CollectDiagnostics);
            }
            finally
            {
                ReturnLeases(leases);
                leases.Clear();
            }

            offset += count;
        }

        if (reduction == ComputeReductionKind.Average)
        {
            combined /= source.Length;
        }

        return new ComputeBackendExecution<float>(
            combined,
            compilation.CompilationTime,
            executionTime,
            uploadTime,
            downloadTime,
            compilation.CacheHit,
            accelerator.Name,
            chunkPlan.ChunkCount,
            chunkPlan.ChunkElementCount,
            checked((long)source.Length * sizeof(float)),
            checked((long)chunkPlan.ChunkCount * sizeof(float)));
    }

    private static float CombineReductionPartial(
        float combined,
        float partial,
        ComputeReductionKind reduction,
        bool hasCombinedValue)
    {
        if (!hasCombinedValue)
        {
            return partial;
        }

        if (reduction is ComputeReductionKind.Sum or
            ComputeReductionKind.Average)
        {
            return combined + partial;
        }

        if (float.IsNaN(combined) || float.IsNaN(partial))
        {
            return combined + partial;
        }

        return reduction == ComputeReductionKind.Min
            ? MathF.Min(combined, partial)
            : MathF.Max(combined, partial);
    }

    private MemoryBuffer1D<float, Stride1D.Dense> ExecuteReductionPasses(
        MemoryBuffer1D<float, Stride1D.Dense> source,
        int sourceLength,
        ComputeReductionKind reduction,
        CompiledKernel kernel,
        List<GpuFloatMemoryPool.Lease> leases)
    {
        MemoryBuffer1D<float, Stride1D.Dense> current = source;
        int currentLength = sourceLength;
        int kernelReduction = reduction == ComputeReductionKind.Average
            ? (int)ComputeReductionKind.Sum
            : (int)reduction;

        while (currentLength > 1)
        {
            int outputLength =
                (currentLength + GpuKernels.ReductionElementsPerOutput - 1) /
                GpuKernels.ReductionElementsPerOutput;
            GpuFloatMemoryPool.Lease outputLease =
                memoryPool.Rent(outputLength);
            leases.Add(outputLease);
            MemoryBuffer1D<float, Stride1D.Dense> output =
                outputLease.Buffer;

            kernel.Reduction!(
                outputLength,
                current.View,
                output.View,
                currentLength,
                kernelReduction);
            current = output;
            currentLength = outputLength;
        }

        accelerator.Synchronize();
        return current;
    }

    private static void ReturnLeases(
        List<GpuFloatMemoryPool.Lease> leases)
    {
        for (int index = leases.Count - 1; index >= 0; index--)
        {
            leases[index].Dispose();
        }
    }

    private ComputeBackendExecution<float[]> ExecuteMapOnDevice(
        float[] source,
        GpuProgram program,
        CompiledKernel kernel,
        CancellationToken cancellationToken,
        bool collectDiagnostics,
        TimeSpan compilationTime,
        bool kernelCacheHit)
    {
        cancellationToken.ThrowIfCancellationRequested();

        if (source.Length == 0)
        {
            return new ComputeBackendExecution<float[]>(
                [],
                compilationTime,
                TimeSpan.Zero,
                KernelCacheHit: kernelCacheHit,
                DeviceName: accelerator.Name);
        }

        long uploadStarted = StartTiming(collectDiagnostics);
        using GpuFloatMemoryPool.Lease sourceLease =
            memoryPool.Rent(source.Length);
        MemoryBuffer1D<float, Stride1D.Dense> sourceBuffer =
            sourceLease.Buffer;
        sourceBuffer.CopyFromCPU(source);
        using MemoryBuffer1D<GpuInstruction, Stride1D.Dense> programBuffer =
            accelerator.Allocate1D(program.Instructions);
        using GpuFloatMemoryPool.Lease destinationLease =
            memoryPool.Rent(source.Length);
        MemoryBuffer1D<float, Stride1D.Dense> destinationBuffer =
            destinationLease.Buffer;
        TimeSpan uploadTime = StopTiming(uploadStarted, collectDiagnostics);

        long executionStarted = StartTiming(collectDiagnostics);
        kernel.Map!(
            accelerator.DefaultStream,
            source.Length,
            sourceBuffer.View,
            destinationBuffer.View,
            programBuffer.View,
            program.Instructions.Length);
        accelerator.Synchronize();
        TimeSpan executionTime = StopTiming(executionStarted, collectDiagnostics);
        cancellationToken.ThrowIfCancellationRequested();

        long downloadStarted = StartTiming(collectDiagnostics);
        float[] result = destinationBuffer.GetAsArray1D();
        TimeSpan downloadTime = StopTiming(downloadStarted, collectDiagnostics);

        return new ComputeBackendExecution<float[]>(
            result,
            compilationTime,
            executionTime,
            uploadTime,
            downloadTime,
            kernelCacheHit,
            accelerator.Name,
            ChunkCount: 1,
            ChunkElementCount: source.Length,
            UploadedBytes: checked((long)source.Length * sizeof(float)),
            DownloadedBytes: checked((long)result.Length * sizeof(float)));
    }

    private ComputeBackendExecution<float[]> ExecuteMapChunked(
        float[] source,
        GpuProgram program,
        CompiledKernel kernel,
        GpuChunkPlan chunkPlan,
        CancellationToken cancellationToken,
        bool collectDiagnostics,
        TimeSpan compilationTime,
        bool kernelCacheHit)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var result = new float[source.Length];
        TimeSpan uploadTime = TimeSpan.Zero;
        TimeSpan executionTime = TimeSpan.Zero;
        TimeSpan downloadTime = TimeSpan.Zero;
        using MemoryBuffer1D<GpuInstruction, Stride1D.Dense> programBuffer =
            accelerator.Allocate1D(program.Instructions);

        for (int offset = 0; offset < source.Length;)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int count =
                Math.Min(chunkPlan.ChunkElementCount, source.Length - offset);

            using GpuFloatMemoryPool.Lease sourceLease =
                memoryPool.Rent(count);
            using GpuFloatMemoryPool.Lease destinationLease =
                memoryPool.Rent(count);
            MemoryBuffer1D<float, Stride1D.Dense> sourceBuffer =
                sourceLease.Buffer;
            MemoryBuffer1D<float, Stride1D.Dense> destinationBuffer =
                destinationLease.Buffer;

            long uploadStarted = StartTiming(collectDiagnostics);
            sourceBuffer.View.CopyFromCPU(
                accelerator.DefaultStream,
                source.AsSpan(offset, count));
            accelerator.Synchronize();
            uploadTime += StopTiming(uploadStarted, collectDiagnostics);

            long executionStarted = StartTiming(collectDiagnostics);
            kernel.Map!(
                accelerator.DefaultStream,
                count,
                sourceBuffer.View,
                destinationBuffer.View,
                programBuffer.View,
                program.Instructions.Length);
            accelerator.Synchronize();
            executionTime +=
                StopTiming(executionStarted, collectDiagnostics);

            cancellationToken.ThrowIfCancellationRequested();
            long downloadStarted = StartTiming(collectDiagnostics);
            destinationBuffer.View.CopyToCPU(
                accelerator.DefaultStream,
                result.AsSpan(offset, count));
            accelerator.Synchronize();
            downloadTime += StopTiming(downloadStarted, collectDiagnostics);
            offset += count;
        }

        return new ComputeBackendExecution<float[]>(
            result,
            compilationTime,
            executionTime,
            uploadTime,
            downloadTime,
            kernelCacheHit,
            accelerator.Name,
            chunkPlan.ChunkCount,
            chunkPlan.ChunkElementCount,
            checked((long)source.Length * sizeof(float)),
            checked((long)result.Length * sizeof(float)));
    }

    private ComputeBackendExecution<float[]> ExecuteMapDoubleBuffered(
        float[] source,
        GpuProgram program,
        CompiledKernel kernel,
        GpuChunkPlan chunkPlan,
        CancellationToken cancellationToken,
        bool collectDiagnostics,
        TimeSpan compilationTime,
        bool kernelCacheHit)
    {
        cancellationToken.ThrowIfCancellationRequested();
        long pipelineStarted = StartTiming(collectDiagnostics);
        var result = new float[source.Length];
        using MemoryBuffer1D<GpuInstruction, Stride1D.Dense> programBuffer =
            accelerator.Allocate1D(program.Instructions);
        accelerator.DefaultStream.Synchronize();
        using var first =
            new StreamingMapSlot(
                accelerator,
                memoryPool,
                chunkPlan.ChunkElementCount);
        using var second =
            new StreamingMapSlot(
                accelerator,
                memoryPool,
                chunkPlan.ChunkElementCount);
        StreamingMapSlot[] slots = [first, second];

        int chunkIndex = 0;
        for (int offset = 0; offset < source.Length;)
        {
            cancellationToken.ThrowIfCancellationRequested();
            StreamingMapSlot slot = slots[chunkIndex & 1];
            CompleteStreamingMapSlot(
                slot,
                result,
                cancellationToken);

            int count =
                Math.Min(
                    chunkPlan.ChunkElementCount,
                    source.Length - offset);
            source.AsSpan(offset, count)
                .CopyTo(slot.Input.Span[..count]);
            slot.SourceBuffer.View.CopyFromPageLockedAsync(
                slot.Stream,
                slot.Input);
            kernel.Map!(
                slot.Stream,
                count,
                slot.SourceBuffer.View,
                slot.DestinationBuffer.View,
                programBuffer.View,
                program.Instructions.Length);
            slot.DestinationBuffer.View.CopyToPageLockedAsync(
                slot.Stream,
                slot.Output);
            slot.PendingOffset = offset;
            slot.PendingCount = count;

            offset += count;
            chunkIndex++;
        }

        CompleteStreamingMapSlot(first, result, cancellationToken);
        CompleteStreamingMapSlot(second, result, cancellationToken);
        TimeSpan pipelineTime =
            StopTiming(pipelineStarted, collectDiagnostics);
        long transferredElements =
            checked(
                (long)chunkPlan.ChunkCount *
                chunkPlan.ChunkElementCount);
        return new ComputeBackendExecution<float[]>(
            result,
            compilationTime,
            pipelineTime,
            KernelCacheHit: kernelCacheHit,
            DeviceName: accelerator.Name,
            ChunkCount: chunkPlan.ChunkCount,
            ChunkElementCount: chunkPlan.ChunkElementCount,
            UploadedBytes:
                checked(transferredElements * sizeof(float)),
            DownloadedBytes:
                checked(transferredElements * sizeof(float)),
            IsStreaming: true,
            StreamCount: 2);
    }

    private static void CompleteStreamingMapSlot(
        StreamingMapSlot slot,
        float[] destination,
        CancellationToken cancellationToken)
    {
        if (slot.PendingCount == 0)
        {
            return;
        }

        slot.Stream.Synchronize();
        cancellationToken.ThrowIfCancellationRequested();
        slot.Output.Span[..slot.PendingCount].CopyTo(
            destination.AsSpan(
                slot.PendingOffset,
                slot.PendingCount));
        slot.PendingCount = 0;
    }

    private ComputeBackendExecution<float[]> ExecuteMapInPlaceOnDevice(
        float[] source,
        GpuProgram program,
        CompiledKernel kernel,
        CancellationToken cancellationToken,
        bool collectDiagnostics,
        TimeSpan compilationTime,
        bool kernelCacheHit)
    {
        cancellationToken.ThrowIfCancellationRequested();

        if (source.Length == 0)
        {
            return new ComputeBackendExecution<float[]>(
                source,
                compilationTime,
                TimeSpan.Zero,
                KernelCacheHit: kernelCacheHit,
                DeviceName: accelerator.Name);
        }

        long uploadStarted = StartTiming(collectDiagnostics);
        using GpuFloatMemoryPool.Lease sourceLease =
            memoryPool.Rent(source.Length);
        MemoryBuffer1D<float, Stride1D.Dense> sourceBuffer =
            sourceLease.Buffer;
        sourceBuffer.CopyFromCPU(source);
        using MemoryBuffer1D<GpuInstruction, Stride1D.Dense> programBuffer =
            accelerator.Allocate1D(program.Instructions);
        TimeSpan uploadTime = StopTiming(uploadStarted, collectDiagnostics);

        long executionStarted = StartTiming(collectDiagnostics);
        kernel.Map!(
            accelerator.DefaultStream,
            source.Length,
            sourceBuffer.View,
            sourceBuffer.View,
            programBuffer.View,
            program.Instructions.Length);
        accelerator.Synchronize();
        TimeSpan executionTime = StopTiming(executionStarted, collectDiagnostics);
        cancellationToken.ThrowIfCancellationRequested();

        long downloadStarted = StartTiming(collectDiagnostics);
        sourceBuffer.View.CopyToCPU(
            accelerator.DefaultStream,
            source.AsSpan());
        accelerator.Synchronize();
        TimeSpan downloadTime = StopTiming(downloadStarted, collectDiagnostics);

        return new ComputeBackendExecution<float[]>(
            source,
            compilationTime,
            executionTime,
            uploadTime,
            downloadTime,
            kernelCacheHit,
            accelerator.Name,
            ChunkCount: 1,
            ChunkElementCount: source.Length,
            UploadedBytes: checked((long)source.Length * sizeof(float)),
            DownloadedBytes: checked((long)source.Length * sizeof(float)));
    }

    private ComputeBackendExecution<float[]> ExecuteMapInPlaceChunked(
        float[] source,
        GpuProgram program,
        CompiledKernel kernel,
        GpuChunkPlan chunkPlan,
        CancellationToken cancellationToken,
        bool collectDiagnostics,
        TimeSpan compilationTime,
        bool kernelCacheHit)
    {
        cancellationToken.ThrowIfCancellationRequested();
        TimeSpan uploadTime = TimeSpan.Zero;
        TimeSpan executionTime = TimeSpan.Zero;
        TimeSpan downloadTime = TimeSpan.Zero;
        using MemoryBuffer1D<GpuInstruction, Stride1D.Dense> programBuffer =
            accelerator.Allocate1D(program.Instructions);

        for (int offset = 0; offset < source.Length;)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int count =
                Math.Min(chunkPlan.ChunkElementCount, source.Length - offset);
            using GpuFloatMemoryPool.Lease sourceLease =
                memoryPool.Rent(count);
            MemoryBuffer1D<float, Stride1D.Dense> sourceBuffer =
                sourceLease.Buffer;

            long uploadStarted = StartTiming(collectDiagnostics);
            sourceBuffer.View.CopyFromCPU(
                accelerator.DefaultStream,
                source.AsSpan(offset, count));
            accelerator.Synchronize();
            uploadTime += StopTiming(uploadStarted, collectDiagnostics);

            long executionStarted = StartTiming(collectDiagnostics);
            kernel.Map!(
                accelerator.DefaultStream,
                count,
                sourceBuffer.View,
                sourceBuffer.View,
                programBuffer.View,
                program.Instructions.Length);
            accelerator.Synchronize();
            executionTime +=
                StopTiming(executionStarted, collectDiagnostics);

            cancellationToken.ThrowIfCancellationRequested();
            long downloadStarted = StartTiming(collectDiagnostics);
            sourceBuffer.View.CopyToCPU(
                accelerator.DefaultStream,
                source.AsSpan(offset, count));
            accelerator.Synchronize();
            downloadTime += StopTiming(downloadStarted, collectDiagnostics);
            offset += count;
        }

        return new ComputeBackendExecution<float[]>(
            source,
            compilationTime,
            executionTime,
            uploadTime,
            downloadTime,
            kernelCacheHit,
            accelerator.Name,
            chunkPlan.ChunkCount,
            chunkPlan.ChunkElementCount,
            checked((long)source.Length * sizeof(float)),
            checked((long)source.Length * sizeof(float)));
    }

    private ComputeBackendExecution<float[]> ExecuteZipOnDevice(
        float[] left,
        float[] right,
        GpuProgram program,
        CompiledKernel kernel,
        CancellationToken cancellationToken,
        bool collectDiagnostics,
        TimeSpan compilationTime,
        bool kernelCacheHit)
    {
        cancellationToken.ThrowIfCancellationRequested();

        if (left.Length == 0)
        {
            return new ComputeBackendExecution<float[]>(
                [],
                compilationTime,
                TimeSpan.Zero,
                KernelCacheHit: kernelCacheHit,
                DeviceName: accelerator.Name);
        }

        long uploadStarted = StartTiming(collectDiagnostics);
        using GpuFloatMemoryPool.Lease leftLease =
            memoryPool.Rent(left.Length);
        MemoryBuffer1D<float, Stride1D.Dense> leftBuffer = leftLease.Buffer;
        leftBuffer.CopyFromCPU(left);
        using GpuFloatMemoryPool.Lease rightLease =
            memoryPool.Rent(right.Length);
        MemoryBuffer1D<float, Stride1D.Dense> rightBuffer = rightLease.Buffer;
        rightBuffer.CopyFromCPU(right);
        using MemoryBuffer1D<GpuInstruction, Stride1D.Dense> programBuffer =
            accelerator.Allocate1D(program.Instructions);
        using GpuFloatMemoryPool.Lease destinationLease =
            memoryPool.Rent(left.Length);
        MemoryBuffer1D<float, Stride1D.Dense> destinationBuffer =
            destinationLease.Buffer;
        TimeSpan uploadTime = StopTiming(uploadStarted, collectDiagnostics);

        long executionStarted = StartTiming(collectDiagnostics);
        kernel.Zip!(
            left.Length,
            leftBuffer.View,
            rightBuffer.View,
            destinationBuffer.View,
            programBuffer.View,
            program.Instructions.Length);
        accelerator.Synchronize();
        TimeSpan executionTime = StopTiming(executionStarted, collectDiagnostics);
        cancellationToken.ThrowIfCancellationRequested();

        long downloadStarted = StartTiming(collectDiagnostics);
        float[] result = destinationBuffer.GetAsArray1D();
        TimeSpan downloadTime = StopTiming(downloadStarted, collectDiagnostics);

        return new ComputeBackendExecution<float[]>(
            result,
            compilationTime,
            executionTime,
            uploadTime,
            downloadTime,
            kernelCacheHit,
            accelerator.Name,
            ChunkCount: 1,
            ChunkElementCount: left.Length,
            UploadedBytes: checked((long)left.Length * sizeof(float) * 2),
            DownloadedBytes: checked((long)result.Length * sizeof(float)));
    }

    private ComputeBackendExecution<float[]> ExecuteZipChunked(
        float[] left,
        float[] right,
        GpuProgram program,
        CompiledKernel kernel,
        GpuChunkPlan chunkPlan,
        CancellationToken cancellationToken,
        bool collectDiagnostics,
        TimeSpan compilationTime,
        bool kernelCacheHit)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var result = new float[left.Length];
        TimeSpan uploadTime = TimeSpan.Zero;
        TimeSpan executionTime = TimeSpan.Zero;
        TimeSpan downloadTime = TimeSpan.Zero;
        using MemoryBuffer1D<GpuInstruction, Stride1D.Dense> programBuffer =
            accelerator.Allocate1D(program.Instructions);

        for (int offset = 0; offset < left.Length;)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int count =
                Math.Min(chunkPlan.ChunkElementCount, left.Length - offset);

            using GpuFloatMemoryPool.Lease leftLease =
                memoryPool.Rent(count);
            using GpuFloatMemoryPool.Lease rightLease =
                memoryPool.Rent(count);
            using GpuFloatMemoryPool.Lease destinationLease =
                memoryPool.Rent(count);
            MemoryBuffer1D<float, Stride1D.Dense> leftBuffer =
                leftLease.Buffer;
            MemoryBuffer1D<float, Stride1D.Dense> rightBuffer =
                rightLease.Buffer;
            MemoryBuffer1D<float, Stride1D.Dense> destinationBuffer =
                destinationLease.Buffer;

            long uploadStarted = StartTiming(collectDiagnostics);
            leftBuffer.View.CopyFromCPU(
                accelerator.DefaultStream,
                left.AsSpan(offset, count));
            rightBuffer.View.CopyFromCPU(
                accelerator.DefaultStream,
                right.AsSpan(offset, count));
            accelerator.Synchronize();
            uploadTime += StopTiming(uploadStarted, collectDiagnostics);

            long executionStarted = StartTiming(collectDiagnostics);
            kernel.Zip!(
                count,
                leftBuffer.View,
                rightBuffer.View,
                destinationBuffer.View,
                programBuffer.View,
                program.Instructions.Length);
            accelerator.Synchronize();
            executionTime +=
                StopTiming(executionStarted, collectDiagnostics);

            cancellationToken.ThrowIfCancellationRequested();
            long downloadStarted = StartTiming(collectDiagnostics);
            destinationBuffer.View.CopyToCPU(
                accelerator.DefaultStream,
                result.AsSpan(offset, count));
            accelerator.Synchronize();
            downloadTime += StopTiming(downloadStarted, collectDiagnostics);
            offset += count;
        }

        return new ComputeBackendExecution<float[]>(
            result,
            compilationTime,
            executionTime,
            uploadTime,
            downloadTime,
            kernelCacheHit,
            accelerator.Name,
            chunkPlan.ChunkCount,
            chunkPlan.ChunkElementCount,
            checked((long)left.Length * sizeof(float) * 2),
            checked((long)result.Length * sizeof(float)));
    }

    private ComputeBackendExecution<float[]> ExecuteZipInPlaceOnDevice(
        float[] target,
        float[] right,
        GpuProgram program,
        CompiledKernel kernel,
        CancellationToken cancellationToken,
        bool collectDiagnostics,
        TimeSpan compilationTime,
        bool kernelCacheHit)
    {
        cancellationToken.ThrowIfCancellationRequested();

        if (target.Length == 0)
        {
            return new ComputeBackendExecution<float[]>(
                target,
                compilationTime,
                TimeSpan.Zero,
                KernelCacheHit: kernelCacheHit,
                DeviceName: accelerator.Name);
        }

        long uploadStarted = StartTiming(collectDiagnostics);
        using GpuFloatMemoryPool.Lease targetLease =
            memoryPool.Rent(target.Length);
        using GpuFloatMemoryPool.Lease rightLease =
            memoryPool.Rent(right.Length);
        MemoryBuffer1D<float, Stride1D.Dense> targetBuffer =
            targetLease.Buffer;
        MemoryBuffer1D<float, Stride1D.Dense> rightBuffer =
            rightLease.Buffer;
        targetBuffer.CopyFromCPU(target);
        rightBuffer.CopyFromCPU(right);
        using MemoryBuffer1D<GpuInstruction, Stride1D.Dense> programBuffer =
            accelerator.Allocate1D(program.Instructions);
        TimeSpan uploadTime = StopTiming(uploadStarted, collectDiagnostics);

        long executionStarted = StartTiming(collectDiagnostics);
        kernel.Zip!(
            target.Length,
            targetBuffer.View,
            rightBuffer.View,
            targetBuffer.View,
            programBuffer.View,
            program.Instructions.Length);
        accelerator.Synchronize();
        TimeSpan executionTime =
            StopTiming(executionStarted, collectDiagnostics);
        cancellationToken.ThrowIfCancellationRequested();

        long downloadStarted = StartTiming(collectDiagnostics);
        targetBuffer.View.CopyToCPU(
            accelerator.DefaultStream,
            target.AsSpan());
        accelerator.Synchronize();
        TimeSpan downloadTime =
            StopTiming(downloadStarted, collectDiagnostics);

        return new ComputeBackendExecution<float[]>(
            target,
            compilationTime,
            executionTime,
            uploadTime,
            downloadTime,
            kernelCacheHit,
            accelerator.Name,
            ChunkCount: 1,
            ChunkElementCount: target.Length,
            UploadedBytes: checked((long)target.Length * sizeof(float) * 2),
            DownloadedBytes: checked((long)target.Length * sizeof(float)));
    }

    private ComputeBackendExecution<float[]> ExecuteZipInPlaceChunked(
        float[] target,
        float[] right,
        GpuProgram program,
        CompiledKernel kernel,
        GpuChunkPlan chunkPlan,
        CancellationToken cancellationToken,
        bool collectDiagnostics,
        TimeSpan compilationTime,
        bool kernelCacheHit)
    {
        cancellationToken.ThrowIfCancellationRequested();
        TimeSpan uploadTime = TimeSpan.Zero;
        TimeSpan executionTime = TimeSpan.Zero;
        TimeSpan downloadTime = TimeSpan.Zero;
        using MemoryBuffer1D<GpuInstruction, Stride1D.Dense> programBuffer =
            accelerator.Allocate1D(program.Instructions);

        for (int offset = 0; offset < target.Length;)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int count =
                Math.Min(chunkPlan.ChunkElementCount, target.Length - offset);
            using GpuFloatMemoryPool.Lease targetLease =
                memoryPool.Rent(count);
            using GpuFloatMemoryPool.Lease rightLease =
                memoryPool.Rent(count);
            MemoryBuffer1D<float, Stride1D.Dense> targetBuffer =
                targetLease.Buffer;
            MemoryBuffer1D<float, Stride1D.Dense> rightBuffer =
                rightLease.Buffer;

            long uploadStarted = StartTiming(collectDiagnostics);
            targetBuffer.View.CopyFromCPU(
                accelerator.DefaultStream,
                target.AsSpan(offset, count));
            rightBuffer.View.CopyFromCPU(
                accelerator.DefaultStream,
                right.AsSpan(offset, count));
            accelerator.Synchronize();
            uploadTime += StopTiming(uploadStarted, collectDiagnostics);

            long executionStarted = StartTiming(collectDiagnostics);
            kernel.Zip!(
                count,
                targetBuffer.View,
                rightBuffer.View,
                targetBuffer.View,
                programBuffer.View,
                program.Instructions.Length);
            accelerator.Synchronize();
            executionTime +=
                StopTiming(executionStarted, collectDiagnostics);

            cancellationToken.ThrowIfCancellationRequested();
            long downloadStarted = StartTiming(collectDiagnostics);
            targetBuffer.View.CopyToCPU(
                accelerator.DefaultStream,
                target.AsSpan(offset, count));
            accelerator.Synchronize();
            downloadTime += StopTiming(downloadStarted, collectDiagnostics);
            offset += count;
        }

        return new ComputeBackendExecution<float[]>(
            target,
            compilationTime,
            executionTime,
            uploadTime,
            downloadTime,
            kernelCacheHit,
            accelerator.Name,
            chunkPlan.ChunkCount,
            chunkPlan.ChunkElementCount,
            checked((long)target.Length * sizeof(float) * 2),
            checked((long)target.Length * sizeof(float)));
    }

    internal void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref disposed) != 0,
            this);
    }

    internal long GetAutomaticMemoryBudget(long? requestedBudget)
    {
        ThrowIfDisposed();
        long safeDeviceBudget =
            accelerator.MemorySize / AutoMemoryBudgetDenominator *
            AutoMemoryBudgetNumerator;
        return requestedBudget is long requested
            ? Math.Min(requested, safeDeviceBudget)
            : safeDeviceBudget;
    }

    internal void Download<T>(
        ComputeBuffer<T> source,
        Span<T> destination)
        where T : unmanaged
    {
        ThrowIfDisposed();
        ValidateOwnedBuffer(source);
        ValidateElementType(typeof(T));

        ComputeBufferNode<T> sourceNode = source.AcquireNode();
        try
        {
            if (destination.Length < sourceNode.Length)
            {
                throw new ArgumentException(
                    $"The destination span must contain at least " +
                    $"{sourceNode.Length} elements, but contains " +
                    $"{destination.Length}.",
                    nameof(destination));
            }

            if (sourceNode.Length == 0)
            {
                return;
            }

            sourceNode.GetBuffer().View.CopyToCPU(
                accelerator.DefaultStream,
                destination[..sourceNode.Length]);
        }
        finally
        {
            sourceNode.Release();
        }
    }

    internal ComputeBuffer<T> Select<T>(
        ComputeBuffer<T> source,
        Expression<Func<T, T>> expression)
        where T : unmanaged
    {
        ThrowIfDisposed();
        ValidateOwnedBuffer(source);
        ValidateElementType(typeof(T));

        ComputeBufferNode<T> sourceNode = source.AcquireNode();
        try
        {
            ComputeExpressionPlan plan =
                StrictComputeOptimizer.Optimize(
                    ComputeExpressionParser.Parse(expression));
            return new ComputeBuffer<T>(
                this,
                new MapBufferNode<T>(this, sourceNode, plan));
        }
        catch
        {
            sourceNode.Release();
            throw;
        }
    }

    internal void SelectInPlace<T>(
        ComputeBuffer<T> source,
        Expression<Func<T, T>> expression)
        where T : unmanaged
    {
        ThrowIfDisposed();
        ValidateOwnedBuffer(source);
        ValidateElementType(typeof(T));
        ComputeExpressionPlan plan =
            StrictComputeOptimizer.Optimize(
                ComputeExpressionParser.Parse(expression));

        source.ReplaceNode(
            sourceNode =>
                new InPlaceMapBufferNode<T>(
                    this,
                    sourceNode,
                    plan));
    }

    internal ComputeBuffer<T> Zip<T>(
        ComputeBuffer<T> left,
        ComputeBuffer<T> right,
        Expression<Func<T, T, T>> expression)
        where T : unmanaged
    {
        ThrowIfDisposed();
        ValidateOwnedBuffer(left);
        ValidateOwnedBuffer(right);
        ValidateElementType(typeof(T));

        ComputeBufferNode<T> leftNode = left.AcquireNode();
        ComputeBufferNode<T>? rightNode = null;
        try
        {
            rightNode = right.AcquireNode();
            if (leftNode.Length != rightNode.Length)
            {
                throw new ComputeBufferMismatchException(
                    $"Zip requires buffers of equal length, but received " +
                    $"{leftNode.Length} and {rightNode.Length}.");
            }

            ComputeExpressionPlan plan =
                StrictComputeOptimizer.Optimize(
                    ComputeExpressionParser.Parse(expression));
            ComputeBuffer<T> result = new(
                this,
                new ZipBufferNode<T>(
                    this,
                    leftNode,
                    rightNode,
                    plan));
            rightNode = null;
            leftNode = null!;
            return result;
        }
        finally
        {
            leftNode?.Release();
            rightNode?.Release();
        }
    }

    internal void ZipInPlace<T>(
        ComputeBuffer<T> left,
        ComputeBuffer<T> right,
        Expression<Func<T, T, T>> expression)
        where T : unmanaged
    {
        ThrowIfDisposed();
        ValidateOwnedBuffer(left);
        ValidateOwnedBuffer(right);
        ValidateElementType(typeof(T));
        ComputeExpressionPlan plan =
            StrictComputeOptimizer.Optimize(
                ComputeExpressionParser.Parse(expression));

        left.ReplaceNode(
            leftNode =>
            {
                ComputeBufferNode<T> rightNode = right.AcquireNode();
                if (leftNode.Length != rightNode.Length)
                {
                    rightNode.Release();
                    throw new ComputeBufferMismatchException(
                        $"ZipInPlace requires buffers of equal length, but " +
                        $"received {leftNode.Length} and {rightNode.Length}.");
                }

                return new InPlaceZipBufferNode<T>(
                    this,
                    leftNode,
                    rightNode,
                    plan);
            });
    }

    internal MemoryBuffer1D<T, Stride1D.Dense> ExecuteGraphMap<T>(
        MemoryBuffer1D<T, Stride1D.Dense> source,
        int length,
        ComputeExpressionPlan plan)
        where T : unmanaged
    {
        ThrowIfDisposed();
        ValidateElementType(typeof(T));
        GpuProgram program = GetOrCreateProgram(plan, out _);
        CompiledKernel kernel = GetOrCompileKernel(ComputeKernelKind.Map, out _);
        var floatSource =
            (MemoryBuffer1D<float, Stride1D.Dense>)(object)source;
        MemoryBuffer1D<float, Stride1D.Dense> destination =
            accelerator.Allocate1D<float>(length);
        try
        {
            using MemoryBuffer1D<GpuInstruction, Stride1D.Dense> programBuffer =
                accelerator.Allocate1D(program.Instructions);
            kernel.Map!(
                accelerator.DefaultStream,
                length,
                floatSource.View,
                destination.View,
                programBuffer.View,
                program.Instructions.Length);
            accelerator.Synchronize();
            return (MemoryBuffer1D<T, Stride1D.Dense>)(object)destination;
        }
        catch
        {
            destination.Dispose();
            throw;
        }
    }

    internal void ExecuteGraphMapInPlace<T>(
        MemoryBuffer1D<T, Stride1D.Dense> buffer,
        int length,
        ComputeExpressionPlan plan)
        where T : unmanaged
    {
        ThrowIfDisposed();
        ValidateElementType(typeof(T));
        GpuProgram program = GetOrCreateProgram(plan, out _);
        CompiledKernel kernel =
            GetOrCompileKernel(ComputeKernelKind.Map, out _);
        var floatBuffer =
            (MemoryBuffer1D<float, Stride1D.Dense>)(object)buffer;
        using MemoryBuffer1D<GpuInstruction, Stride1D.Dense> programBuffer =
            accelerator.Allocate1D(program.Instructions);
        kernel.Map!(
            accelerator.DefaultStream,
            length,
            floatBuffer.View,
            floatBuffer.View,
            programBuffer.View,
            program.Instructions.Length);
        accelerator.Synchronize();
    }

    internal MemoryBuffer1D<T, Stride1D.Dense> ExecuteGraphZip<T>(
        MemoryBuffer1D<T, Stride1D.Dense> left,
        MemoryBuffer1D<T, Stride1D.Dense> right,
        int length,
        ComputeExpressionPlan plan)
        where T : unmanaged
    {
        ThrowIfDisposed();
        ValidateElementType(typeof(T));
        GpuProgram program = GetOrCreateProgram(plan, out _);
        CompiledKernel kernel = GetOrCompileKernel(ComputeKernelKind.Zip, out _);
        var floatLeft =
            (MemoryBuffer1D<float, Stride1D.Dense>)(object)left;
        var floatRight =
            (MemoryBuffer1D<float, Stride1D.Dense>)(object)right;
        MemoryBuffer1D<float, Stride1D.Dense> destination =
            accelerator.Allocate1D<float>(length);

        try
        {
            using MemoryBuffer1D<GpuInstruction, Stride1D.Dense> programBuffer =
                accelerator.Allocate1D(program.Instructions);
            kernel.Zip!(
                length,
                floatLeft.View,
                floatRight.View,
                destination.View,
                programBuffer.View,
                program.Instructions.Length);
            accelerator.Synchronize();
            return (MemoryBuffer1D<T, Stride1D.Dense>)(object)destination;
        }
        catch
        {
            destination.Dispose();
            throw;
        }
    }

    internal void ExecuteGraphZipInPlace<T>(
        MemoryBuffer1D<T, Stride1D.Dense> left,
        MemoryBuffer1D<T, Stride1D.Dense> right,
        int length,
        ComputeExpressionPlan plan)
        where T : unmanaged
    {
        ThrowIfDisposed();
        ValidateElementType(typeof(T));
        GpuProgram program = GetOrCreateProgram(plan, out _);
        CompiledKernel kernel =
            GetOrCompileKernel(ComputeKernelKind.Zip, out _);
        var floatLeft =
            (MemoryBuffer1D<float, Stride1D.Dense>)(object)left;
        var floatRight =
            (MemoryBuffer1D<float, Stride1D.Dense>)(object)right;
        using MemoryBuffer1D<GpuInstruction, Stride1D.Dense> programBuffer =
            accelerator.Allocate1D(program.Instructions);
        kernel.Zip!(
            length,
            floatLeft.View,
            floatRight.View,
            floatLeft.View,
            programBuffer.View,
            program.Instructions.Length);
        accelerator.Synchronize();
    }

    internal long GraphCopyOnWriteCount =>
        Interlocked.Read(ref graphCopyOnWriteCount);

    internal long GraphInPlaceReuseCount =>
        Interlocked.Read(ref graphInPlaceReuseCount);

    internal void RecordGraphCopyOnWrite() =>
        Interlocked.Increment(ref graphCopyOnWriteCount);

    internal void RecordGraphInPlaceReuse() =>
        Interlocked.Increment(ref graphInPlaceReuseCount);

    internal T Reduce<T>(
        ComputeBuffer<T> source,
        ComputeReductionKind reduction)
        where T : unmanaged
    {
        ThrowIfDisposed();
        ValidateOwnedBuffer(source);
        ValidateElementType(typeof(T));

        ComputeBufferNode<T> sourceNode = source.AcquireNode();
        try
        {
            if (sourceNode.Length == 0)
            {
                if (reduction == ComputeReductionKind.Sum)
                {
                    return (T)(object)0.0f;
                }

                throw new InvalidOperationException(
                    $"Cannot compute {reduction} for an empty buffer.");
            }

            CompiledKernel kernel = GetOrCompileKernel(
                ComputeKernelKind.Reduction,
                out _);
            var sourceBuffer =
                (MemoryBuffer1D<float, Stride1D.Dense>)(object)
                sourceNode.GetBuffer();
            var leases = new List<GpuFloatMemoryPool.Lease>();

            try
            {
                MemoryBuffer1D<float, Stride1D.Dense> resultBuffer =
                    ExecuteReductionPasses(
                        sourceBuffer,
                        sourceNode.Length,
                        reduction,
                        kernel,
                        leases);
                float result = resultBuffer.GetAsArray1D()[0];
                if (reduction == ComputeReductionKind.Average)
                {
                    result /= sourceNode.Length;
                }

                return (T)(object)result;
            }
            finally
            {
                ReturnLeases(leases);
            }
        }
        finally
        {
            sourceNode.Release();
        }
    }

    private ComputeCompilationResult PrecompileCore(
        LambdaExpression expression,
        ComputeKernelKind kind,
        Type elementType)
    {
        ArgumentNullException.ThrowIfNull(expression);
        ThrowIfDisposed();
        ValidateElementType(elementType);

        long planningStarted = Stopwatch.GetTimestamp();
        _ = GetOrCreateProgram(expression, out bool programCacheHit);
        TimeSpan planningTime = Stopwatch.GetElapsedTime(planningStarted);

        _ = GetOrCompileKernel(kind, out KernelCompilation compilation);
        bool cacheHit = programCacheHit && compilation.CacheHit;

        return new ComputeCompilationResult(
            cacheHit,
            planningTime,
            compilation.CompilationTime,
            ComputeBackendKind.Gpu,
            accelerator.Name);
    }

    private ComputeCompilationResult PrecompileReductionCore(
        ComputeReductionKind reduction,
        Type elementType)
    {
        ThrowIfDisposed();
        ValidateElementType(elementType);
        _ = reduction switch
        {
            ComputeReductionKind.Sum or
            ComputeReductionKind.Min or
            ComputeReductionKind.Max or
            ComputeReductionKind.Average => reduction,
            _ => throw new ArgumentOutOfRangeException(nameof(reduction))
        };

        _ = GetOrCompileKernel(
            ComputeKernelKind.Reduction,
            out KernelCompilation compilation);
        return new ComputeCompilationResult(
            compilation.CacheHit,
            TimeSpan.Zero,
            compilation.CompilationTime,
            ComputeBackendKind.Gpu,
            accelerator.Name);
    }

    private ComputeCompilationResult PrecompileHistogramCore(
        Type elementType)
    {
        ThrowIfDisposed();
        ValidateElementType(elementType);
        _ = GetOrCompileKernel(
            ComputeKernelKind.Histogram,
            out KernelCompilation compilation);
        return new ComputeCompilationResult(
            compilation.CacheHit,
            TimeSpan.Zero,
            compilation.CompilationTime,
            ComputeBackendKind.Gpu,
            accelerator.Name);
    }

    private GpuProgram GetOrCreateProgram(
        LambdaExpression expression,
        out bool cacheHit) =>
        GetOrCreateProgram(
            StrictComputeOptimizer.Optimize(ComputeExpressionParser.Parse(expression)),
            out cacheHit);

    private GpuProgram GetOrCreateProgram(
        ComputeExpressionPlan plan,
        out bool cacheHit)
    {
        GpuProgram candidate = GpuProgramCompiler.Compile(plan);
        GpuProgram result = programs.GetOrAdd(candidate.StructuralKey, candidate);
        cacheHit = !ReferenceEquals(result, candidate);
        return result;
    }

    private CompiledKernel GetOrCompileKernel(
        ComputeKernelKind kind,
        out KernelCompilation compilation)
    {
        Lazy<CompiledKernel> candidate = new(
            () =>
            {
                long started = Stopwatch.GetTimestamp();
                CompiledKernel result = CompileKernel(kind);
                result.CompilationTime = Stopwatch.GetElapsedTime(started);
                return result;
            },
            LazyThreadSafetyMode.ExecutionAndPublication);

        Lazy<CompiledKernel> lazy = kernels.GetOrAdd(kind, candidate);

        CompiledKernel kernel = lazy.Value;
        bool cacheHit = !ReferenceEquals(lazy, candidate);
        compilation = new KernelCompilation(
            cacheHit,
            cacheHit ? TimeSpan.Zero : kernel.CompilationTime);
        return kernel;
    }

    private CompiledKernel CompileKernel(ComputeKernelKind kind) =>
        kind switch
        {
            ComputeKernelKind.Map => new CompiledKernel
            {
                Map = accelerator.LoadAutoGroupedKernel<
                    Index1D,
                    ArrayView<float>,
                    ArrayView<float>,
                    ArrayView<GpuInstruction>,
                    int>(GpuKernels.Map)
            },
            ComputeKernelKind.Zip => new CompiledKernel
            {
                Zip = accelerator.LoadAutoGroupedStreamKernel<
                    Index1D,
                    ArrayView<float>,
                    ArrayView<float>,
                    ArrayView<float>,
                    ArrayView<GpuInstruction>,
                    int>(GpuKernels.Zip)
            },
            ComputeKernelKind.Reduction => new CompiledKernel
            {
                Reduction = accelerator.LoadAutoGroupedStreamKernel<
                    Index1D,
                    ArrayView<float>,
                    ArrayView<float>,
                    int,
                    int>(GpuKernels.Reduce)
            },
            ComputeKernelKind.Histogram => new CompiledKernel
            {
                Histogram = accelerator.LoadAutoGroupedStreamKernel<
                    Index1D,
                    ArrayView<float>,
                    ArrayView<int>,
                    int,
                    float,
                    float,
                    float>(GpuKernels.Histogram)
            },
            _ => throw new ArgumentOutOfRangeException(nameof(kind))
        };

    private static Device SelectDevice(IlGpuContext context, int? acceleratorIndex)
    {
        if (acceleratorIndex is int index)
        {
            if ((uint)index >= (uint)context.Devices.Length)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(ComputeContextOptions.AcceleratorIndex),
                    index,
                    $"Accelerator index must be between 0 and {context.Devices.Length - 1}.");
            }

            return context.Devices[index];
        }

        return context.Devices.FirstOrDefault(
                   device => string.Equals(
                       device.AcceleratorType.ToString(),
                       "Cuda",
                       StringComparison.OrdinalIgnoreCase))
               ?? context.GetPreferredDevice(preferCPU: false);
    }

    private static void ValidateElementType(Type elementType)
    {
        if (elementType != typeof(float))
        {
            throw new NotSupportedException(
                $"GPU execution currently supports float, not '{elementType.Name}'.");
        }
    }

    private void ValidateOwnedBuffer<T>(ComputeBuffer<T> buffer)
        where T : unmanaged
    {
        if (!ReferenceEquals(buffer.Context, this))
        {
            throw new ComputeBufferMismatchException(
                "GPU buffers must belong to the same ComputeContext.");
        }
    }

    private static long StartTiming(bool enabled) =>
        enabled ? Stopwatch.GetTimestamp() : 0L;

    private static TimeSpan StopTiming(long started, bool enabled) =>
        enabled ? Stopwatch.GetElapsedTime(started) : TimeSpan.Zero;

    internal sealed class CompiledKernel
    {
        internal Action<
            AcceleratorStream,
            Index1D,
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<GpuInstruction>,
            int>? Map
        { get; init; }

        internal Action<
            Index1D,
            ArrayView<float>,
            ArrayView<float>,
            int,
            int>? Reduction
        { get; init; }

        internal Action<
            Index1D,
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<GpuInstruction>,
            int>? Zip
        { get; init; }

        internal Action<
            Index1D,
            ArrayView<float>,
            ArrayView<int>,
            int,
            float,
            float,
            float>? Histogram
        { get; init; }

        internal TimeSpan CompilationTime { get; set; }
    }

    private sealed class StreamingMapSlot : IDisposable
    {
        private GpuFloatMemoryPool.Lease? sourceLease;
        private GpuFloatMemoryPool.Lease? destinationLease;

        internal StreamingMapSlot(
            Accelerator accelerator,
            GpuFloatMemoryPool memoryPool,
            int chunkElementCount)
        {
            Stream = accelerator.CreateStream();
            try
            {
                sourceLease = memoryPool.Rent(chunkElementCount);
                destinationLease = memoryPool.Rent(chunkElementCount);
                Input =
                    accelerator.AllocatePageLocked1D<float>(
                        chunkElementCount,
                        uninitialized: true);
                Output =
                    accelerator.AllocatePageLocked1D<float>(
                        chunkElementCount,
                        uninitialized: true);
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        internal AcceleratorStream Stream { get; }

        internal MemoryBuffer1D<float, Stride1D.Dense> SourceBuffer =>
            sourceLease!.Buffer;

        internal MemoryBuffer1D<float, Stride1D.Dense> DestinationBuffer =>
            destinationLease!.Buffer;

        internal PageLockedArray1D<float> Input { get; private set; } = null!;

        internal PageLockedArray1D<float> Output { get; private set; } = null!;

        internal int PendingOffset { get; set; }

        internal int PendingCount { get; set; }

        public void Dispose()
        {
            Stream.Synchronize();
            Output?.Dispose();
            Input?.Dispose();
            destinationLease?.Dispose();
            sourceLease?.Dispose();
            destinationLease = null;
            sourceLease = null;
            Stream.Dispose();
        }
    }

    private readonly record struct KernelCompilation(
        bool CacheHit,
        TimeSpan CompilationTime);
}
