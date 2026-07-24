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

    /// <summary>Synchronously prepares several map, zip, or reduction kernels.</summary>
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

            results[index] = descriptor.Kind == ComputeKernelKind.Reduction
                ? PrecompileReductionCore(
                    descriptor.Reduction!.Value,
                    descriptor.ElementType)
                : PrecompileCore(
                    descriptor.Expression!,
                    descriptor.Kind,
                    descriptor.ElementType);
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

        return ExecuteMapOnDevice(
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

        return ExecuteZipOnDevice(
            left,
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

        if (source.Length == 0)
        {
            return new ComputeBackendExecution<float>(
                0f,
                compilation.CompilationTime,
                TimeSpan.Zero,
                KernelCacheHit: compilation.CacheHit,
                DeviceName: accelerator.Name);
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
                accelerator.Name);
        }
        finally
        {
            ReturnLeases(leases);
        }
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
            accelerator.Name);
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
            accelerator.Name);
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
                Map = accelerator.LoadAutoGroupedStreamKernel<
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

        internal TimeSpan CompilationTime { get; set; }
    }

    private readonly record struct KernelCompilation(
        bool CacheHit,
        TimeSpan CompilationTime);
}
