using System.Collections.Concurrent;
using System.Numerics;
using FastCompute.Expressions;
using FastCompute.Gpu;
using ILGPU;
using ILGPU.Runtime;

namespace FastCompute;

public sealed partial class ComputeContext
{
    private readonly ConcurrentDictionary<
        (Type Type, ComputeKernelKind Kind),
        Lazy<object>> numericKernels = new();

    private PreparedCompute<T> PrepareNumeric<T>(
        System.Linq.Expressions.Expression<Func<T, T>> expression)
        where T : unmanaged, INumber<T>
    {
        NumericExpressionProgram<T> program =
            NumericExpressionParser.Parse<T>(expression);
        ValidateNumericProgram(program);
        _ = PrecompileNumericTemplate(
            typeof(T),
            ComputeKernelKind.Map);
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = this
        };
        return new PreparedCompute<T>(
            this,
            source => ExecuteNumericMap(source, program, options));
    }

    private ComputeCompilationResult PrecompileNumericExpression(
        System.Linq.Expressions.LambdaExpression expression,
        ComputeKernelKind kind,
        Type elementType)
    {
        long planningStarted = System.Diagnostics.Stopwatch.GetTimestamp();
        if (elementType == typeof(double))
        {
            ValidateNumericProgram(
                NumericExpressionParser.Parse<double>(expression));
        }
        else
        {
            ValidateNumericProgram(
                NumericExpressionParser.Parse<int>(expression));
        }

        TimeSpan planningTime =
            System.Diagnostics.Stopwatch.GetElapsedTime(planningStarted);
        ComputeCompilationResult template =
            PrecompileNumericTemplate(elementType, kind);
        return template with { PlanningTime = planningTime };
    }

    private ComputeCompilationResult PrecompileNumericTemplate(
        Type elementType,
        ComputeKernelKind kind)
    {
        var key = (Type: elementType, Kind: kind);
        bool cacheHit = numericKernels.ContainsKey(key);
        long compilationStarted =
            System.Diagnostics.Stopwatch.GetTimestamp();
        Lazy<object> lazy = numericKernels.GetOrAdd(
            key,
            item => new Lazy<object>(
                () => CompileNumericKernel(item.Type, item.Kind),
                LazyThreadSafetyMode.ExecutionAndPublication));
        _ = lazy.Value;
        TimeSpan compilationTime = cacheHit
            ? TimeSpan.Zero
            : System.Diagnostics.Stopwatch.GetElapsedTime(
                compilationStarted);
        return new ComputeCompilationResult(
            cacheHit,
            TimeSpan.Zero,
            compilationTime,
            ComputeBackendKind.Gpu,
            accelerator.Name);
    }

    private ComputeBuffer<T> SelectNumericBuffer<T>(
        ComputeBuffer<T> source,
        System.Linq.Expressions.Expression<Func<T, T>> expression)
        where T : unmanaged, INumber<T>
    {
        NumericExpressionProgram<T> program =
            NumericExpressionParser.Parse<T>(expression);
        ValidateNumericProgram(program);
        ComputeBufferNode<T> sourceNode = source.AcquireNode();
        try
        {
            if (sourceNode.Length == 0)
            {
                return new ComputeBuffer<T>(
                    this,
                    new BufferSourceNode<T>(this, length: 0));
            }

            MemoryBuffer1D<T, Stride1D.Dense> result =
                ExecuteResidentNumericMap(
                    sourceNode.GetBuffer(),
                    sourceNode.Length,
                    program);
            return new ComputeBuffer<T>(
                this,
                new BufferSourceNode<T>(this, result));
        }
        finally
        {
            sourceNode.Release();
        }
    }

    private void SelectNumericBufferInPlace<T>(
        ComputeBuffer<T> source,
        System.Linq.Expressions.Expression<Func<T, T>> expression)
        where T : unmanaged, INumber<T>
    {
        NumericExpressionProgram<T> program =
            NumericExpressionParser.Parse<T>(expression);
        ValidateNumericProgram(program);
        source.ReplaceNode(
            sourceNode =>
            {
                try
                {
                    if (sourceNode.Length == 0)
                    {
                        return new BufferSourceNode<T>(
                            this,
                            length: 0);
                    }

                    MemoryBuffer1D<T, Stride1D.Dense> result =
                        ExecuteResidentNumericMap(
                            sourceNode.GetBuffer(),
                            sourceNode.Length,
                            program);
                    return new BufferSourceNode<T>(this, result);
                }
                finally
                {
                    sourceNode.Release();
                }
            });
    }

    private ComputeBuffer<T> ZipNumericBuffers<T>(
        ComputeBuffer<T> left,
        ComputeBuffer<T> right,
        System.Linq.Expressions.Expression<Func<T, T, T>> expression)
        where T : unmanaged, INumber<T>
    {
        NumericExpressionProgram<T> program =
            NumericExpressionParser.Parse<T>(expression);
        ValidateNumericProgram(program);
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

            if (leftNode.Length == 0)
            {
                return new ComputeBuffer<T>(
                    this,
                    new BufferSourceNode<T>(this, length: 0));
            }

            MemoryBuffer1D<T, Stride1D.Dense> result =
                ExecuteResidentNumericZip(
                    leftNode.GetBuffer(),
                    rightNode.GetBuffer(),
                    leftNode.Length,
                    program);
            return new ComputeBuffer<T>(
                this,
                new BufferSourceNode<T>(this, result));
        }
        finally
        {
            leftNode.Release();
            rightNode?.Release();
        }
    }

    private void ZipNumericBuffersInPlace<T>(
        ComputeBuffer<T> left,
        ComputeBuffer<T> right,
        System.Linq.Expressions.Expression<Func<T, T, T>> expression)
        where T : unmanaged, INumber<T>
    {
        NumericExpressionProgram<T> program =
            NumericExpressionParser.Parse<T>(expression);
        ValidateNumericProgram(program);
        left.ReplaceNode(
            leftNode =>
            {
                ComputeBufferNode<T> rightNode = right.AcquireNode();
                try
                {
                    if (leftNode.Length != rightNode.Length)
                    {
                        throw new ComputeBufferMismatchException(
                            $"ZipInPlace requires buffers of equal length, " +
                            $"but received {leftNode.Length} and " +
                            $"{rightNode.Length}.");
                    }

                    if (leftNode.Length == 0)
                    {
                        return new BufferSourceNode<T>(
                            this,
                            length: 0);
                    }

                    MemoryBuffer1D<T, Stride1D.Dense> result =
                        ExecuteResidentNumericZip(
                            leftNode.GetBuffer(),
                            rightNode.GetBuffer(),
                            leftNode.Length,
                            program);
                    return new BufferSourceNode<T>(this, result);
                }
                finally
                {
                    leftNode.Release();
                    rightNode.Release();
                }
            });
    }

    private T ReduceNumericBuffer<T>(
        ComputeBuffer<T> source,
        ComputeReductionKind reduction)
        where T : unmanaged, INumber<T>
    {
        ComputeBufferNode<T> sourceNode = source.AcquireNode();
        try
        {
            if (sourceNode.Length == 0)
            {
                if (reduction == ComputeReductionKind.Sum)
                {
                    return T.Zero;
                }

                throw new InvalidOperationException(
                    $"Cannot compute {reduction} for an empty buffer.");
            }

            ComputeReductionKind effective = reduction ==
                ComputeReductionKind.Average
                ? ComputeReductionKind.Sum
                : reduction;
            T result;
            if (typeof(T) == typeof(double))
            {
                var kernel = GetNumericKernel<
                    Action<
                        Index1D,
                        ArrayView<double>,
                        ArrayView<double>,
                        int,
                        int>>(
                    typeof(double),
                    ComputeKernelKind.Reduction);
                double reduced = ReduceDoubleBuffer(
                    (MemoryBuffer1D<double, Stride1D.Dense>)(object)
                    sourceNode.GetBuffer(),
                    sourceNode.Length,
                    effective,
                    kernel);
                result = (T)(object)reduced;
            }
            else
            {
                var kernel = GetNumericKernel<
                    Action<
                        Index1D,
                        ArrayView<int>,
                        ArrayView<int>,
                        int,
                        int>>(
                    typeof(int),
                    ComputeKernelKind.Reduction);
                int reduced = ReduceIntBuffer(
                    (MemoryBuffer1D<int, Stride1D.Dense>)(object)
                    sourceNode.GetBuffer(),
                    sourceNode.Length,
                    effective,
                    kernel);
                result = (T)(object)reduced;
            }

            return reduction == ComputeReductionKind.Average
                ? result / T.CreateChecked(sourceNode.Length)
                : result;
        }
        finally
        {
            sourceNode.Release();
        }
    }

    private MemoryBuffer1D<T, Stride1D.Dense>
        ExecuteResidentNumericMap<T>(
            MemoryBuffer1D<T, Stride1D.Dense> source,
            int length,
            NumericExpressionProgram<T> program)
        where T : unmanaged, INumber<T>
    {
        if (typeof(T) == typeof(double))
        {
            var instructions = ((NumericExpressionProgram<double>)(object)
                    program)
                .Instructions
                .Select(
                    item => new DoubleGpuInstruction(
                        (int)item.OpCode,
                        item.Operand))
                .ToArray();
            var kernel = GetNumericKernel<
                Action<
                    Index1D,
                    ArrayView<double>,
                    ArrayView<double>,
                    ArrayView<DoubleGpuInstruction>,
                    int>>(
                typeof(double),
                ComputeKernelKind.Map);
            var destination =
                accelerator.Allocate1D<double>(length);
            try
            {
                using MemoryBuffer1D<
                    DoubleGpuInstruction,
                    Stride1D.Dense> instructionBuffer =
                    accelerator.Allocate1D(instructions);
                kernel(
                    length,
                    ((MemoryBuffer1D<double, Stride1D.Dense>)(object)source)
                    .View,
                    destination.View,
                    instructionBuffer.View,
                    instructions.Length);
                accelerator.Synchronize();
                return (MemoryBuffer1D<T, Stride1D.Dense>)(object)
                    destination;
            }
            catch
            {
                destination.Dispose();
                throw;
            }
        }

        {
            var instructions = ((NumericExpressionProgram<int>)(object)
                    program)
                .Instructions
                .Select(
                    item => new IntGpuInstruction(
                        (int)item.OpCode,
                        item.Operand))
                .ToArray();
            var kernel = GetNumericKernel<
                Action<
                    Index1D,
                    ArrayView<int>,
                    ArrayView<int>,
                    ArrayView<IntGpuInstruction>,
                    int>>(
                typeof(int),
                ComputeKernelKind.Map);
            var destination = accelerator.Allocate1D<int>(length);
            try
            {
                using MemoryBuffer1D<
                    IntGpuInstruction,
                    Stride1D.Dense> instructionBuffer =
                    accelerator.Allocate1D(instructions);
                kernel(
                    length,
                    ((MemoryBuffer1D<int, Stride1D.Dense>)(object)source)
                    .View,
                    destination.View,
                    instructionBuffer.View,
                    instructions.Length);
                accelerator.Synchronize();
                return (MemoryBuffer1D<T, Stride1D.Dense>)(object)
                    destination;
            }
            catch
            {
                destination.Dispose();
                throw;
            }
        }
    }

    private MemoryBuffer1D<T, Stride1D.Dense>
        ExecuteResidentNumericZip<T>(
            MemoryBuffer1D<T, Stride1D.Dense> left,
            MemoryBuffer1D<T, Stride1D.Dense> right,
            int length,
            NumericExpressionProgram<T> program)
        where T : unmanaged, INumber<T>
    {
        if (typeof(T) == typeof(double))
        {
            var instructions = ((NumericExpressionProgram<double>)(object)
                    program)
                .Instructions
                .Select(
                    item => new DoubleGpuInstruction(
                        (int)item.OpCode,
                        item.Operand))
                .ToArray();
            var kernel = GetNumericKernel<
                Action<
                    Index1D,
                    ArrayView<double>,
                    ArrayView<double>,
                    ArrayView<double>,
                    ArrayView<DoubleGpuInstruction>,
                    int>>(
                typeof(double),
                ComputeKernelKind.Zip);
            var destination =
                accelerator.Allocate1D<double>(length);
            try
            {
                using MemoryBuffer1D<
                    DoubleGpuInstruction,
                    Stride1D.Dense> instructionBuffer =
                    accelerator.Allocate1D(instructions);
                kernel(
                    length,
                    ((MemoryBuffer1D<double, Stride1D.Dense>)(object)left)
                    .View,
                    ((MemoryBuffer1D<double, Stride1D.Dense>)(object)right)
                    .View,
                    destination.View,
                    instructionBuffer.View,
                    instructions.Length);
                accelerator.Synchronize();
                return (MemoryBuffer1D<T, Stride1D.Dense>)(object)
                    destination;
            }
            catch
            {
                destination.Dispose();
                throw;
            }
        }

        {
            var instructions = ((NumericExpressionProgram<int>)(object)
                    program)
                .Instructions
                .Select(
                    item => new IntGpuInstruction(
                        (int)item.OpCode,
                        item.Operand))
                .ToArray();
            var kernel = GetNumericKernel<
                Action<
                    Index1D,
                    ArrayView<int>,
                    ArrayView<int>,
                    ArrayView<int>,
                    ArrayView<IntGpuInstruction>,
                    int>>(
                typeof(int),
                ComputeKernelKind.Zip);
            var destination = accelerator.Allocate1D<int>(length);
            try
            {
                using MemoryBuffer1D<
                    IntGpuInstruction,
                    Stride1D.Dense> instructionBuffer =
                    accelerator.Allocate1D(instructions);
                kernel(
                    length,
                    ((MemoryBuffer1D<int, Stride1D.Dense>)(object)left)
                    .View,
                    ((MemoryBuffer1D<int, Stride1D.Dense>)(object)right)
                    .View,
                    destination.View,
                    instructionBuffer.View,
                    instructions.Length);
                accelerator.Synchronize();
                return (MemoryBuffer1D<T, Stride1D.Dense>)(object)
                    destination;
            }
            catch
            {
                destination.Dispose();
                throw;
            }
        }
    }

    internal T[] ExecuteNumericMap<T>(
        T[] source,
        NumericExpressionProgram<T> program,
        ComputeOptions options)
        where T : unmanaged, INumber<T>
    {
        ThrowIfDisposed();
        options.CancellationToken.ThrowIfCancellationRequested();
        ValidateNumericGpuType<T>();
        ValidateNumericProgram(program);
        if (source.Length == 0)
        {
            return [];
        }

        return typeof(T) == typeof(double)
            ? (T[])(object)ExecuteDoubleMap(
                (double[])(object)source,
                (NumericExpressionProgram<double>)(object)program,
                options)
            : (T[])(object)ExecuteIntMap(
                (int[])(object)source,
                (NumericExpressionProgram<int>)(object)program,
                options);
    }

    internal T[] ExecuteNumericZip<T>(
        T[] left,
        T[] right,
        NumericExpressionProgram<T> program,
        ComputeOptions options)
        where T : unmanaged, INumber<T>
    {
        ThrowIfDisposed();
        options.CancellationToken.ThrowIfCancellationRequested();
        ValidateNumericGpuType<T>();
        ValidateNumericProgram(program);
        if (left.Length == 0)
        {
            return [];
        }

        return typeof(T) == typeof(double)
            ? (T[])(object)ExecuteDoubleZip(
                (double[])(object)left,
                (double[])(object)right,
                (NumericExpressionProgram<double>)(object)program,
                options)
            : (T[])(object)ExecuteIntZip(
                (int[])(object)left,
                (int[])(object)right,
                (NumericExpressionProgram<int>)(object)program,
                options);
    }

    internal T ExecuteNumericReduction<T>(
        T[] source,
        ComputeReductionKind reduction,
        ComputeOptions options)
        where T : unmanaged, INumber<T>
    {
        ThrowIfDisposed();
        options.CancellationToken.ThrowIfCancellationRequested();
        ValidateNumericGpuType<T>();
        return typeof(T) == typeof(double)
            ? (T)(object)ExecuteDoubleReduction(
                (double[])(object)source,
                reduction,
                options)
            : (T)(object)ExecuteIntReduction(
                (int[])(object)source,
                reduction,
                options);
    }

    internal T ExecuteNumericMappedReduction<T>(
        T[] source,
        NumericExpressionProgram<T> program,
        ComputeReductionKind reduction,
        ComputeOptions options)
        where T : unmanaged, INumber<T>
    {
        ThrowIfDisposed();
        options.CancellationToken.ThrowIfCancellationRequested();
        ValidateNumericGpuType<T>();
        ValidateNumericProgram(program);
        return typeof(T) == typeof(double)
            ? (T)(object)ExecuteDoubleMappedReduction(
                (double[])(object)source,
                (NumericExpressionProgram<double>)(object)program,
                reduction,
                options)
            : (T)(object)ExecuteIntMappedReduction(
                (int[])(object)source,
                (NumericExpressionProgram<int>)(object)program,
                reduction,
                options);
    }

    private double[] ExecuteDoubleMap(
        double[] source,
        NumericExpressionProgram<double> program,
        ComputeOptions options)
    {
        var instructions = program.Instructions
            .Select(
                item => new DoubleGpuInstruction(
                    (int)item.OpCode,
                    item.Operand))
            .ToArray();
        var kernel = GetNumericKernel<
            Action<
                Index1D,
                ArrayView<double>,
                ArrayView<double>,
                ArrayView<DoubleGpuInstruction>,
                int>>(
            typeof(double),
            ComputeKernelKind.Map);
        var destination = new double[source.Length];
        int chunkSize = GetNumericChunkSize(
            source.Length,
            sizeof(double),
            fullLengthBufferCount: 2,
            instructions.LongLength *
            (sizeof(int) + sizeof(double)),
            options);

        using MemoryBuffer1D<DoubleGpuInstruction, Stride1D.Dense>
            instructionBuffer = accelerator.Allocate1D(instructions);
        for (int offset = 0; offset < source.Length; offset += chunkSize)
        {
            options.CancellationToken.ThrowIfCancellationRequested();
            int count = Math.Min(chunkSize, source.Length - offset);
            using MemoryBuffer1D<double, Stride1D.Dense> sourceBuffer =
                accelerator.Allocate1D<double>(count);
            using MemoryBuffer1D<double, Stride1D.Dense> destinationBuffer =
                accelerator.Allocate1D<double>(count);
            sourceBuffer.View.CopyFromCPU(
                accelerator.DefaultStream,
                source.AsSpan(offset, count));
            kernel(
                count,
                sourceBuffer.View,
                destinationBuffer.View,
                instructionBuffer.View,
                instructions.Length);
            destinationBuffer.View.CopyToCPU(
                accelerator.DefaultStream,
                destination.AsSpan(offset, count));
            accelerator.Synchronize();
        }

        return destination;
    }

    private int[] ExecuteIntMap(
        int[] source,
        NumericExpressionProgram<int> program,
        ComputeOptions options)
    {
        var instructions = program.Instructions
            .Select(
                item => new IntGpuInstruction(
                    (int)item.OpCode,
                    item.Operand))
            .ToArray();
        var kernel = GetNumericKernel<
            Action<
                Index1D,
                ArrayView<int>,
                ArrayView<int>,
                ArrayView<IntGpuInstruction>,
                int>>(
            typeof(int),
            ComputeKernelKind.Map);
        var destination = new int[source.Length];
        int chunkSize = GetNumericChunkSize(
            source.Length,
            sizeof(int),
            fullLengthBufferCount: 2,
            instructions.LongLength * (sizeof(int) * 2L),
            options);

        using MemoryBuffer1D<IntGpuInstruction, Stride1D.Dense>
            instructionBuffer = accelerator.Allocate1D(instructions);
        for (int offset = 0; offset < source.Length; offset += chunkSize)
        {
            options.CancellationToken.ThrowIfCancellationRequested();
            int count = Math.Min(chunkSize, source.Length - offset);
            using MemoryBuffer1D<int, Stride1D.Dense> sourceBuffer =
                accelerator.Allocate1D<int>(count);
            using MemoryBuffer1D<int, Stride1D.Dense> destinationBuffer =
                accelerator.Allocate1D<int>(count);
            sourceBuffer.View.CopyFromCPU(
                accelerator.DefaultStream,
                source.AsSpan(offset, count));
            kernel(
                count,
                sourceBuffer.View,
                destinationBuffer.View,
                instructionBuffer.View,
                instructions.Length);
            destinationBuffer.View.CopyToCPU(
                accelerator.DefaultStream,
                destination.AsSpan(offset, count));
            accelerator.Synchronize();
        }

        return destination;
    }

    private double[] ExecuteDoubleZip(
        double[] left,
        double[] right,
        NumericExpressionProgram<double> program,
        ComputeOptions options)
    {
        var instructions = program.Instructions
            .Select(
                item => new DoubleGpuInstruction(
                    (int)item.OpCode,
                    item.Operand))
            .ToArray();
        var kernel = GetNumericKernel<
            Action<
                Index1D,
                ArrayView<double>,
                ArrayView<double>,
                ArrayView<double>,
                ArrayView<DoubleGpuInstruction>,
                int>>(
            typeof(double),
            ComputeKernelKind.Zip);
        var destination = new double[left.Length];
        int chunkSize = GetNumericChunkSize(
            left.Length,
            sizeof(double),
            fullLengthBufferCount: 3,
            instructions.LongLength *
            (sizeof(int) + sizeof(double)),
            options);

        using MemoryBuffer1D<DoubleGpuInstruction, Stride1D.Dense>
            instructionBuffer = accelerator.Allocate1D(instructions);
        for (int offset = 0; offset < left.Length; offset += chunkSize)
        {
            options.CancellationToken.ThrowIfCancellationRequested();
            int count = Math.Min(chunkSize, left.Length - offset);
            using MemoryBuffer1D<double, Stride1D.Dense> leftBuffer =
                accelerator.Allocate1D<double>(count);
            using MemoryBuffer1D<double, Stride1D.Dense> rightBuffer =
                accelerator.Allocate1D<double>(count);
            using MemoryBuffer1D<double, Stride1D.Dense> destinationBuffer =
                accelerator.Allocate1D<double>(count);
            leftBuffer.View.CopyFromCPU(
                accelerator.DefaultStream,
                left.AsSpan(offset, count));
            rightBuffer.View.CopyFromCPU(
                accelerator.DefaultStream,
                right.AsSpan(offset, count));
            kernel(
                count,
                leftBuffer.View,
                rightBuffer.View,
                destinationBuffer.View,
                instructionBuffer.View,
                instructions.Length);
            destinationBuffer.View.CopyToCPU(
                accelerator.DefaultStream,
                destination.AsSpan(offset, count));
            accelerator.Synchronize();
        }

        return destination;
    }

    private int[] ExecuteIntZip(
        int[] left,
        int[] right,
        NumericExpressionProgram<int> program,
        ComputeOptions options)
    {
        var instructions = program.Instructions
            .Select(
                item => new IntGpuInstruction(
                    (int)item.OpCode,
                    item.Operand))
            .ToArray();
        var kernel = GetNumericKernel<
            Action<
                Index1D,
                ArrayView<int>,
                ArrayView<int>,
                ArrayView<int>,
                ArrayView<IntGpuInstruction>,
                int>>(
            typeof(int),
            ComputeKernelKind.Zip);
        var destination = new int[left.Length];
        int chunkSize = GetNumericChunkSize(
            left.Length,
            sizeof(int),
            fullLengthBufferCount: 3,
            instructions.LongLength * (sizeof(int) * 2L),
            options);

        using MemoryBuffer1D<IntGpuInstruction, Stride1D.Dense>
            instructionBuffer = accelerator.Allocate1D(instructions);
        for (int offset = 0; offset < left.Length; offset += chunkSize)
        {
            options.CancellationToken.ThrowIfCancellationRequested();
            int count = Math.Min(chunkSize, left.Length - offset);
            using MemoryBuffer1D<int, Stride1D.Dense> leftBuffer =
                accelerator.Allocate1D<int>(count);
            using MemoryBuffer1D<int, Stride1D.Dense> rightBuffer =
                accelerator.Allocate1D<int>(count);
            using MemoryBuffer1D<int, Stride1D.Dense> destinationBuffer =
                accelerator.Allocate1D<int>(count);
            leftBuffer.View.CopyFromCPU(
                accelerator.DefaultStream,
                left.AsSpan(offset, count));
            rightBuffer.View.CopyFromCPU(
                accelerator.DefaultStream,
                right.AsSpan(offset, count));
            kernel(
                count,
                leftBuffer.View,
                rightBuffer.View,
                destinationBuffer.View,
                instructionBuffer.View,
                instructions.Length);
            destinationBuffer.View.CopyToCPU(
                accelerator.DefaultStream,
                destination.AsSpan(offset, count));
            accelerator.Synchronize();
        }

        return destination;
    }

    private double ExecuteDoubleReduction(
        double[] source,
        ComputeReductionKind reduction,
        ComputeOptions options)
    {
        ComputeReductionKind effective = reduction ==
            ComputeReductionKind.Average
            ? ComputeReductionKind.Sum
            : reduction;
        var kernel = GetNumericKernel<
            Action<
                Index1D,
                ArrayView<double>,
                ArrayView<double>,
                int,
                int>>(
            typeof(double),
            ComputeKernelKind.Reduction);
        int chunkSize = GetNumericChunkSize(
            source.Length,
            sizeof(double),
            fullLengthBufferCount: 2,
            fixedBytes: 0,
            options);
        double combined = 0d;
        bool hasValue = false;
        for (int offset = 0; offset < source.Length; offset += chunkSize)
        {
            options.CancellationToken.ThrowIfCancellationRequested();
            int count = Math.Min(chunkSize, source.Length - offset);
            using MemoryBuffer1D<double, Stride1D.Dense> sourceBuffer =
                accelerator.Allocate1D<double>(count);
            sourceBuffer.View.CopyFromCPU(
                accelerator.DefaultStream,
                source.AsSpan(offset, count));
            double partial = ReduceDoubleBuffer(
                sourceBuffer,
                count,
                effective,
                kernel);
            combined = CombineDouble(
                combined,
                partial,
                effective,
                ref hasValue);
        }

        return reduction == ComputeReductionKind.Average
            ? combined / source.Length
            : combined;
    }

    private double ExecuteDoubleMappedReduction(
        double[] source,
        NumericExpressionProgram<double> program,
        ComputeReductionKind reduction,
        ComputeOptions options)
    {
        ComputeReductionKind effective = reduction ==
            ComputeReductionKind.Average
            ? ComputeReductionKind.Sum
            : reduction;
        var instructions = program.Instructions
            .Select(item => new DoubleGpuInstruction(
                (int)item.OpCode,
                item.Operand))
            .ToArray();
        var mapReductionKernel = GetNumericKernel<
            Action<
                Index1D,
                ArrayView<double>,
                ArrayView<double>,
                int,
                ArrayView<DoubleGpuInstruction>,
                int,
                int>>(
            typeof(double),
            ComputeKernelKind.MapReduction);
        var reductionKernel = GetNumericKernel<
            Action<
                Index1D,
                ArrayView<double>,
                ArrayView<double>,
                int,
                int>>(
            typeof(double),
            ComputeKernelKind.Reduction);
        int chunkSize = GetNumericChunkSize(
            source.Length,
            sizeof(double),
            fullLengthBufferCount: 2,
            instructions.LongLength * (sizeof(int) + sizeof(double)),
            options);
        double combined = 0d;
        bool hasValue = false;
        using MemoryBuffer1D<DoubleGpuInstruction, Stride1D.Dense>
            instructionBuffer = accelerator.Allocate1D(instructions);
        for (int offset = 0; offset < source.Length; offset += chunkSize)
        {
            options.CancellationToken.ThrowIfCancellationRequested();
            int count = Math.Min(chunkSize, source.Length - offset);
            int firstPassLength =
                (count + GpuKernels.ReductionElementsPerOutput - 1) /
                GpuKernels.ReductionElementsPerOutput;
            using MemoryBuffer1D<double, Stride1D.Dense> sourceBuffer =
                accelerator.Allocate1D<double>(count);
            using MemoryBuffer1D<double, Stride1D.Dense> firstPassBuffer =
                accelerator.Allocate1D<double>(firstPassLength);
            sourceBuffer.View.CopyFromCPU(
                accelerator.DefaultStream,
                source.AsSpan(offset, count));
            mapReductionKernel(
                firstPassLength,
                sourceBuffer.View,
                firstPassBuffer.View,
                count,
                instructionBuffer.View,
                instructions.Length,
                (int)effective);
            double partial = ReduceDoubleBuffer(
                firstPassBuffer,
                firstPassLength,
                effective,
                reductionKernel);
            combined = CombineDouble(
                combined,
                partial,
                effective,
                ref hasValue);
        }

        return reduction == ComputeReductionKind.Average
            ? combined / source.Length
            : combined;
    }

    private int ExecuteIntReduction(
        int[] source,
        ComputeReductionKind reduction,
        ComputeOptions options)
    {
        ComputeReductionKind effective = reduction ==
            ComputeReductionKind.Average
            ? ComputeReductionKind.Sum
            : reduction;
        var kernel = GetNumericKernel<
            Action<
                Index1D,
                ArrayView<int>,
                ArrayView<int>,
                int,
                int>>(
            typeof(int),
            ComputeKernelKind.Reduction);
        int chunkSize = GetNumericChunkSize(
            source.Length,
            sizeof(int),
            fullLengthBufferCount: 2,
            fixedBytes: 0,
            options);
        int combined = 0;
        bool hasValue = false;
        for (int offset = 0; offset < source.Length; offset += chunkSize)
        {
            options.CancellationToken.ThrowIfCancellationRequested();
            int count = Math.Min(chunkSize, source.Length - offset);
            using MemoryBuffer1D<int, Stride1D.Dense> sourceBuffer =
                accelerator.Allocate1D<int>(count);
            sourceBuffer.View.CopyFromCPU(
                accelerator.DefaultStream,
                source.AsSpan(offset, count));
            int partial = ReduceIntBuffer(
                sourceBuffer,
                count,
                effective,
                kernel);
            combined = CombineInt(
                combined,
                partial,
                effective,
                ref hasValue);
        }

        return reduction == ComputeReductionKind.Average
            ? combined / source.Length
            : combined;
    }

    private int ExecuteIntMappedReduction(
        int[] source,
        NumericExpressionProgram<int> program,
        ComputeReductionKind reduction,
        ComputeOptions options)
    {
        ComputeReductionKind effective = reduction ==
            ComputeReductionKind.Average
            ? ComputeReductionKind.Sum
            : reduction;
        var instructions = program.Instructions
            .Select(item => new IntGpuInstruction(
                (int)item.OpCode,
                item.Operand))
            .ToArray();
        var mapReductionKernel = GetNumericKernel<
            Action<
                Index1D,
                ArrayView<int>,
                ArrayView<int>,
                int,
                ArrayView<IntGpuInstruction>,
                int,
                int>>(
            typeof(int),
            ComputeKernelKind.MapReduction);
        var reductionKernel = GetNumericKernel<
            Action<
                Index1D,
                ArrayView<int>,
                ArrayView<int>,
                int,
                int>>(
            typeof(int),
            ComputeKernelKind.Reduction);
        int chunkSize = GetNumericChunkSize(
            source.Length,
            sizeof(int),
            fullLengthBufferCount: 2,
            instructions.LongLength * (sizeof(int) * 2L),
            options);
        int combined = 0;
        bool hasValue = false;
        using MemoryBuffer1D<IntGpuInstruction, Stride1D.Dense>
            instructionBuffer = accelerator.Allocate1D(instructions);
        for (int offset = 0; offset < source.Length; offset += chunkSize)
        {
            options.CancellationToken.ThrowIfCancellationRequested();
            int count = Math.Min(chunkSize, source.Length - offset);
            int firstPassLength =
                (count + GpuKernels.ReductionElementsPerOutput - 1) /
                GpuKernels.ReductionElementsPerOutput;
            using MemoryBuffer1D<int, Stride1D.Dense> sourceBuffer =
                accelerator.Allocate1D<int>(count);
            using MemoryBuffer1D<int, Stride1D.Dense> firstPassBuffer =
                accelerator.Allocate1D<int>(firstPassLength);
            sourceBuffer.View.CopyFromCPU(
                accelerator.DefaultStream,
                source.AsSpan(offset, count));
            mapReductionKernel(
                firstPassLength,
                sourceBuffer.View,
                firstPassBuffer.View,
                count,
                instructionBuffer.View,
                instructions.Length,
                (int)effective);
            int partial = ReduceIntBuffer(
                firstPassBuffer,
                firstPassLength,
                effective,
                reductionKernel);
            combined = CombineInt(
                combined,
                partial,
                effective,
                ref hasValue);
        }

        return reduction == ComputeReductionKind.Average
            ? combined / source.Length
            : combined;
    }

    private double ReduceDoubleBuffer(
        MemoryBuffer1D<double, Stride1D.Dense> input,
        int length,
        ComputeReductionKind reduction,
        Action<
            Index1D,
            ArrayView<double>,
            ArrayView<double>,
            int,
            int> kernel)
    {
        MemoryBuffer1D<double, Stride1D.Dense> current = input;
        var intermediates =
            new List<MemoryBuffer1D<double, Stride1D.Dense>>();
        try
        {
            int currentLength = length;
            while (currentLength > 1)
            {
                int outputLength =
                    (currentLength +
                     GpuKernels.ReductionElementsPerOutput - 1) /
                    GpuKernels.ReductionElementsPerOutput;
                MemoryBuffer1D<double, Stride1D.Dense> output =
                    accelerator.Allocate1D<double>(outputLength);
                intermediates.Add(output);
                kernel(
                    outputLength,
                    current.View,
                    output.View,
                    currentLength,
                    (int)reduction);
                current = output;
                currentLength = outputLength;
            }

            accelerator.Synchronize();
            return current.GetAsArray1D()[0];
        }
        finally
        {
            foreach (MemoryBuffer1D<double, Stride1D.Dense> buffer
                     in intermediates)
            {
                buffer.Dispose();
            }
        }
    }

    private int ReduceIntBuffer(
        MemoryBuffer1D<int, Stride1D.Dense> input,
        int length,
        ComputeReductionKind reduction,
        Action<
            Index1D,
            ArrayView<int>,
            ArrayView<int>,
            int,
            int> kernel)
    {
        MemoryBuffer1D<int, Stride1D.Dense> current = input;
        var intermediates =
            new List<MemoryBuffer1D<int, Stride1D.Dense>>();
        try
        {
            int currentLength = length;
            while (currentLength > 1)
            {
                int outputLength =
                    (currentLength +
                     GpuKernels.ReductionElementsPerOutput - 1) /
                    GpuKernels.ReductionElementsPerOutput;
                MemoryBuffer1D<int, Stride1D.Dense> output =
                    accelerator.Allocate1D<int>(outputLength);
                intermediates.Add(output);
                kernel(
                    outputLength,
                    current.View,
                    output.View,
                    currentLength,
                    (int)reduction);
                current = output;
                currentLength = outputLength;
            }

            accelerator.Synchronize();
            return current.GetAsArray1D()[0];
        }
        finally
        {
            foreach (MemoryBuffer1D<int, Stride1D.Dense> buffer
                     in intermediates)
            {
                buffer.Dispose();
            }
        }
    }

    private TDelegate GetNumericKernel<TDelegate>(
        Type type,
        ComputeKernelKind kind)
        where TDelegate : Delegate
    {
        Lazy<object> lazy = numericKernels.GetOrAdd(
            (type, kind),
            key => new Lazy<object>(
                () => CompileNumericKernel(key.Type, key.Kind),
                LazyThreadSafetyMode.ExecutionAndPublication));
        return (TDelegate)lazy.Value;
    }

    private object CompileNumericKernel(Type type, ComputeKernelKind kind)
    {
        if (type == typeof(double))
        {
            return kind switch
            {
                ComputeKernelKind.Map =>
                    accelerator.LoadAutoGroupedStreamKernel<
                        Index1D,
                        ArrayView<double>,
                        ArrayView<double>,
                        ArrayView<DoubleGpuInstruction>,
                        int>(GpuKernels.MapDouble),
                ComputeKernelKind.Zip =>
                    accelerator.LoadAutoGroupedStreamKernel<
                        Index1D,
                        ArrayView<double>,
                        ArrayView<double>,
                        ArrayView<double>,
                        ArrayView<DoubleGpuInstruction>,
                        int>(GpuKernels.ZipDouble),
                ComputeKernelKind.Reduction =>
                    accelerator.LoadAutoGroupedStreamKernel<
                        Index1D,
                        ArrayView<double>,
                        ArrayView<double>,
                        int,
                        int>(GpuKernels.ReduceDouble),
                ComputeKernelKind.MapReduction =>
                    accelerator.LoadAutoGroupedStreamKernel<
                        Index1D,
                        ArrayView<double>,
                        ArrayView<double>,
                        int,
                        ArrayView<DoubleGpuInstruction>,
                        int,
                        int>(GpuKernels.MapReduceDouble),
                _ => throw new ArgumentOutOfRangeException(nameof(kind))
            };
        }

        return kind switch
        {
            ComputeKernelKind.Map =>
                accelerator.LoadAutoGroupedStreamKernel<
                    Index1D,
                    ArrayView<int>,
                    ArrayView<int>,
                    ArrayView<IntGpuInstruction>,
                    int>(GpuKernels.MapInt),
            ComputeKernelKind.Zip =>
                accelerator.LoadAutoGroupedStreamKernel<
                    Index1D,
                    ArrayView<int>,
                    ArrayView<int>,
                    ArrayView<int>,
                    ArrayView<IntGpuInstruction>,
                    int>(GpuKernels.ZipInt),
            ComputeKernelKind.Reduction =>
                accelerator.LoadAutoGroupedStreamKernel<
                    Index1D,
                    ArrayView<int>,
                    ArrayView<int>,
                    int,
                    int>(GpuKernels.ReduceInt),
            ComputeKernelKind.MapReduction =>
                accelerator.LoadAutoGroupedStreamKernel<
                    Index1D,
                    ArrayView<int>,
                    ArrayView<int>,
                    int,
                    ArrayView<IntGpuInstruction>,
                    int,
                    int>(GpuKernels.MapReduceInt),
            _ => throw new ArgumentOutOfRangeException(nameof(kind))
        };
    }

    private int GetNumericChunkSize(
        int length,
        int elementSize,
        int fullLengthBufferCount,
        long fixedBytes,
        ComputeOptions options)
    {
        if (!options.EnableGpuChunking)
        {
            long required = checked(
                (long)length * elementSize * fullLengthBufferCount +
                fixedBytes);
            long budget = GetAutomaticMemoryBudget(
                options.GpuMemoryBudgetBytes);
            if (required > budget)
            {
                throw new ComputeGpuMemoryBudgetExceededException(
                    required,
                    budget);
            }

            return Math.Max(1, length);
        }

        long available = Math.Max(
            0,
            GetAutomaticMemoryBudget(options.GpuMemoryBudgetBytes) -
            fixedBytes);
        long byBudget = available /
            (elementSize * (long)fullLengthBufferCount);
        int chunkSize = (int)Math.Min(
            length,
            Math.Max(1, Math.Min(int.MaxValue, byBudget)));
        if (options.GpuChunkElementCount is int configured)
        {
            if (configured <= 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(options.GpuChunkElementCount));
            }

            chunkSize = Math.Min(chunkSize, configured);
        }

        return Math.Max(1, chunkSize);
    }

    private static double CombineDouble(
        double current,
        double value,
        ComputeReductionKind reduction,
        ref bool hasValue)
    {
        if (!hasValue)
        {
            hasValue = true;
            return value;
        }

        return reduction switch
        {
            ComputeReductionKind.Sum => current + value,
            ComputeReductionKind.Min => Math.Min(current, value),
            ComputeReductionKind.Max => Math.Max(current, value),
            _ => throw new ArgumentOutOfRangeException(nameof(reduction))
        };
    }

    private static int CombineInt(
        int current,
        int value,
        ComputeReductionKind reduction,
        ref bool hasValue)
    {
        if (!hasValue)
        {
            hasValue = true;
            return value;
        }

        return reduction switch
        {
            ComputeReductionKind.Sum => current + value,
            ComputeReductionKind.Min => Math.Min(current, value),
            ComputeReductionKind.Max => Math.Max(current, value),
            _ => throw new ArgumentOutOfRangeException(nameof(reduction))
        };
    }

    private static void ValidateNumericGpuType<T>()
        where T : unmanaged
    {
        if (typeof(T) != typeof(double) && typeof(T) != typeof(int))
        {
            throw new NotSupportedException(
                $"Typed GPU execution does not support '{typeof(T).Name}'.");
        }
    }

    private static void ValidateNumericProgram<T>(
        NumericExpressionProgram<T> program)
        where T : unmanaged
    {
        if (program.MaximumStackDepth >
            GpuProgramCompiler.MaximumStackDepth)
        {
            throw new GpuExpressionNotSupportedException(
                System.Linq.Expressions.ExpressionType.Lambda,
                "numeric expression",
                $"The expression needs stack depth {program.MaximumStackDepth}, " +
                $"but the GPU limit is {GpuProgramCompiler.MaximumStackDepth}.",
                ["Split the expression into smaller operations."]);
        }
    }
}
