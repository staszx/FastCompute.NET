using System.Diagnostics;
using FastCompute.Backends.Cpu;
using FastCompute.Expressions;

namespace FastCompute.Backends.ParallelCpu;

internal sealed class ParallelComputeBackend : IComputeBackend
{
    private const int MinimumChunkSize = 4_096;

    internal static ParallelComputeBackend Instance { get; } = new();

    private ParallelComputeBackend()
    {
    }

    public ComputeBackendKind Kind => ComputeBackendKind.ParallelCpu;

    public bool IsAvailable => true;

    public bool Supports(ComputeExpressionPlan plan) => true;

    public ComputeBackendExecution<float[]> ExecuteMap(
        float[] source,
        ComputeExpressionPlan plan,
        ComputeExecutionContext context)
    {
        long compilationStarted = StartTiming(context.CollectDiagnostics);
        Func<float, float> operation = CpuExpressionCompiler.CompileUnary(plan);
        TimeSpan compilationTime = StopTiming(compilationStarted, context.CollectDiagnostics);
        var destination = new float[source.Length];

        long executionStarted = StartTiming(context.CollectDiagnostics);
        ExecuteChunks(
            source.Length,
            context,
            (start, end) =>
            {
                for (int index = start; index < end; index++)
                {
                    CheckCancellation(index - start, context.CancellationToken);
                    destination[index] = operation(source[index]);
                }
            });

        return new ComputeBackendExecution<float[]>(
            destination,
            compilationTime,
            StopTiming(executionStarted, context.CollectDiagnostics));
    }

    public ComputeBackendExecution<float[]> ExecuteZip(
        float[] left,
        float[] right,
        ComputeExpressionPlan plan,
        ComputeExecutionContext context)
    {
        long compilationStarted = StartTiming(context.CollectDiagnostics);
        Func<float, float, float> operation = CpuExpressionCompiler.CompileBinary(plan);
        TimeSpan compilationTime = StopTiming(compilationStarted, context.CollectDiagnostics);
        var destination = new float[left.Length];

        long executionStarted = StartTiming(context.CollectDiagnostics);
        ExecuteChunks(
            left.Length,
            context,
            (start, end) =>
            {
                for (int index = start; index < end; index++)
                {
                    CheckCancellation(index - start, context.CancellationToken);
                    destination[index] = operation(left[index], right[index]);
                }
            });

        return new ComputeBackendExecution<float[]>(
            destination,
            compilationTime,
            StopTiming(executionStarted, context.CollectDiagnostics));
    }

    public ComputeBackendExecution<float> Reduce(
        float[] source,
        ComputeReductionKind reduction,
        ComputeExecutionContext context)
    {
        ComputeReductionKind effectiveReduction =
            reduction == ComputeReductionKind.Average
                ? ComputeReductionKind.Sum
                : reduction;
        int chunkSize = GetChunkSize(source.Length, context.MaxDegreeOfParallelism);
        int chunkCount = GetChunkCount(source.Length, chunkSize);
        var partialResults = new float[chunkCount];
        ParallelOptions options = CreateParallelOptions(context);

        long executionStarted = StartTiming(context.CollectDiagnostics);
        Parallel.For(
            0,
            chunkCount,
            options,
            chunkIndex =>
            {
                int start = chunkIndex * chunkSize;
                int end = (int)Math.Min((long)start + chunkSize, source.Length);
                float partial = effectiveReduction == ComputeReductionKind.Sum
                    ? 0f
                    : source[start];
                int firstIndex = effectiveReduction == ComputeReductionKind.Sum
                    ? start
                    : start + 1;

                for (int index = firstIndex; index < end; index++)
                {
                    CheckCancellation(index - start, context.CancellationToken);
                    partial = ApplyReduction(
                        effectiveReduction,
                        partial,
                        source[index]);
                }

                partialResults[chunkIndex] = partial;
            });

        float result = effectiveReduction == ComputeReductionKind.Sum
            ? 0f
            : partialResults[0];
        int firstPartial = effectiveReduction == ComputeReductionKind.Sum ? 0 : 1;
        for (int index = firstPartial; index < partialResults.Length; index++)
        {
            CheckCancellation(index, context.CancellationToken);
            result = ApplyReduction(
                effectiveReduction,
                result,
                partialResults[index]);
        }

        if (reduction == ComputeReductionKind.Average)
        {
            result /= source.Length;
        }

        return new ComputeBackendExecution<float>(
            result,
            TimeSpan.Zero,
            StopTiming(executionStarted, context.CollectDiagnostics));
    }

    private static float ApplyReduction(
        ComputeReductionKind reduction,
        float left,
        float right) =>
        reduction switch
        {
            ComputeReductionKind.Sum => left + right,
            ComputeReductionKind.Min => GpuMath.Min(left, right),
            ComputeReductionKind.Max => GpuMath.Max(left, right),
            _ => throw new ArgumentOutOfRangeException(nameof(reduction))
        };

    private static void ExecuteChunks(
        int length,
        ComputeExecutionContext context,
        Action<int, int> executeRange)
    {
        int chunkSize = GetChunkSize(length, context.MaxDegreeOfParallelism);
        int chunkCount = GetChunkCount(length, chunkSize);
        ParallelOptions options = CreateParallelOptions(context);

        Parallel.For(
            0,
            chunkCount,
            options,
            chunkIndex =>
            {
                int start = chunkIndex * chunkSize;
                int end = (int)Math.Min((long)start + chunkSize, length);
                executeRange(start, end);
            });
    }

    private static int GetChunkSize(int length, int? maxDegreeOfParallelism)
    {
        int processorCount = maxDegreeOfParallelism ?? Environment.ProcessorCount;
        int targetChunkCount = Math.Max(1, processorCount * 4);
        long calculated = ((long)length + targetChunkCount - 1) / targetChunkCount;
        return (int)Math.Min(int.MaxValue, Math.Max(MinimumChunkSize, calculated));
    }

    private static int GetChunkCount(int length, int chunkSize) =>
        (int)(((long)length + chunkSize - 1) / chunkSize);

    private static ParallelOptions CreateParallelOptions(ComputeExecutionContext context) =>
        new()
        {
            CancellationToken = context.CancellationToken,
            MaxDegreeOfParallelism = context.MaxDegreeOfParallelism ?? -1
        };

    private static void CheckCancellation(int localIndex, CancellationToken cancellationToken)
    {
        if ((localIndex & 0xFFF) == 0)
        {
            cancellationToken.ThrowIfCancellationRequested();
        }
    }

    private static long StartTiming(bool enabled) =>
        enabled ? Stopwatch.GetTimestamp() : 0L;

    private static TimeSpan StopTiming(long started, bool enabled) =>
        enabled ? Stopwatch.GetElapsedTime(started) : TimeSpan.Zero;
}
