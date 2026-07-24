using System.Diagnostics;
using FastCompute.Backends.Cpu;
using FastCompute.Expressions;

namespace FastCompute.Backends.Scalar;

internal sealed class ScalarComputeBackend : IComputeBackend
{
    internal static ScalarComputeBackend Instance { get; } = new();

    private ScalarComputeBackend()
    {
    }

    public ComputeBackendKind Kind => ComputeBackendKind.Scalar;

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
        for (int index = 0; index < source.Length; index++)
        {
            CheckCancellation(index, context.CancellationToken);
            destination[index] = operation(source[index]);
        }

        return new ComputeBackendExecution<float[]>(
            destination,
            compilationTime,
            StopTiming(executionStarted, context.CollectDiagnostics));
    }

    internal ComputeBackendExecution<float[]> ExecuteDelegateMap(
        float[] source,
        Func<float, float> operation,
        ComputeExecutionContext context)
    {
        var destination = new float[source.Length];

        long executionStarted = StartTiming(context.CollectDiagnostics);
        for (int index = 0; index < source.Length; index++)
        {
            CheckCancellation(index, context.CancellationToken);
            destination[index] = operation(source[index]);
        }

        return new ComputeBackendExecution<float[]>(
            destination,
            TimeSpan.Zero,
            StopTiming(executionStarted, context.CollectDiagnostics));
    }

    internal ComputeBackendExecution<float[]> ExecuteMapInPlace(
        float[] source,
        ComputeExpressionPlan plan,
        ComputeExecutionContext context)
    {
        long compilationStarted = StartTiming(context.CollectDiagnostics);
        Func<float, float> operation = CpuExpressionCompiler.CompileUnary(plan);
        TimeSpan compilationTime = StopTiming(compilationStarted, context.CollectDiagnostics);

        long executionStarted = StartTiming(context.CollectDiagnostics);
        for (int index = 0; index < source.Length; index++)
        {
            CheckCancellation(index, context.CancellationToken);
            source[index] = operation(source[index]);
        }

        return new ComputeBackendExecution<float[]>(
            source,
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
        for (int index = 0; index < left.Length; index++)
        {
            CheckCancellation(index, context.CancellationToken);
            destination[index] = operation(left[index], right[index]);
        }

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
        long executionStarted = StartTiming(context.CollectDiagnostics);
        float result = reduction switch
        {
            ComputeReductionKind.Sum => Sum(source, context.CancellationToken),
            ComputeReductionKind.Min => Minimum(source, context.CancellationToken),
            ComputeReductionKind.Max => Maximum(source, context.CancellationToken),
            ComputeReductionKind.Average =>
                Sum(source, context.CancellationToken) / source.Length,
            _ => throw new ArgumentOutOfRangeException(nameof(reduction))
        };

        return new ComputeBackendExecution<float>(
            result,
            TimeSpan.Zero,
            StopTiming(executionStarted, context.CollectDiagnostics));
    }

    private static float Sum(float[] source, CancellationToken cancellationToken)
    {
        float result = 0f;
        for (int index = 0; index < source.Length; index++)
        {
            CheckCancellation(index, cancellationToken);
            result += source[index];
        }

        return result;
    }

    private static float Minimum(
        float[] source,
        CancellationToken cancellationToken)
    {
        float result = source[0];
        for (int index = 1; index < source.Length; index++)
        {
            CheckCancellation(index, cancellationToken);
            result = GpuMath.Min(result, source[index]);
        }

        return result;
    }

    private static float Maximum(
        float[] source,
        CancellationToken cancellationToken)
    {
        float result = source[0];
        for (int index = 1; index < source.Length; index++)
        {
            CheckCancellation(index, cancellationToken);
            result = GpuMath.Max(result, source[index]);
        }

        return result;
    }

    private static void CheckCancellation(int index, CancellationToken cancellationToken)
    {
        if ((index & 0xFFF) == 0)
        {
            cancellationToken.ThrowIfCancellationRequested();
        }
    }

    private static long StartTiming(bool enabled) =>
        enabled ? Stopwatch.GetTimestamp() : 0L;

    private static TimeSpan StopTiming(long started, bool enabled) =>
        enabled ? Stopwatch.GetElapsedTime(started) : TimeSpan.Zero;
}
