using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using FastCompute.Backends.Cpu;
using FastCompute.Expressions;

namespace FastCompute.Backends.Simd;

internal sealed class SimdComputeBackend : IComputeBackend
{
    private const int CancellationCheckMask = 0xFFF;
    internal static SimdComputeBackend Instance { get; } = new();

    private SimdComputeBackend()
    {
    }

    public ComputeBackendKind Kind => ComputeBackendKind.Simd;

    public bool IsAvailable => Avx.IsSupported;

    public bool Supports(ComputeExpressionPlan plan) =>
        IsAvailable && SimdExpressionCompiler.Supports(plan);

    public ComputeBackendExecution<float[]> ExecuteMap(
        float[] source,
        ComputeExpressionPlan plan,
        ComputeExecutionContext context)
    {
        long compilationStarted = StartTiming(context.CollectDiagnostics);
        Func<Vector256<float>, Vector256<float>> vectorOperation =
            SimdExpressionCompiler.CompileUnary(plan);
        Func<float, float> scalarOperation =
            CpuExpressionCompiler.CompileUnary(plan);
        TimeSpan compilationTime =
            StopTiming(compilationStarted, context.CollectDiagnostics);
        var destination = new float[source.Length];
        int vectorizedLength =
            source.Length - source.Length % Vector256<float>.Count;
        ref float sourceReference =
            ref MemoryMarshal.GetArrayDataReference(source);
        ref float destinationReference =
            ref MemoryMarshal.GetArrayDataReference(destination);

        long executionStarted = StartTiming(context.CollectDiagnostics);
        for (int offset = 0; offset < vectorizedLength; offset += Vector256<float>.Count)
        {
            CheckCancellation(offset, context.CancellationToken);
            Vector256<float> sourceVector =
                Vector256.LoadUnsafe(ref sourceReference, (nuint)offset);
            Vector256<float> result = vectorOperation(sourceVector);
            result.StoreUnsafe(ref destinationReference, (nuint)offset);
        }

        for (int index = vectorizedLength; index < source.Length; index++)
        {
            CheckCancellation(index, context.CancellationToken);
            destination[index] = scalarOperation(source[index]);
        }

        return new ComputeBackendExecution<float[]>(
            destination,
            compilationTime,
            StopTiming(executionStarted, context.CollectDiagnostics));
    }

    internal ComputeBackendExecution<float[]> ExecuteMapInPlace(
        float[] source,
        ComputeExpressionPlan plan,
        ComputeExecutionContext context)
    {
        long compilationStarted = StartTiming(context.CollectDiagnostics);
        Func<Vector256<float>, Vector256<float>> vectorOperation =
            SimdExpressionCompiler.CompileUnary(plan);
        Func<float, float> scalarOperation =
            CpuExpressionCompiler.CompileUnary(plan);
        TimeSpan compilationTime =
            StopTiming(compilationStarted, context.CollectDiagnostics);
        int vectorizedLength =
            source.Length - source.Length % Vector256<float>.Count;
        ref float sourceReference =
            ref MemoryMarshal.GetArrayDataReference(source);

        long executionStarted = StartTiming(context.CollectDiagnostics);
        for (int offset = 0; offset < vectorizedLength; offset += Vector256<float>.Count)
        {
            CheckCancellation(offset, context.CancellationToken);
            Vector256<float> sourceVector =
                Vector256.LoadUnsafe(ref sourceReference, (nuint)offset);
            Vector256<float> result = vectorOperation(sourceVector);
            result.StoreUnsafe(ref sourceReference, (nuint)offset);
        }

        for (int index = vectorizedLength; index < source.Length; index++)
        {
            CheckCancellation(index, context.CancellationToken);
            source[index] = scalarOperation(source[index]);
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
        Func<
            Vector256<float>,
            Vector256<float>,
            Vector256<float>> vectorOperation =
            SimdExpressionCompiler.CompileBinary(plan);
        Func<float, float, float> scalarOperation =
            CpuExpressionCompiler.CompileBinary(plan);
        TimeSpan compilationTime =
            StopTiming(compilationStarted, context.CollectDiagnostics);
        var destination = new float[left.Length];
        int vectorizedLength =
            left.Length - left.Length % Vector256<float>.Count;
        ref float leftReference = ref MemoryMarshal.GetArrayDataReference(left);
        ref float rightReference = ref MemoryMarshal.GetArrayDataReference(right);
        ref float destinationReference =
            ref MemoryMarshal.GetArrayDataReference(destination);

        long executionStarted = StartTiming(context.CollectDiagnostics);
        for (int offset = 0; offset < vectorizedLength; offset += Vector256<float>.Count)
        {
            CheckCancellation(offset, context.CancellationToken);
            Vector256<float> leftVector =
                Vector256.LoadUnsafe(ref leftReference, (nuint)offset);
            Vector256<float> rightVector =
                Vector256.LoadUnsafe(ref rightReference, (nuint)offset);
            Vector256<float> result =
                vectorOperation(leftVector, rightVector);
            result.StoreUnsafe(ref destinationReference, (nuint)offset);
        }

        for (int index = vectorizedLength; index < left.Length; index++)
        {
            CheckCancellation(index, context.CancellationToken);
            destination[index] = scalarOperation(left[index], right[index]);
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
        ComputeReductionKind effectiveReduction =
            reduction == ComputeReductionKind.Average
                ? ComputeReductionKind.Sum
                : reduction;
        int vectorizedLength =
            source.Length - source.Length % Vector256<float>.Count;
        ref float sourceReference =
            ref MemoryMarshal.GetArrayDataReference(source);

        long executionStarted = StartTiming(context.CollectDiagnostics);
        float result = effectiveReduction switch
        {
            ComputeReductionKind.Sum =>
                Sum(source, vectorizedLength, ref sourceReference, context.CancellationToken),
            ComputeReductionKind.Min =>
                Minimum(source, vectorizedLength, ref sourceReference, context.CancellationToken),
            ComputeReductionKind.Max =>
                Maximum(source, vectorizedLength, ref sourceReference, context.CancellationToken),
            _ => throw new ArgumentOutOfRangeException(nameof(reduction))
        };

        if (reduction == ComputeReductionKind.Average)
        {
            result /= source.Length;
        }

        return new ComputeBackendExecution<float>(
            result,
            TimeSpan.Zero,
            StopTiming(executionStarted, context.CollectDiagnostics));
    }

    private static float Sum(
        float[] source,
        int vectorizedLength,
        ref float sourceReference,
        CancellationToken cancellationToken)
    {
        Vector256<float> accumulator = Vector256<float>.Zero;
        for (int offset = 0; offset < vectorizedLength; offset += Vector256<float>.Count)
        {
            CheckCancellation(offset, cancellationToken);
            accumulator = Avx.Add(
                accumulator,
                Vector256.LoadUnsafe(ref sourceReference, (nuint)offset));
        }

        float result = 0f;
        for (int lane = 0; lane < Vector256<float>.Count; lane++)
        {
            result += accumulator.GetElement(lane);
        }

        for (int index = vectorizedLength; index < source.Length; index++)
        {
            CheckCancellation(index, cancellationToken);
            result += source[index];
        }

        return result;
    }

    private static float Minimum(
        float[] source,
        int vectorizedLength,
        ref float sourceReference,
        CancellationToken cancellationToken) =>
        Extremum(
            source,
            vectorizedLength,
            ref sourceReference,
            cancellationToken,
            ComputeReductionKind.Min);

    private static float Maximum(
        float[] source,
        int vectorizedLength,
        ref float sourceReference,
        CancellationToken cancellationToken) =>
        Extremum(
            source,
            vectorizedLength,
            ref sourceReference,
            cancellationToken,
            ComputeReductionKind.Max);

    private static float Extremum(
        float[] source,
        int vectorizedLength,
        ref float sourceReference,
        CancellationToken cancellationToken,
        ComputeReductionKind reduction)
    {
        if (vectorizedLength == 0)
        {
            float scalarResult = source[0];
            for (int index = 1; index < source.Length; index++)
            {
                scalarResult = ApplyScalarExtremum(
                    reduction,
                    scalarResult,
                    source[index]);
            }

            return scalarResult;
        }

        Vector256<float> accumulator =
            Vector256.LoadUnsafe(ref sourceReference);
        for (int offset = Vector256<float>.Count;
             offset < vectorizedLength;
             offset += Vector256<float>.Count)
        {
            CheckCancellation(offset, cancellationToken);
            Vector256<float> value =
                Vector256.LoadUnsafe(ref sourceReference, (nuint)offset);
            accumulator = reduction == ComputeReductionKind.Min
                ? SimdVectorOperations.Minimum(accumulator, value)
                : SimdVectorOperations.Maximum(accumulator, value);
        }

        float result = accumulator.GetElement(0);
        for (int lane = 1; lane < Vector256<float>.Count; lane++)
        {
            result = ApplyScalarExtremum(
                reduction,
                result,
                accumulator.GetElement(lane));
        }

        for (int index = vectorizedLength; index < source.Length; index++)
        {
            CheckCancellation(index, cancellationToken);
            result = ApplyScalarExtremum(reduction, result, source[index]);
        }

        return result;
    }

    private static float ApplyScalarExtremum(
        ComputeReductionKind reduction,
        float left,
        float right) =>
        reduction == ComputeReductionKind.Min
            ? GpuMath.Min(left, right)
            : GpuMath.Max(left, right);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void CheckCancellation(
        int index,
        CancellationToken cancellationToken)
    {
        if ((index & CancellationCheckMask) == 0)
        {
            cancellationToken.ThrowIfCancellationRequested();
        }
    }

    private static long StartTiming(bool enabled) =>
        enabled ? Stopwatch.GetTimestamp() : 0L;

    private static TimeSpan StopTiming(long started, bool enabled) =>
        enabled ? Stopwatch.GetElapsedTime(started) : TimeSpan.Zero;
}
