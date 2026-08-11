using System.Numerics;
using FastCompute.Backends.Cpu;
using FastCompute.Expressions;

namespace FastCompute.Backends.Simd;

internal static class NumericSimdExecutor
{
    private const int CancellationCheckMask = 0xFFF;

    internal static bool IsAvailable<T>()
        where T : unmanaged, INumber<T> =>
        Vector.IsHardwareAccelerated &&
        Vector<T>.IsSupported;

    internal static bool Supports<T>(NumericExpressionProgram<T> program)
        where T : unmanaged, INumber<T> =>
        IsAvailable<T>() &&
        program.Instructions.All(
            instruction => instruction.OpCode is
                NumericOpCode.Parameter0 or
                NumericOpCode.Parameter1 or
                NumericOpCode.Constant or
                NumericOpCode.Negate or
                NumericOpCode.Add or
                NumericOpCode.Subtract or
                NumericOpCode.Multiply or
                NumericOpCode.Divide or
                NumericOpCode.Abs or
                NumericOpCode.Min or
                NumericOpCode.Max or
                NumericOpCode.Clamp);

    internal static T[] Map<T>(
        T[] source,
        NumericExpressionProgram<T> program,
        CancellationToken cancellationToken)
        where T : unmanaged, INumber<T>
    {
        EnsureSupported(program);
        var destination = new T[source.Length];
        int vectorizedLength =
            source.Length - source.Length % Vector<T>.Count;
        var stack = new Vector<T>[program.MaximumStackDepth];

        for (int offset = 0;
             offset < vectorizedLength;
             offset += Vector<T>.Count)
        {
            CheckCancellation(offset, cancellationToken);
            var input = new Vector<T>(source, offset);
            Vector<T> result =
                Evaluate(input, Vector<T>.Zero, program, stack);
            result.CopyTo(destination, offset);
        }

        for (int index = vectorizedLength; index < source.Length; index++)
        {
            CheckCancellation(index, cancellationToken);
            destination[index] = NumericCpuExecutor.Evaluate(
                source[index],
                T.Zero,
                program);
        }

        return destination;
    }

    internal static T[] Zip<T>(
        T[] left,
        T[] right,
        NumericExpressionProgram<T> program,
        CancellationToken cancellationToken)
        where T : unmanaged, INumber<T>
    {
        EnsureSupported(program);
        var destination = new T[left.Length];
        int vectorizedLength =
            left.Length - left.Length % Vector<T>.Count;
        var stack = new Vector<T>[program.MaximumStackDepth];

        for (int offset = 0;
             offset < vectorizedLength;
             offset += Vector<T>.Count)
        {
            CheckCancellation(offset, cancellationToken);
            Vector<T> result = Evaluate(
                new Vector<T>(left, offset),
                new Vector<T>(right, offset),
                program,
                stack);
            result.CopyTo(destination, offset);
        }

        for (int index = vectorizedLength; index < left.Length; index++)
        {
            CheckCancellation(index, cancellationToken);
            destination[index] = NumericCpuExecutor.Evaluate(
                left[index],
                right[index],
                program);
        }

        return destination;
    }

    internal static T Reduce<T>(
        T[] source,
        ComputeReductionKind reduction,
        CancellationToken cancellationToken)
        where T : unmanaged, INumber<T>
    {
        if (!IsAvailable<T>())
        {
            throw new ComputeBackendUnavailableException(
                ComputeBackendKind.Simd);
        }

        if (source.Length < Vector<T>.Count)
        {
            return NumericCpuExecutor.ReduceScalar(
                source,
                reduction,
                cancellationToken);
        }

        int vectorizedLength =
            source.Length - source.Length % Vector<T>.Count;
        Vector<T> accumulator =
            reduction is ComputeReductionKind.Sum or
                ComputeReductionKind.Average
                ? Vector<T>.Zero
                : new Vector<T>(source, 0);
        int offset = reduction is ComputeReductionKind.Sum or
            ComputeReductionKind.Average
            ? 0
            : Vector<T>.Count;
        for (; offset < vectorizedLength; offset += Vector<T>.Count)
        {
            CheckCancellation(offset, cancellationToken);
            var value = new Vector<T>(source, offset);
            accumulator = reduction switch
            {
                ComputeReductionKind.Sum or ComputeReductionKind.Average =>
                    Vector.Add(accumulator, value),
                ComputeReductionKind.Min => Vector.Min(accumulator, value),
                ComputeReductionKind.Max => Vector.Max(accumulator, value),
                _ => throw new ArgumentOutOfRangeException(nameof(reduction))
            };
        }

        T result = reduction is ComputeReductionKind.Sum or
            ComputeReductionKind.Average
            ? T.Zero
            : accumulator[0];
        int firstLane = reduction is ComputeReductionKind.Sum or
            ComputeReductionKind.Average
            ? 0
            : 1;
        for (int lane = firstLane; lane < Vector<T>.Count; lane++)
        {
            result = ApplyReduction(reduction, result, accumulator[lane]);
        }

        for (int index = vectorizedLength; index < source.Length; index++)
        {
            result = ApplyReduction(reduction, result, source[index]);
        }

        return reduction == ComputeReductionKind.Average
            ? result / T.CreateChecked(source.Length)
            : result;
    }

    internal static T ReduceMapped<T>(
        T[] source,
        NumericExpressionProgram<T> program,
        ComputeReductionKind reduction,
        CancellationToken cancellationToken)
        where T : unmanaged, INumber<T>
    {
        EnsureSupported(program);
        if (source.Length < Vector<T>.Count)
        {
            return NumericCpuExecutor.ReduceMappedScalar(
                source,
                program,
                reduction,
                cancellationToken);
        }

        int vectorizedLength =
            source.Length - source.Length % Vector<T>.Count;
        bool isSum = reduction is ComputeReductionKind.Sum or
            ComputeReductionKind.Average;
        var stack = new Vector<T>[program.MaximumStackDepth];
        Vector<T> accumulator = isSum
            ? Vector<T>.Zero
            : Evaluate(
                new Vector<T>(source, 0),
                Vector<T>.Zero,
                program,
                stack);
        int offset = isSum ? 0 : Vector<T>.Count;
        for (; offset < vectorizedLength; offset += Vector<T>.Count)
        {
            CheckCancellation(offset, cancellationToken);
            Vector<T> value = Evaluate(
                new Vector<T>(source, offset),
                Vector<T>.Zero,
                program,
                stack);
            accumulator = reduction switch
            {
                ComputeReductionKind.Sum or ComputeReductionKind.Average =>
                    Vector.Add(accumulator, value),
                ComputeReductionKind.Min => Vector.Min(accumulator, value),
                ComputeReductionKind.Max => Vector.Max(accumulator, value),
                _ => throw new ArgumentOutOfRangeException(nameof(reduction))
            };
        }

        T result = isSum ? T.Zero : accumulator[0];
        int firstLane = isSum ? 0 : 1;
        for (int lane = firstLane; lane < Vector<T>.Count; lane++)
        {
            result = ApplyReduction(reduction, result, accumulator[lane]);
        }

        for (int index = vectorizedLength; index < source.Length; index++)
        {
            CheckCancellation(index, cancellationToken);
            result = ApplyReduction(
                reduction,
                result,
                NumericCpuExecutor.Evaluate(
                    source[index],
                    T.Zero,
                    program));
        }

        return reduction == ComputeReductionKind.Average
            ? result / T.CreateChecked(source.Length)
            : result;
    }

    internal static T ReduceZipped<T>(
        T[] left,
        T[] right,
        NumericExpressionProgram<T> program,
        ComputeReductionKind reduction,
        CancellationToken cancellationToken)
        where T : unmanaged, INumber<T>
    {
        EnsureSupported(program);
        if (left.Length < Vector<T>.Count)
        {
            return NumericCpuExecutor.ReduceZippedScalar(
                left,
                right,
                program,
                reduction,
                cancellationToken);
        }

        int vectorizedLength =
            left.Length - left.Length % Vector<T>.Count;
        bool isSum = reduction is ComputeReductionKind.Sum or
            ComputeReductionKind.Average;
        var stack = new Vector<T>[program.MaximumStackDepth];
        Vector<T> accumulator = isSum
            ? Vector<T>.Zero
            : Evaluate(
                new Vector<T>(left, 0),
                new Vector<T>(right, 0),
                program,
                stack);
        int offset = isSum ? 0 : Vector<T>.Count;
        for (; offset < vectorizedLength; offset += Vector<T>.Count)
        {
            CheckCancellation(offset, cancellationToken);
            Vector<T> value = Evaluate(
                new Vector<T>(left, offset),
                new Vector<T>(right, offset),
                program,
                stack);
            accumulator = reduction switch
            {
                ComputeReductionKind.Sum or ComputeReductionKind.Average =>
                    Vector.Add(accumulator, value),
                ComputeReductionKind.Min => Vector.Min(accumulator, value),
                ComputeReductionKind.Max => Vector.Max(accumulator, value),
                _ => throw new ArgumentOutOfRangeException(nameof(reduction))
            };
        }

        T result = isSum ? T.Zero : accumulator[0];
        int firstLane = isSum ? 0 : 1;
        for (int lane = firstLane; lane < Vector<T>.Count; lane++)
        {
            result = ApplyReduction(reduction, result, accumulator[lane]);
        }

        for (int index = vectorizedLength; index < left.Length; index++)
        {
            CheckCancellation(index, cancellationToken);
            result = ApplyReduction(
                reduction,
                result,
                NumericCpuExecutor.Evaluate(
                    left[index],
                    right[index],
                    program));
        }

        return reduction == ComputeReductionKind.Average
            ? result / T.CreateChecked(left.Length)
            : result;
    }

    private static Vector<T> Evaluate<T>(
        Vector<T> parameter0,
        Vector<T> parameter1,
        NumericExpressionProgram<T> program,
        Vector<T>[] stack)
        where T : unmanaged, INumber<T>
    {
        int stackPointer = 0;
        foreach (NumericInstruction<T> instruction in program.Instructions)
        {
            switch (instruction.OpCode)
            {
                case NumericOpCode.Parameter0:
                    stack[stackPointer++] = parameter0;
                    break;
                case NumericOpCode.Parameter1:
                    stack[stackPointer++] = parameter1;
                    break;
                case NumericOpCode.Constant:
                    stack[stackPointer++] =
                        new Vector<T>(instruction.Operand);
                    break;
                case NumericOpCode.Negate:
                    stack[stackPointer - 1] =
                        Vector.Negate(stack[stackPointer - 1]);
                    break;
                case NumericOpCode.Abs:
                    stack[stackPointer - 1] =
                        Vector.Abs(stack[stackPointer - 1]);
                    break;
                case NumericOpCode.Clamp:
                {
                    Vector<T> maximum = stack[--stackPointer];
                    Vector<T> minimum = stack[--stackPointer];
                    stack[stackPointer - 1] = Vector.Min(
                        Vector.Max(stack[stackPointer - 1], minimum),
                        maximum);
                    break;
                }
                default:
                {
                    Vector<T> right = stack[--stackPointer];
                    Vector<T> left = stack[stackPointer - 1];
                    stack[stackPointer - 1] =
                        ApplyBinary(instruction.OpCode, left, right);
                    break;
                }
            }
        }

        return stack[0];
    }

    private static Vector<T> ApplyBinary<T>(
        NumericOpCode operation,
        Vector<T> left,
        Vector<T> right)
        where T : unmanaged, INumber<T> =>
        operation switch
        {
            NumericOpCode.Add => Vector.Add(left, right),
            NumericOpCode.Subtract => Vector.Subtract(left, right),
            NumericOpCode.Multiply => Vector.Multiply(left, right),
            NumericOpCode.Divide => Vector.Divide(left, right),
            NumericOpCode.Min => Vector.Min(left, right),
            NumericOpCode.Max => Vector.Max(left, right),
            _ => throw new InvalidOperationException(
                $"Unexpected SIMD operation '{operation}'.")
        };

    private static T ApplyReduction<T>(
        ComputeReductionKind reduction,
        T left,
        T right)
        where T : unmanaged, INumber<T> =>
        reduction switch
        {
            ComputeReductionKind.Sum or ComputeReductionKind.Average =>
                left + right,
            ComputeReductionKind.Min => T.Min(left, right),
            ComputeReductionKind.Max => T.Max(left, right),
            _ => throw new ArgumentOutOfRangeException(nameof(reduction))
        };

    private static void EnsureSupported<T>(
        NumericExpressionProgram<T> program)
        where T : unmanaged, INumber<T>
    {
        if (!IsAvailable<T>())
        {
            throw new ComputeBackendUnavailableException(
                ComputeBackendKind.Simd);
        }

        if (!Supports(program))
        {
            throw new ComputeBackendNotSupportedException(
                ComputeBackendKind.Simd,
                $"the requested {typeof(T).Name} expression",
                "Scalar, ParallelCpu, or Gpu");
        }
    }

    private static void CheckCancellation(
        int index,
        CancellationToken cancellationToken)
    {
        if ((index & CancellationCheckMask) == 0)
        {
            cancellationToken.ThrowIfCancellationRequested();
        }
    }
}
