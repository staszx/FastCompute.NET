using System.Numerics;
using FastCompute.Expressions;

namespace FastCompute.Backends.Cpu;

internal static class NumericCpuExecutor
{
    private const int CancellationCheckMask = 0xFFF;

    internal static T[] MapScalar<T>(
        T[] source,
        NumericExpressionProgram<T> program,
        CancellationToken cancellationToken)
        where T : unmanaged, INumber<T>
    {
        var destination = new T[source.Length];
        for (int index = 0; index < source.Length; index++)
        {
            CheckCancellation(index, cancellationToken);
            destination[index] = Evaluate(
                source[index],
                T.Zero,
                program);
        }

        return destination;
    }

    internal static T[] MapParallel<T>(
        T[] source,
        NumericExpressionProgram<T> program,
        ComputeOptions options)
        where T : unmanaged, INumber<T>
    {
        var destination = new T[source.Length];
        Parallel.For(
            0,
            source.Length,
            new ParallelOptions
            {
                CancellationToken = options.CancellationToken,
                MaxDegreeOfParallelism =
                    options.MaxDegreeOfParallelism ?? -1
            },
            index =>
                destination[index] = Evaluate(
                    source[index],
                    T.Zero,
                    program));
        return destination;
    }

    internal static T[] ZipScalar<T>(
        T[] left,
        T[] right,
        NumericExpressionProgram<T> program,
        CancellationToken cancellationToken)
        where T : unmanaged, INumber<T>
    {
        var destination = new T[left.Length];
        for (int index = 0; index < left.Length; index++)
        {
            CheckCancellation(index, cancellationToken);
            destination[index] = Evaluate(
                left[index],
                right[index],
                program);
        }

        return destination;
    }

    internal static T[] ZipParallel<T>(
        T[] left,
        T[] right,
        NumericExpressionProgram<T> program,
        ComputeOptions options)
        where T : unmanaged, INumber<T>
    {
        var destination = new T[left.Length];
        Parallel.For(
            0,
            left.Length,
            new ParallelOptions
            {
                CancellationToken = options.CancellationToken,
                MaxDegreeOfParallelism =
                    options.MaxDegreeOfParallelism ?? -1
            },
            index =>
                destination[index] = Evaluate(
                    left[index],
                    right[index],
                    program));
        return destination;
    }

    internal static T ReduceScalar<T>(
        T[] source,
        ComputeReductionKind reduction,
        CancellationToken cancellationToken)
        where T : unmanaged, INumber<T>
    {
        T result = reduction is ComputeReductionKind.Sum or
            ComputeReductionKind.Average
            ? T.Zero
            : source[0];
        int firstIndex = reduction is ComputeReductionKind.Sum or
            ComputeReductionKind.Average
            ? 0
            : 1;

        for (int index = firstIndex; index < source.Length; index++)
        {
            CheckCancellation(index, cancellationToken);
            result = ApplyReduction(reduction, result, source[index]);
        }

        return reduction == ComputeReductionKind.Average
            ? result / T.CreateChecked(source.Length)
            : result;
    }

    internal static T ReduceParallel<T>(
        T[] source,
        ComputeReductionKind reduction,
        ComputeOptions options)
        where T : unmanaged, INumber<T>
    {
        int processorCount = options.MaxDegreeOfParallelism is > 0
            ? options.MaxDegreeOfParallelism.Value
            : Environment.ProcessorCount;
        int chunkCount = Math.Min(
            source.Length,
            Math.Max(1, processorCount));
        int chunkSize = (source.Length + chunkCount - 1) / chunkCount;
        var partials = new T[chunkCount];

        Parallel.For(
            0,
            chunkCount,
            new ParallelOptions
            {
                CancellationToken = options.CancellationToken,
                MaxDegreeOfParallelism =
                    options.MaxDegreeOfParallelism ?? -1
            },
            chunk =>
            {
                int start = chunk * chunkSize;
                int end = Math.Min(start + chunkSize, source.Length);
                T partial = reduction is ComputeReductionKind.Sum or
                    ComputeReductionKind.Average
                    ? T.Zero
                    : source[start];
                int first = reduction is ComputeReductionKind.Sum or
                    ComputeReductionKind.Average
                    ? start
                    : start + 1;
                for (int index = first; index < end; index++)
                {
                    partial = ApplyReduction(
                        reduction,
                        partial,
                        source[index]);
                }

                partials[chunk] = partial;
            });

        T result = reduction is ComputeReductionKind.Sum or
            ComputeReductionKind.Average
            ? T.Zero
            : partials[0];
        int firstPartial = reduction is ComputeReductionKind.Sum or
            ComputeReductionKind.Average
            ? 0
            : 1;
        for (int index = firstPartial; index < partials.Length; index++)
        {
            result = ApplyReduction(reduction, result, partials[index]);
        }

        return reduction == ComputeReductionKind.Average
            ? result / T.CreateChecked(source.Length)
            : result;
    }

    internal static T Evaluate<T>(
        T parameter0,
        T parameter1,
        NumericExpressionProgram<T> program)
        where T : unmanaged, INumber<T>
    {
        Span<T> stack = stackalloc T[program.MaximumStackDepth];
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
                    stack[stackPointer++] = instruction.Operand;
                    break;
                case NumericOpCode.Negate:
                    stack[stackPointer - 1] = -stack[stackPointer - 1];
                    break;
                case NumericOpCode.Abs:
                    stack[stackPointer - 1] = T.Abs(stack[stackPointer - 1]);
                    break;
                case NumericOpCode.Sqrt:
                case NumericOpCode.Sin:
                case NumericOpCode.Cos:
                case NumericOpCode.Tan:
                case NumericOpCode.Exp:
                case NumericOpCode.Log:
                case NumericOpCode.Log10:
                case NumericOpCode.Floor:
                case NumericOpCode.Ceiling:
                case NumericOpCode.Round:
                    stack[stackPointer - 1] = ApplyDoubleUnary(
                        instruction.OpCode,
                        stack[stackPointer - 1]);
                    break;
                case NumericOpCode.Clamp:
                {
                    T maximum = stack[--stackPointer];
                    T minimum = stack[--stackPointer];
                    stack[stackPointer - 1] =
                        T.Clamp(
                            stack[stackPointer - 1],
                            minimum,
                            maximum);
                    break;
                }
                default:
                {
                    T right = stack[--stackPointer];
                    T left = stack[stackPointer - 1];
                    stack[stackPointer - 1] =
                        ApplyBinary(instruction.OpCode, left, right);
                    break;
                }
            }
        }

        return stack[0];
    }

    private static T ApplyBinary<T>(
        NumericOpCode operation,
        T left,
        T right)
        where T : unmanaged, INumber<T> =>
        operation switch
        {
            NumericOpCode.Add => left + right,
            NumericOpCode.Subtract => left - right,
            NumericOpCode.Multiply => left * right,
            NumericOpCode.Divide => left / right,
            NumericOpCode.Min => T.Min(left, right),
            NumericOpCode.Max => T.Max(left, right),
            NumericOpCode.Pow =>
                ApplyDoubleBinary(operation, left, right),
            _ => throw new InvalidOperationException(
                $"Unexpected numeric operation '{operation}'.")
        };

    private static T ApplyDoubleUnary<T>(
        NumericOpCode operation,
        T value)
        where T : unmanaged, INumber<T>
    {
        double operand = double.CreateChecked(value);
        double result = operation switch
        {
            NumericOpCode.Sqrt => Math.Sqrt(operand),
            NumericOpCode.Sin => Math.Sin(operand),
            NumericOpCode.Cos => Math.Cos(operand),
            NumericOpCode.Tan => Math.Tan(operand),
            NumericOpCode.Exp => Math.Exp(operand),
            NumericOpCode.Log => Math.Log(operand),
            NumericOpCode.Log10 => Math.Log10(operand),
            NumericOpCode.Floor => Math.Floor(operand),
            NumericOpCode.Ceiling => Math.Ceiling(operand),
            NumericOpCode.Round => Math.Round(operand),
            _ => throw new InvalidOperationException(
                $"Unexpected double operation '{operation}'.")
        };
        return T.CreateChecked(result);
    }

    private static T ApplyDoubleBinary<T>(
        NumericOpCode operation,
        T left,
        T right)
        where T : unmanaged, INumber<T>
    {
        if (operation != NumericOpCode.Pow)
        {
            throw new InvalidOperationException(
                $"Unexpected double operation '{operation}'.");
        }

        return T.CreateChecked(
            Math.Pow(
                double.CreateChecked(left),
                double.CreateChecked(right)));
    }

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
