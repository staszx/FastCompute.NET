using System.Globalization;
using System.Text;
using FastCompute.Expressions;

namespace FastCompute.Gpu;

internal static class GpuOpCode
{
    internal const int Parameter0 = 0;
    internal const int Parameter1 = 1;
    internal const int Constant = 2;
    internal const int Negate = 3;
    internal const int Add = 4;
    internal const int Subtract = 5;
    internal const int Multiply = 6;
    internal const int Divide = 7;
    internal const int Abs = 8;
    internal const int Min = 9;
    internal const int Max = 10;
    internal const int Clamp = 11;
    internal const int Sqrt = 12;
    internal const int Sin = 13;
    internal const int Cos = 14;
    internal const int Tan = 15;
    internal const int Exp = 16;
    internal const int Log = 17;
    internal const int Log10 = 18;
    internal const int Pow = 19;
    internal const int Floor = 20;
    internal const int Ceiling = 21;
    internal const int Round = 22;
}

internal readonly record struct GpuInstruction(int OpCode, float Operand);

internal sealed record GpuProgram(
    GpuInstruction[] Instructions,
    string StructuralKey,
    int ParameterCount);

internal static class GpuProgramCompiler
{
    internal const int MaximumInstructionCount = 64;
    internal const int MaximumStackDepth = 32;

    internal static GpuProgram Compile(ComputeExpressionPlan plan)
    {
        var instructions = new List<GpuInstruction>();
        int stackDepth = 0;
        int maximumStackDepth = 0;

        Emit(plan.Root, instructions, ref stackDepth, ref maximumStackDepth);

        if (instructions.Count > MaximumInstructionCount)
        {
            throw new ComputeException(
                $"GPU expressions may contain at most {MaximumInstructionCount} instructions.");
        }

        if (maximumStackDepth > MaximumStackDepth)
        {
            throw new ComputeException(
                $"GPU expressions may require at most {MaximumStackDepth} stack values.");
        }

        var key = new StringBuilder(plan.ParameterCount.ToString(CultureInfo.InvariantCulture));
        foreach (GpuInstruction instruction in instructions)
        {
            key.Append('|')
                .Append(instruction.OpCode.ToString(CultureInfo.InvariantCulture))
                .Append(':')
                .Append(BitConverter.SingleToInt32Bits(instruction.Operand).ToString("X8", CultureInfo.InvariantCulture));
        }

        return new GpuProgram(instructions.ToArray(), key.ToString(), plan.ParameterCount);
    }

    private static void Emit(
        ComputeNode node,
        List<GpuInstruction> instructions,
        ref int stackDepth,
        ref int maximumStackDepth)
    {
        switch (node)
        {
            case ParameterNode parameter:
                Push(
                    new GpuInstruction(
                        parameter.Index == 0 ? GpuOpCode.Parameter0 : GpuOpCode.Parameter1,
                        0f),
                    instructions,
                    ref stackDepth,
                    ref maximumStackDepth);
                return;

            case ConstantNode constant:
                Push(
                    new GpuInstruction(GpuOpCode.Constant, constant.Value),
                    instructions,
                    ref stackDepth,
                    ref maximumStackDepth);
                return;

            case UnaryNode unary:
                Emit(unary.Operand, instructions, ref stackDepth, ref maximumStackDepth);
                instructions.Add(new GpuInstruction(GpuOpCode.Negate, 0f));
                return;

            case BinaryNode binary:
                Emit(binary.Left, instructions, ref stackDepth, ref maximumStackDepth);
                Emit(binary.Right, instructions, ref stackDepth, ref maximumStackDepth);
                stackDepth--;
                instructions.Add(new GpuInstruction(Map(binary.Operation), 0f));
                return;

            case FunctionNode function:
                foreach (ComputeNode argument in function.Arguments)
                {
                    Emit(argument, instructions, ref stackDepth, ref maximumStackDepth);
                }

                stackDepth -= function.Arguments.Count - 1;
                instructions.Add(new GpuInstruction(Map(function.Function), 0f));
                return;

            default:
                throw new ComputeException($"Unsupported GPU IR node '{node.GetType().Name}'.");
        }
    }

    private static void Push(
        GpuInstruction instruction,
        List<GpuInstruction> instructions,
        ref int stackDepth,
        ref int maximumStackDepth)
    {
        instructions.Add(instruction);
        stackDepth++;
        maximumStackDepth = Math.Max(maximumStackDepth, stackDepth);
    }

    private static int Map(BinaryOperation operation) => operation switch
    {
        BinaryOperation.Add => GpuOpCode.Add,
        BinaryOperation.Subtract => GpuOpCode.Subtract,
        BinaryOperation.Multiply => GpuOpCode.Multiply,
        BinaryOperation.Divide => GpuOpCode.Divide,
        _ => throw new ArgumentOutOfRangeException(nameof(operation))
    };

    private static int Map(ComputeFunction function) => function switch
    {
        ComputeFunction.Abs => GpuOpCode.Abs,
        ComputeFunction.Min => GpuOpCode.Min,
        ComputeFunction.Max => GpuOpCode.Max,
        ComputeFunction.Clamp => GpuOpCode.Clamp,
        ComputeFunction.Sqrt => GpuOpCode.Sqrt,
        ComputeFunction.Sin => GpuOpCode.Sin,
        ComputeFunction.Cos => GpuOpCode.Cos,
        ComputeFunction.Tan => GpuOpCode.Tan,
        ComputeFunction.Exp => GpuOpCode.Exp,
        ComputeFunction.Log => GpuOpCode.Log,
        ComputeFunction.Log10 => GpuOpCode.Log10,
        ComputeFunction.Pow => GpuOpCode.Pow,
        ComputeFunction.Floor => GpuOpCode.Floor,
        ComputeFunction.Ceiling => GpuOpCode.Ceiling,
        ComputeFunction.Round => GpuOpCode.Round,
        _ => throw new ArgumentOutOfRangeException(nameof(function))
    };
}
