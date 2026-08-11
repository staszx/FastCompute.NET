using FastCompute.Expressions;

namespace FastCompute.Gpu;

internal sealed record CompositeGpuProgram(
    GpuInstruction[] Instructions,
    int[] OutputOffsets,
    int[] OutputInstructionCounts);

internal static class CompositeGpuProgramCompiler
{
    internal static CompositeGpuProgram Compile(
        ComputeValueExpressionProgram expression)
    {
        var instructions = new List<GpuInstruction>();
        var offsets = new int[expression.Outputs.Count];
        var counts = new int[expression.Outputs.Count];

        for (int output = 0; output < expression.Outputs.Count; output++)
        {
            offsets[output] = instructions.Count;
            int stackDepth = 0;
            int maximumStackDepth = 0;
            Emit(
                expression.Outputs[output],
                instructions,
                ref stackDepth,
                ref maximumStackDepth);
            counts[output] = instructions.Count - offsets[output];

            if (counts[output] > GpuProgramCompiler.MaximumInstructionCount)
            {
                throw new ComputeException(
                    $"GPU component expressions may contain at most " +
                    $"{GpuProgramCompiler.MaximumInstructionCount} instructions.");
            }

            if (maximumStackDepth > GpuProgramCompiler.MaximumStackDepth)
            {
                throw new ComputeException(
                    $"GPU component expressions may require at most " +
                    $"{GpuProgramCompiler.MaximumStackDepth} stack values.");
            }
        }

        return new CompositeGpuProgram(
            instructions.ToArray(),
            offsets,
            counts);
    }

    private static void Emit(
        ComputeValueExpressionNode node,
        List<GpuInstruction> instructions,
        ref int stackDepth,
        ref int maximumStackDepth)
    {
        switch (node)
        {
            case ComputeValueComponentNode component:
                Push(
                    new GpuInstruction(GpuOpCode.Component, component.Index),
                    instructions,
                    ref stackDepth,
                    ref maximumStackDepth);
                return;
            case ComputeValueConstantNode constant:
                Push(
                    new GpuInstruction(GpuOpCode.Constant, constant.Value),
                    instructions,
                    ref stackDepth,
                    ref maximumStackDepth);
                return;
            case ComputeValueNegateNode negate:
                Emit(negate.Operand, instructions, ref stackDepth, ref maximumStackDepth);
                instructions.Add(new GpuInstruction(GpuOpCode.Negate, 0f));
                return;
            case ComputeValueBinaryNode binary:
                Emit(binary.Left, instructions, ref stackDepth, ref maximumStackDepth);
                Emit(binary.Right, instructions, ref stackDepth, ref maximumStackDepth);
                instructions.Add(new GpuInstruction(ToGpuOpCode(binary.Operation), 0f));
                stackDepth--;
                return;
            default:
                throw new NotSupportedException(
                    $"Unknown composite expression node '{node.GetType().Name}'.");
        }
    }

    private static void Push(
        GpuInstruction instruction,
        List<GpuInstruction> instructions,
        ref int stackDepth,
        ref int maximumStackDepth)
    {
        instructions.Add(instruction);
        maximumStackDepth = Math.Max(maximumStackDepth, ++stackDepth);
    }

    private static int ToGpuOpCode(ComputeValueBinaryOperation operation) =>
        operation switch
        {
            ComputeValueBinaryOperation.Add => GpuOpCode.Add,
            ComputeValueBinaryOperation.Subtract => GpuOpCode.Subtract,
            ComputeValueBinaryOperation.Multiply => GpuOpCode.Multiply,
            ComputeValueBinaryOperation.Divide => GpuOpCode.Divide,
            _ => throw new ArgumentOutOfRangeException(nameof(operation))
        };
}
