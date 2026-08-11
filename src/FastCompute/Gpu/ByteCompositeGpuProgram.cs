using FastCompute.Expressions;

namespace FastCompute.Gpu;

/// <summary>Represents one internal byte-component GPU instruction.</summary>
[System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
public readonly record struct ByteGpuInstruction(int OpCode, int Operand);

internal static class ByteGpuOpCode
{
    internal const int Component = 0;
    internal const int Constant = 1;
    internal const int Negate = 2;
    internal const int Narrow = 3;
    internal const int Add = 4;
    internal const int Subtract = 5;
    internal const int Multiply = 6;
    internal const int Divide = 7;
}

internal sealed record ByteCompositeGpuProgram(
    ByteGpuInstruction[] Instructions,
    int[] OutputOffsets,
    int[] OutputInstructionCounts);

internal static class ByteCompositeGpuProgramCompiler
{
    internal static ByteCompositeGpuProgram Compile(ByteComputeProgram program)
    {
        var instructions = new List<ByteGpuInstruction>();
        var offsets = new int[program.Outputs.Count];
        var counts = new int[program.Outputs.Count];
        for (int output = 0; output < program.Outputs.Count; output++)
        {
            offsets[output] = instructions.Count;
            Emit(program.Outputs[output], instructions);
            counts[output] = instructions.Count - offsets[output];
            if (counts[output] > GpuProgramCompiler.MaximumInstructionCount)
                throw new ComputeException($"GPU byte-component expressions may contain at most {GpuProgramCompiler.MaximumInstructionCount} instructions.");
        }
        return new ByteCompositeGpuProgram(instructions.ToArray(), offsets, counts);
    }

    private static void Emit(ByteComputeNode node, List<ByteGpuInstruction> instructions)
    {
        switch (node)
        {
            case ByteComponentNode component:
                instructions.Add(new ByteGpuInstruction(ByteGpuOpCode.Component, component.Index));
                break;
            case ByteConstantNode constant:
                instructions.Add(new ByteGpuInstruction(ByteGpuOpCode.Constant, constant.Value));
                break;
            case ByteNegateNode negate:
                Emit(negate.Operand, instructions);
                instructions.Add(new ByteGpuInstruction(ByteGpuOpCode.Negate, 0));
                break;
            case ByteNarrowNode narrow:
                Emit(narrow.Operand, instructions);
                instructions.Add(new ByteGpuInstruction(ByteGpuOpCode.Narrow, 0));
                break;
            case ByteBinaryNode binary:
                Emit(binary.Left, instructions);
                Emit(binary.Right, instructions);
                instructions.Add(new ByteGpuInstruction(binary.Operation switch
                {
                    ByteComputeOperation.Add => ByteGpuOpCode.Add,
                    ByteComputeOperation.Subtract => ByteGpuOpCode.Subtract,
                    ByteComputeOperation.Multiply => ByteGpuOpCode.Multiply,
                    ByteComputeOperation.Divide => ByteGpuOpCode.Divide,
                    _ => throw new ArgumentOutOfRangeException()
                }, 0));
                break;
            default:
                throw new NotSupportedException($"Unknown byte compute node '{node.GetType().Name}'.");
        }
    }
}
