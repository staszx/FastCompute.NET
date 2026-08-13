using System.ComponentModel;
using ILGPU;

namespace FastCompute.Gpu;

/// <summary>Contains generic byte-component compute kernels required by ILGPU.</summary>
[EditorBrowsable(EditorBrowsableState.Never)]
public static class ByteCompositeGpuKernels
{
    /// <summary>Executes a packed byte-component map.</summary>
    public static void Map(Index1D index, ArrayView<byte> source, ArrayView<byte> destination, ArrayView<ByteGpuInstruction> program, ArrayView<int> outputOffsets, ArrayView<int> outputInstructionCounts, int sourceComponents, int destinationComponents)
    {
        int inputOffset = index * sourceComponents;
        int outputOffset = index * destinationComponents;
        for (int component = 0; component < destinationComponents; component++)
            destination[outputOffset + component] = (byte)Evaluate(source, inputOffset, program, outputOffsets[component], outputInstructionCounts[component]);
    }

    /// <summary>Executes a packed byte-component projection to float.</summary>
    public static void Project(Index1D index, ArrayView<byte> source, ArrayView<float> destination, ArrayView<ByteGpuInstruction> program, int instructionCount, int sourceComponents) =>
        destination[index] = Evaluate(source, index * sourceComponents, program, 0, instructionCount);

    private static int Evaluate(ArrayView<byte> source, int sourceOffset, ArrayView<ByteGpuInstruction> program, int programOffset, int instructionCount)
    {
        ArrayView<int> stack = LocalMemory.Allocate<int>(GpuProgramCompiler.MaximumStackDepth);
        int stackPointer = 0;
        int end = programOffset + instructionCount;
        for (int index = programOffset; index < end; index++)
        {
            ByteGpuInstruction instruction = program[index];
            if (instruction.OpCode == ByteGpuOpCode.Component) stack[stackPointer++] = source[sourceOffset + instruction.Operand];
            else if (instruction.OpCode == ByteGpuOpCode.Constant) stack[stackPointer++] = instruction.Operand;
            else if (instruction.OpCode == ByteGpuOpCode.Negate) stack[stackPointer - 1] = -stack[stackPointer - 1];
            else if (instruction.OpCode == ByteGpuOpCode.Narrow) stack[stackPointer - 1] &= 255;
            else
            {
                int right = stack[--stackPointer];
                int left = stack[stackPointer - 1];
                stack[stackPointer - 1] = instruction.OpCode == ByteGpuOpCode.Add ? left + right
                    : instruction.OpCode == ByteGpuOpCode.Subtract ? left - right
                    : instruction.OpCode == ByteGpuOpCode.Multiply ? left * right : left / right;
            }
        }
        return stack[0];
    }
}
