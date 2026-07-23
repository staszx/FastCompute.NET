using ILGPU;
using ILGPU.Algorithms;

namespace FastCompute.Gpu;

internal static class GpuKernels
{
    internal const int ReductionElementsPerOutput = 256;

    internal static void Map(
        Index1D index,
        ArrayView<float> source,
        ArrayView<float> destination,
        ArrayView<GpuInstruction> program,
        int instructionCount)
    {
        destination[index] = Evaluate(
            source[index],
            0f,
            program,
            instructionCount);
    }

    internal static void Zip(
        Index1D index,
        ArrayView<float> left,
        ArrayView<float> right,
        ArrayView<float> destination,
        ArrayView<GpuInstruction> program,
        int instructionCount)
    {
        destination[index] = Evaluate(
            left[index],
            right[index],
            program,
            instructionCount);
    }

    internal static void Reduce(
        Index1D outputIndex,
        ArrayView<float> source,
        ArrayView<float> destination,
        int sourceLength,
        int reduction)
    {
        int start = outputIndex * ReductionElementsPerOutput;
        int end = XMath.Min(start + ReductionElementsPerOutput, sourceLength);
        float result = reduction == (int)ComputeReductionKind.Sum
            ? 0f
            : source[start];
        int firstIndex = reduction == (int)ComputeReductionKind.Sum
            ? start
            : start + 1;

        for (int index = firstIndex; index < end; index++)
        {
            float value = source[index];
            if (reduction == (int)ComputeReductionKind.Sum)
            {
                result += value;
            }
            else if (XMath.IsNaN(result) || XMath.IsNaN(value))
            {
                result += value;
            }
            else if (reduction == (int)ComputeReductionKind.Min)
            {
                result = XMath.Min(result, value);
            }
            else
            {
                result = XMath.Max(result, value);
            }
        }

        destination[outputIndex] = result;
    }

    private static float Evaluate(
        float parameter0,
        float parameter1,
        ArrayView<GpuInstruction> program,
        int instructionCount)
    {
        ArrayView<float> stack = LocalMemory.Allocate<float>(GpuProgramCompiler.MaximumStackDepth);
        int stackPointer = 0;

        for (int instructionIndex = 0;
             instructionIndex < instructionCount;
             instructionIndex++)
        {
            GpuInstruction instruction = program[instructionIndex];
            int operation = instruction.OpCode;

            if (operation == GpuOpCode.Parameter0)
            {
                stack[stackPointer++] = parameter0;
            }
            else if (operation == GpuOpCode.Parameter1)
            {
                stack[stackPointer++] = parameter1;
            }
            else if (operation == GpuOpCode.Constant)
            {
                stack[stackPointer++] = instruction.Operand;
            }
            else if (operation == GpuOpCode.Negate)
            {
                stack[stackPointer - 1] = -stack[stackPointer - 1];
            }
            else if (operation == GpuOpCode.Abs)
            {
                stack[stackPointer - 1] = XMath.Abs(stack[stackPointer - 1]);
            }
            else if (operation == GpuOpCode.Sqrt)
            {
                stack[stackPointer - 1] = XMath.Sqrt(stack[stackPointer - 1]);
            }
            else if (operation == GpuOpCode.Sin)
            {
                stack[stackPointer - 1] = XMath.Sin(stack[stackPointer - 1]);
            }
            else if (operation == GpuOpCode.Cos)
            {
                stack[stackPointer - 1] = XMath.Cos(stack[stackPointer - 1]);
            }
            else if (operation == GpuOpCode.Tan)
            {
                stack[stackPointer - 1] = XMath.Tan(stack[stackPointer - 1]);
            }
            else if (operation == GpuOpCode.Exp)
            {
                stack[stackPointer - 1] = XMath.Exp(stack[stackPointer - 1]);
            }
            else if (operation == GpuOpCode.Log)
            {
                stack[stackPointer - 1] = XMath.Log(stack[stackPointer - 1]);
            }
            else if (operation == GpuOpCode.Log10)
            {
                stack[stackPointer - 1] = XMath.Log10(stack[stackPointer - 1]);
            }
            else if (operation == GpuOpCode.Floor)
            {
                stack[stackPointer - 1] = XMath.Floor(stack[stackPointer - 1]);
            }
            else if (operation == GpuOpCode.Ceiling)
            {
                stack[stackPointer - 1] = XMath.Ceiling(stack[stackPointer - 1]);
            }
            else if (operation == GpuOpCode.Round)
            {
                stack[stackPointer - 1] = XMath.Round(stack[stackPointer - 1]);
            }
            else if (operation == GpuOpCode.Clamp)
            {
                float maximum = stack[--stackPointer];
                float minimum = stack[--stackPointer];
                stack[stackPointer - 1] =
                    XMath.Clamp(stack[stackPointer - 1], minimum, maximum);
            }
            else
            {
                float right = stack[--stackPointer];
                float left = stack[stackPointer - 1];
                stack[stackPointer - 1] = ApplyBinary(operation, left, right);
            }
        }

        return stack[0];
    }

    private static float ApplyBinary(int operation, float left, float right)
    {
        if (operation == GpuOpCode.Add)
        {
            return left + right;
        }

        if (operation == GpuOpCode.Subtract)
        {
            return left - right;
        }

        if (operation == GpuOpCode.Multiply)
        {
            return left * right;
        }

        if (operation == GpuOpCode.Divide)
        {
            return left / right;
        }

        if (operation == GpuOpCode.Min)
        {
            return XMath.Min(left, right);
        }

        if (operation == GpuOpCode.Max)
        {
            return XMath.Max(left, right);
        }

        return XMath.Pow(left, right);
    }
}
