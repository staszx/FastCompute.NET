using System.ComponentModel;
using FastCompute.Expressions;
using ILGPU;
using ILGPU.Algorithms;

namespace FastCompute.Gpu;

/// <summary>
/// Contains public entry points required by the dynamically generated ILGPU
/// runtime assembly. These methods are infrastructure, not user-facing APIs.
/// </summary>
[EditorBrowsable(EditorBrowsableState.Never)]
public static class GpuKernels
{
    internal const int ReductionElementsPerOutput = 256;

    /// <summary>Executes the internal unary Map kernel.</summary>
    public static void Map(
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

    /// <summary>Executes the internal binary Zip kernel.</summary>
    public static void Zip(
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

    /// <summary>Executes one stage of the internal reduction kernel.</summary>
    public static void Reduce(
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

    /// <summary>Executes the internal Histogram accumulation kernel.</summary>
    public static void Histogram(
        Index1D index,
        ArrayView<float> source,
        ArrayView<int> histogram,
        int binCount,
        float minimum,
        float maximum,
        float scale,
        int outOfRangeMode)
    {
        float value = source[index];
        if (XMath.IsNaN(value))
        {
            return;
        }

        int binIndex;
        if (value < minimum)
        {
            if (outOfRangeMode == (int)HistogramOutOfRangeMode.Ignore)
            {
                return;
            }

            binIndex = 0;
        }
        else if (value > maximum)
        {
            if (outOfRangeMode == (int)HistogramOutOfRangeMode.Ignore)
            {
                return;
            }

            binIndex = binCount - 1;
        }
        else if (value == maximum)
        {
            binIndex = binCount - 1;
        }
        else
        {
            binIndex = (int)((value - minimum) * scale);
        }

        if ((uint)binIndex < (uint)binCount)
        {
            Atomic.Add(ref histogram[binIndex], 1);
        }
    }

    /// <summary>Executes the internal double-precision unary Map kernel.</summary>
    public static void MapDouble(
        Index1D index,
        ArrayView<double> source,
        ArrayView<double> destination,
        ArrayView<DoubleGpuInstruction> program,
        int instructionCount)
    {
        destination[index] = EvaluateDouble(
            source[index],
            0d,
            program,
            instructionCount);
    }

    /// <summary>Executes the internal double-precision Zip kernel.</summary>
    public static void ZipDouble(
        Index1D index,
        ArrayView<double> left,
        ArrayView<double> right,
        ArrayView<double> destination,
        ArrayView<DoubleGpuInstruction> program,
        int instructionCount)
    {
        destination[index] = EvaluateDouble(
            left[index],
            right[index],
            program,
            instructionCount);
    }

    /// <summary>Executes one double-precision reduction stage.</summary>
    public static void ReduceDouble(
        Index1D outputIndex,
        ArrayView<double> source,
        ArrayView<double> destination,
        int sourceLength,
        int reduction)
    {
        int start = outputIndex * ReductionElementsPerOutput;
        int end = XMath.Min(start + ReductionElementsPerOutput, sourceLength);
        double result = reduction == (int)ComputeReductionKind.Sum
            ? 0d
            : source[start];
        int firstIndex = reduction == (int)ComputeReductionKind.Sum
            ? start
            : start + 1;
        for (int index = firstIndex; index < end; index++)
        {
            double value = source[index];
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

    /// <summary>Executes the internal integer unary Map kernel.</summary>
    public static void MapInt(
        Index1D index,
        ArrayView<int> source,
        ArrayView<int> destination,
        ArrayView<IntGpuInstruction> program,
        int instructionCount)
    {
        destination[index] = EvaluateInt(
            source[index],
            0,
            program,
            instructionCount);
    }

    /// <summary>Executes the internal integer Zip kernel.</summary>
    public static void ZipInt(
        Index1D index,
        ArrayView<int> left,
        ArrayView<int> right,
        ArrayView<int> destination,
        ArrayView<IntGpuInstruction> program,
        int instructionCount)
    {
        destination[index] = EvaluateInt(
            left[index],
            right[index],
            program,
            instructionCount);
    }

    /// <summary>Executes one integer reduction stage.</summary>
    public static void ReduceInt(
        Index1D outputIndex,
        ArrayView<int> source,
        ArrayView<int> destination,
        int sourceLength,
        int reduction)
    {
        int start = outputIndex * ReductionElementsPerOutput;
        int end = XMath.Min(start + ReductionElementsPerOutput, sourceLength);
        int result = reduction == (int)ComputeReductionKind.Sum
            ? 0
            : source[start];
        int firstIndex = reduction == (int)ComputeReductionKind.Sum
            ? start
            : start + 1;
        for (int index = firstIndex; index < end; index++)
        {
            int value = source[index];
            if (reduction == (int)ComputeReductionKind.Sum)
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

    private static double EvaluateDouble(
        double parameter0,
        double parameter1,
        ArrayView<DoubleGpuInstruction> program,
        int instructionCount)
    {
        ArrayView<double> stack =
            LocalMemory.Allocate<double>(
                GpuProgramCompiler.MaximumStackDepth);
        int stackPointer = 0;
        for (int instructionIndex = 0;
             instructionIndex < instructionCount;
             instructionIndex++)
        {
            DoubleGpuInstruction instruction = program[instructionIndex];
            int operation = instruction.OpCode;
            if (operation == (int)NumericOpCode.Parameter0)
            {
                stack[stackPointer++] = parameter0;
            }
            else if (operation == (int)NumericOpCode.Parameter1)
            {
                stack[stackPointer++] = parameter1;
            }
            else if (operation == (int)NumericOpCode.Constant)
            {
                stack[stackPointer++] = instruction.Operand;
            }
            else if (operation == (int)NumericOpCode.Negate)
            {
                stack[stackPointer - 1] = -stack[stackPointer - 1];
            }
            else if (operation == (int)NumericOpCode.Abs)
            {
                stack[stackPointer - 1] = XMath.Abs(stack[stackPointer - 1]);
            }
            else if (operation == (int)NumericOpCode.Sqrt)
            {
                stack[stackPointer - 1] = XMath.Sqrt(stack[stackPointer - 1]);
            }
            else if (operation == (int)NumericOpCode.Sin)
            {
                stack[stackPointer - 1] = XMath.Sin(stack[stackPointer - 1]);
            }
            else if (operation == (int)NumericOpCode.Cos)
            {
                stack[stackPointer - 1] = XMath.Cos(stack[stackPointer - 1]);
            }
            else if (operation == (int)NumericOpCode.Tan)
            {
                stack[stackPointer - 1] = XMath.Tan(stack[stackPointer - 1]);
            }
            else if (operation == (int)NumericOpCode.Exp)
            {
                stack[stackPointer - 1] = XMath.Exp(stack[stackPointer - 1]);
            }
            else if (operation == (int)NumericOpCode.Log)
            {
                stack[stackPointer - 1] = XMath.Log(stack[stackPointer - 1]);
            }
            else if (operation == (int)NumericOpCode.Log10)
            {
                stack[stackPointer - 1] = XMath.Log10(stack[stackPointer - 1]);
            }
            else if (operation == (int)NumericOpCode.Floor)
            {
                stack[stackPointer - 1] = XMath.Floor(stack[stackPointer - 1]);
            }
            else if (operation == (int)NumericOpCode.Ceiling)
            {
                stack[stackPointer - 1] = XMath.Ceiling(stack[stackPointer - 1]);
            }
            else if (operation == (int)NumericOpCode.Round)
            {
                stack[stackPointer - 1] = XMath.Round(stack[stackPointer - 1]);
            }
            else if (operation == (int)NumericOpCode.Clamp)
            {
                double maximum = stack[--stackPointer];
                double minimum = stack[--stackPointer];
                stack[stackPointer - 1] = XMath.Clamp(
                    stack[stackPointer - 1],
                    minimum,
                    maximum);
            }
            else
            {
                double right = stack[--stackPointer];
                double left = stack[stackPointer - 1];
                stack[stackPointer - 1] =
                    ApplyDoubleBinary(operation, left, right);
            }
        }

        return stack[0];
    }

    private static double ApplyDoubleBinary(
        int operation,
        double left,
        double right)
    {
        if (operation == (int)NumericOpCode.Add)
        {
            return left + right;
        }

        if (operation == (int)NumericOpCode.Subtract)
        {
            return left - right;
        }

        if (operation == (int)NumericOpCode.Multiply)
        {
            return left * right;
        }

        if (operation == (int)NumericOpCode.Divide)
        {
            return left / right;
        }

        if (operation == (int)NumericOpCode.Min)
        {
            return XMath.Min(left, right);
        }

        if (operation == (int)NumericOpCode.Max)
        {
            return XMath.Max(left, right);
        }

        return XMath.Pow(left, right);
    }

    private static int EvaluateInt(
        int parameter0,
        int parameter1,
        ArrayView<IntGpuInstruction> program,
        int instructionCount)
    {
        ArrayView<int> stack =
            LocalMemory.Allocate<int>(
                GpuProgramCompiler.MaximumStackDepth);
        int stackPointer = 0;
        for (int instructionIndex = 0;
             instructionIndex < instructionCount;
             instructionIndex++)
        {
            IntGpuInstruction instruction = program[instructionIndex];
            int operation = instruction.OpCode;
            if (operation == (int)NumericOpCode.Parameter0)
            {
                stack[stackPointer++] = parameter0;
            }
            else if (operation == (int)NumericOpCode.Parameter1)
            {
                stack[stackPointer++] = parameter1;
            }
            else if (operation == (int)NumericOpCode.Constant)
            {
                stack[stackPointer++] = instruction.Operand;
            }
            else if (operation == (int)NumericOpCode.Negate)
            {
                stack[stackPointer - 1] = -stack[stackPointer - 1];
            }
            else if (operation == (int)NumericOpCode.Abs)
            {
                int value = stack[stackPointer - 1];
                stack[stackPointer - 1] = value < 0 ? -value : value;
            }
            else if (operation == (int)NumericOpCode.Clamp)
            {
                int maximum = stack[--stackPointer];
                int minimum = stack[--stackPointer];
                stack[stackPointer - 1] = XMath.Clamp(
                    stack[stackPointer - 1],
                    minimum,
                    maximum);
            }
            else
            {
                int right = stack[--stackPointer];
                int left = stack[stackPointer - 1];
                stack[stackPointer - 1] =
                    ApplyIntBinary(operation, left, right);
            }
        }

        return stack[0];
    }

    private static int ApplyIntBinary(
        int operation,
        int left,
        int right)
    {
        if (operation == (int)NumericOpCode.Add)
        {
            return left + right;
        }

        if (operation == (int)NumericOpCode.Subtract)
        {
            return left - right;
        }

        if (operation == (int)NumericOpCode.Multiply)
        {
            return left * right;
        }

        if (operation == (int)NumericOpCode.Divide)
        {
            return left / right;
        }

        return operation == (int)NumericOpCode.Min
            ? XMath.Min(left, right)
            : XMath.Max(left, right);
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
