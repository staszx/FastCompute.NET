using System.ComponentModel;
using ILGPU;
using ILGPU.Algorithms;

namespace FastCompute.Gpu;

/// <summary>Contains image-processing entry points required by ILGPU.</summary>
[EditorBrowsable(EditorBrowsableState.Never)]
public static class ImageGpuKernels
{
    /// <summary>Executes a byte-component map or conversion.</summary>
    public static void ByteCompositeMap(Index1D index, ArrayView<byte> source, ArrayView<byte> destination, ArrayView<ByteGpuInstruction> program, ArrayView<int> outputOffsets, ArrayView<int> outputInstructionCounts, int sourceComponents, int destinationComponents)
    {
        int inputOffset = index * sourceComponents;
        int outputOffset = index * destinationComponents;
        for (int component = 0; component < destinationComponents; component++)
            destination[outputOffset + component] = (byte)EvaluateByte(source, inputOffset, program, outputOffsets[component], outputInstructionCounts[component]);
    }

    /// <summary>Executes a byte-component projection to float.</summary>
    public static void ByteCompositeProject(Index1D index, ArrayView<byte> source, ArrayView<float> destination, ArrayView<ByteGpuInstruction> program, int instructionCount, int sourceComponents)
    {
        destination[index] = EvaluateByte(source, index * sourceComponents, program, 0, instructionCount);
    }

    /// <summary>Converts byte-component pixels to byte-component pixels.</summary>
    public static void ConvertByteToByte(Index1D index, ArrayView<byte> source, ArrayView<byte> destination, int sourceComponents, int destinationComponents, int sourceEncoding, int destinationEncoding)
    {
        ReadByte(source, index, sourceComponents, out float red, out float green, out float blue);
        ConvertEncoding(ref red, ref green, ref blue, sourceEncoding, destinationEncoding);
        WriteByte(destination, index, destinationComponents, red, green, blue);
    }

    /// <summary>Converts byte-component pixels to float-component pixels.</summary>
    public static void ConvertByteToFloat(Index1D index, ArrayView<byte> source, ArrayView<float> destination, int sourceComponents, int destinationComponents, int sourceEncoding, int destinationEncoding)
    {
        ReadByte(source, index, sourceComponents, out float red, out float green, out float blue);
        ConvertEncoding(ref red, ref green, ref blue, sourceEncoding, destinationEncoding);
        WriteFloat(destination, index, destinationComponents, red, green, blue);
    }

    /// <summary>Converts float-component pixels to byte-component pixels.</summary>
    public static void ConvertFloatToByte(Index1D index, ArrayView<float> source, ArrayView<byte> destination, int sourceComponents, int destinationComponents, int sourceEncoding, int destinationEncoding)
    {
        ReadFloat(source, index, sourceComponents, out float red, out float green, out float blue);
        ConvertEncoding(ref red, ref green, ref blue, sourceEncoding, destinationEncoding);
        WriteByte(destination, index, destinationComponents, red, green, blue);
    }

    /// <summary>Converts float-component pixels to float-component pixels.</summary>
    public static void ConvertFloatToFloat(Index1D index, ArrayView<float> source, ArrayView<float> destination, int sourceComponents, int destinationComponents, int sourceEncoding, int destinationEncoding)
    {
        ReadFloat(source, index, sourceComponents, out float red, out float green, out float blue);
        ConvertEncoding(ref red, ref green, ref blue, sourceEncoding, destinationEncoding);
        WriteFloat(destination, index, destinationComponents, red, green, blue);
    }

    /// <summary>Subtracts two floating-point image buffers.</summary>
    public static void Subtract(Index1D index, ArrayView<float> left, ArrayView<float> right, ArrayView<float> destination) =>
        destination[index] = left[index] - right[index];

    /// <summary>Applies the horizontal pass of a box blur.</summary>
    public static void BlurHorizontal(Index1D index, ArrayView<float> source, ArrayView<float> destination, int width, int radius)
    {
        int x = index % width;
        int row = index - x;
        int start = XMath.Max(0, x - radius);
        int end = XMath.Min(width - 1, x + radius);
        float sum = 0f;
        for (int current = start; current <= end; current++) sum += source[row + current];
        destination[index] = sum / (end - start + 1);
    }

    /// <summary>Applies the vertical pass of a box blur.</summary>
    public static void BlurVertical(Index1D index, ArrayView<float> source, ArrayView<float> destination, int width, int height, int radius)
    {
        int x = index % width;
        int y = index / width;
        int start = XMath.Max(0, y - radius);
        int end = XMath.Min(height - 1, y + radius);
        float sum = 0f;
        for (int current = start; current <= end; current++) sum += source[(current * width) + x];
        destination[index] = sum / (end - start + 1);
    }

    /// <summary>Applies a linear-time horizontal sliding-window blur, one row per GPU thread.</summary>
    public static void BlurHorizontalSliding(Index1D rowIndex, ArrayView<float> source, ArrayView<float> destination, int width, int radius)
    {
        int row = rowIndex * width;
        int end = XMath.Min(width - 1, radius);
        float sum = 0f;
        for (int x = 0; x <= end; x++) sum += source[row + x];
        for (int x = 0; x < width; x++)
        {
            int start = XMath.Max(0, x - radius);
            end = XMath.Min(width - 1, x + radius);
            destination[row + x] = sum / (end - start + 1);
            int remove = x - radius;
            int add = x + radius + 1;
            if (remove >= 0) sum -= source[row + remove];
            if (add < width) sum += source[row + add];
        }
    }

    /// <summary>Applies a linear-time vertical sliding-window blur, one column per GPU thread.</summary>
    public static void BlurVerticalSliding(Index1D columnIndex, ArrayView<float> source, ArrayView<float> destination, int width, int height, int radius)
    {
        int x = columnIndex;
        int end = XMath.Min(height - 1, radius);
        float sum = 0f;
        for (int y = 0; y <= end; y++) sum += source[(y * width) + x];
        for (int y = 0; y < height; y++)
        {
            int start = XMath.Max(0, y - radius);
            end = XMath.Min(height - 1, y + radius);
            destination[(y * width) + x] = sum / (end - start + 1);
            int remove = y - radius;
            int add = y + radius + 1;
            if (remove >= 0) sum -= source[(remove * width) + x];
            if (add < height) sum += source[(add * width) + x];
        }
    }

    /// <summary>Downsamples one floating-point grayscale image.</summary>
    public static void Downsample(Index1D index, ArrayView<float> source, ArrayView<float> destination, int sourceWidth, int sourceHeight, int destinationWidth, int destinationHeight)
    {
        int destinationX = index % destinationWidth;
        int destinationY = index / destinationWidth;
        int sourceX0 = destinationX * sourceWidth / destinationWidth;
        int sourceX1 = XMath.Max(sourceX0 + 1, (destinationX + 1) * sourceWidth / destinationWidth);
        int sourceY0 = destinationY * sourceHeight / destinationHeight;
        int sourceY1 = XMath.Max(sourceY0 + 1, (destinationY + 1) * sourceHeight / destinationHeight);
        float sum = 0f;
        int count = 0;
        for (int y = sourceY0; y < sourceY1; y++)
        {
            int offset = y * sourceWidth;
            for (int x = sourceX0; x < sourceX1; x++)
            {
                sum += source[offset + x];
                count++;
            }
        }
        destination[index] = sum / count;
    }

    /// <summary>Resizes one row-major floating-point buffer using bilinear interpolation.</summary>
    public static void ResizeBilinear(Index1D index, ArrayView<float> source, ArrayView<float> destination, int sourceWidth, int sourceHeight, int destinationWidth, int destinationHeight)
    {
        int destinationX = index % destinationWidth;
        int destinationY = index / destinationWidth;
        float sourceX = destinationWidth == 1 ? 0f : destinationX * (sourceWidth - 1f) / (destinationWidth - 1f);
        float sourceY = destinationHeight == 1 ? 0f : destinationY * (sourceHeight - 1f) / (destinationHeight - 1f);
        int x0 = (int)XMath.Floor(sourceX);
        int y0 = (int)XMath.Floor(sourceY);
        int x1 = XMath.Min(sourceWidth - 1, x0 + 1);
        int y1 = XMath.Min(sourceHeight - 1, y0 + 1);
        float horizontal = sourceX - x0;
        float vertical = sourceY - y0;
        float top = source[(y0 * sourceWidth) + x0] + ((source[(y0 * sourceWidth) + x1] - source[(y0 * sourceWidth) + x0]) * horizontal);
        float bottom = source[(y1 * sourceWidth) + x0] + ((source[(y1 * sourceWidth) + x1] - source[(y1 * sourceWidth) + x0]) * horizontal);
        destination[index] = top + ((bottom - top) * vertical);
    }

    /// <summary>Calculates local range contrast in a square neighbourhood.</summary>
    public static void LocalContrast(Index1D index, ArrayView<float> source, ArrayView<float> destination, int width, int height, int radius)
    {
        int x = index % width;
        int y = index / width;
        float minimum = float.MaxValue;
        float maximum = float.MinValue;
        int startY = XMath.Max(0, y - radius);
        int endY = XMath.Min(height - 1, y + radius);
        int startX = XMath.Max(0, x - radius);
        int endX = XMath.Min(width - 1, x + radius);
        for (int currentY = startY; currentY <= endY; currentY++)
        for (int currentX = startX; currentX <= endX; currentX++)
        {
            float value = source[(currentY * width) + currentX];
            minimum = XMath.Min(minimum, value);
            maximum = XMath.Max(maximum, value);
        }
        destination[index] = maximum - minimum;
    }

    /// <summary>Accumulates a radial power-spectrum bin and one of three frequency bands.</summary>
    public static void AccumulateRadialSpectrum(
        Index1D index,
        ArrayView<float> power,
        ArrayView<float> radialSums,
        ArrayView<int> radialCounts,
        ArrayView<float> bandSums,
        int width,
        int height,
        int binCount,
        float lowBoundary,
        float middleBoundary)
    {
        int x = index % width;
        int y = index / width;
        int frequencyX = x <= width / 2 ? x : x - width;
        int frequencyY = y <= height / 2 ? y : y - height;
        float maximum = XMath.Sqrt(((width / 2f) * (width / 2f)) + ((height / 2f) * (height / 2f)));
        float normalizedRadius = XMath.Sqrt((frequencyX * frequencyX) + (frequencyY * frequencyY)) / maximum;
        int bin = XMath.Min(binCount - 1, (int)(normalizedRadius * binCount));
        float value = power[index];
        Atomic.Add(ref radialSums[bin], value);
        Atomic.Add(ref radialCounts[bin], 1);
        int band = normalizedRadius < lowBoundary ? 0 : normalizedRadius < middleBoundary ? 1 : 2;
        Atomic.Add(ref bandSums[band], value);
    }

    /// <summary>Calculates local Shannon entropy over normalized grayscale values.</summary>
    public static void LocalEntropy(Index1D index, ArrayView<float> source, ArrayView<float> destination, int width, int height, int radius, int binCount)
    {
        int x = index % width;
        int y = index / width;
        int startX = XMath.Max(0, x - radius);
        int endX = XMath.Min(width - 1, x + radius);
        int startY = XMath.Max(0, y - radius);
        int endY = XMath.Min(height - 1, y + radius);
        int count = (endX - startX + 1) * (endY - startY + 1);
        float entropy = 0f;
        for (int bin = 0; bin < binCount; bin++)
        {
            int binSamples = 0;
            for (int currentY = startY; currentY <= endY; currentY++)
            for (int currentX = startX; currentX <= endX; currentX++)
            {
                float value = XMath.Clamp(source[(currentY * width) + currentX], 0f, 1f);
                int valueBin = XMath.Min(binCount - 1, (int)(value * binCount));
                if (valueBin == bin) binSamples++;
            }
            if (binSamples > 0)
            {
                float probability = binSamples / (float)count;
                entropy -= probability * XMath.Log2(probability);
            }
        }
        destination[index] = entropy;
    }

    private static void ReadByte(ArrayView<byte> source, int index, int components, out float red, out float green, out float blue)
    {
        int offset = index * components;
        red = source[offset] / 255f;
        green = components == 1 ? red : source[offset + 1] / 255f;
        blue = components == 1 ? red : source[offset + 2] / 255f;
    }

    private static void ReadFloat(ArrayView<float> source, int index, int components, out float red, out float green, out float blue)
    {
        int offset = index * components;
        red = source[offset];
        green = components == 1 ? red : source[offset + 1];
        blue = components == 1 ? red : source[offset + 2];
    }

    private static void WriteByte(ArrayView<byte> destination, int index, int components, float red, float green, float blue)
    {
        int offset = index * components;
        if (components == 1)
        {
            destination[offset] = Quantize(Luminance(red, green, blue));
            return;
        }
        destination[offset] = Quantize(red);
        destination[offset + 1] = Quantize(green);
        destination[offset + 2] = Quantize(blue);
    }

    private static void WriteFloat(ArrayView<float> destination, int index, int components, float red, float green, float blue)
    {
        int offset = index * components;
        if (components == 1)
        {
            destination[offset] = Luminance(red, green, blue);
            return;
        }
        destination[offset] = red;
        destination[offset + 1] = green;
        destination[offset + 2] = blue;
    }

    private static void ConvertEncoding(ref float red, ref float green, ref float blue, int sourceEncoding, int destinationEncoding)
    {
        if (sourceEncoding == destinationEncoding) return;
        red = ChangeEncoding(red, sourceEncoding);
        green = ChangeEncoding(green, sourceEncoding);
        blue = ChangeEncoding(blue, sourceEncoding);
    }

    private static float ChangeEncoding(float value, int sourceEncoding) => sourceEncoding == 0
        ? (value <= 0.04045f ? value / 12.92f : XMath.Pow((value + 0.055f) / 1.055f, 2.4f))
        : (value <= 0.0031308f ? value * 12.92f : (1.055f * XMath.Pow(value, 1f / 2.4f)) - 0.055f);

    private static float Luminance(float red, float green, float blue) =>
        (0.2126f * red) + (0.7152f * green) + (0.0722f * blue);

    private static byte Quantize(float value) =>
        (byte)XMath.Clamp((int)XMath.Round(value * 255f), 0, 255);

    private static int EvaluateByte(ArrayView<byte> source, int sourceOffset, ArrayView<ByteGpuInstruction> program, int programOffset, int instructionCount)
    {
        ArrayView<int> stack = LocalMemory.Allocate<int>(GpuProgramCompiler.MaximumStackDepth);
        int stackPointer = 0;
        int end = programOffset + instructionCount;
        for (int index = programOffset; index < end; index++)
        {
            ByteGpuInstruction instruction = program[index];
            if (instruction.OpCode == ByteGpuOpCode.Component)
                stack[stackPointer++] = source[sourceOffset + instruction.Operand];
            else if (instruction.OpCode == ByteGpuOpCode.Constant)
                stack[stackPointer++] = instruction.Operand;
            else if (instruction.OpCode == ByteGpuOpCode.Negate)
                stack[stackPointer - 1] = -stack[stackPointer - 1];
            else if (instruction.OpCode == ByteGpuOpCode.Narrow)
                stack[stackPointer - 1] &= 255;
            else
            {
                int right = stack[--stackPointer];
                int left = stack[stackPointer - 1];
                stack[stackPointer - 1] = instruction.OpCode == ByteGpuOpCode.Add
                    ? left + right
                    : instruction.OpCode == ByteGpuOpCode.Subtract
                        ? left - right
                        : instruction.OpCode == ByteGpuOpCode.Multiply
                            ? left * right
                            : left / right;
            }
        }
        return stack[0];
    }
}
