using System.ComponentModel;
using ILGPU;

namespace FastCompute.Gpu;

/// <summary>Contains generic convolution kernels required by ILGPU.</summary>
[EditorBrowsable(EditorBrowsableState.Never)]
public static class ConvolutionGpuKernels
{
    /// <summary>Convolves a one-dimensional numeric buffer.</summary>
    public static void Convolve1D(
        Index1D index,
        ArrayView<float> source,
        ArrayView<float> kernel,
        ArrayView<float> destination,
        int boundary)
    {
        int kernelLength = (int)kernel.Length;
        int sourceLength = (int)source.Length;
        int radius = kernelLength / 2;
        float sum = 0f;
        for (int k = 0; k < kernelLength; k++)
        {
            int sourceIndex = index + k - radius;
            if (boundary == 0)
                sourceIndex = sourceIndex < 0 ? 0 : sourceIndex >= sourceLength ? sourceLength - 1 : sourceIndex;
            else if (sourceIndex < 0 || sourceIndex >= sourceLength)
                continue;
            sum += source[sourceIndex] * kernel[k];
        }
        destination[index] = sum;
    }

    /// <summary>Convolves a row-major two-dimensional numeric buffer.</summary>
    public static void Convolve2D(
        Index1D index,
        ArrayView<float> source,
        ArrayView<float> kernel,
        ArrayView<float> destination,
        int width,
        int height,
        int kernelWidth,
        int kernelHeight,
        int boundary)
    {
        int x = index % width;
        int y = index / width;
        int radiusX = kernelWidth / 2;
        int radiusY = kernelHeight / 2;
        float sum = 0f;
        for (int ky = 0; ky < kernelHeight; ky++)
        for (int kx = 0; kx < kernelWidth; kx++)
        {
            int sourceX = x + kx - radiusX;
            int sourceY = y + ky - radiusY;
            if (boundary == 0)
            {
                sourceX = sourceX < 0 ? 0 : sourceX >= width ? width - 1 : sourceX;
                sourceY = sourceY < 0 ? 0 : sourceY >= height ? height - 1 : sourceY;
            }
            else if (sourceX < 0 || sourceX >= width || sourceY < 0 || sourceY >= height)
            {
                continue;
            }
            sum += source[(sourceY * width) + sourceX] * kernel[(ky * kernelWidth) + kx];
        }
        destination[index] = sum;
    }
}
