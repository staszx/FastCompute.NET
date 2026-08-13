using FastCompute.Gpu;
using ILGPU;
using ILGPU.Runtime;

namespace FastCompute;

public sealed partial class ComputeContext
{
    private Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>? convolution1DKernel;
    private Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int>? convolution2DKernel;

    internal float[] ExecuteConvolution1D(float[] source, float[] kernel, ConvolutionBoundary boundary)
    {
        ThrowIfDisposed();
        using MemoryBuffer1D<float, Stride1D.Dense> input = accelerator.Allocate1D(source);
        using MemoryBuffer1D<float, Stride1D.Dense> weights = accelerator.Allocate1D(kernel);
        using MemoryBuffer1D<float, Stride1D.Dense> output = accelerator.Allocate1D<float>(source.Length);
        convolution1DKernel ??= accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(ConvolutionGpuKernels.Convolve1D);
        convolution1DKernel(accelerator.DefaultStream, source.Length, input.View, weights.View, output.View, (int)boundary);
        accelerator.Synchronize();
        return output.GetAsArray1D();
    }

    internal float[] ExecuteConvolution2D(float[] source, int width, int height, float[] kernel, int kernelWidth, int kernelHeight, ConvolutionBoundary boundary)
    {
        ThrowIfDisposed();
        using MemoryBuffer1D<float, Stride1D.Dense> input = accelerator.Allocate1D(source);
        using MemoryBuffer1D<float, Stride1D.Dense> weights = accelerator.Allocate1D(kernel);
        using MemoryBuffer1D<float, Stride1D.Dense> output = accelerator.Allocate1D<float>(source.Length);
        convolution2DKernel ??= accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int>(ConvolutionGpuKernels.Convolve2D);
        convolution2DKernel(accelerator.DefaultStream, source.Length, input.View, weights.View, output.View, width, height, kernelWidth, kernelHeight, (int)boundary);
        accelerator.Synchronize();
        return output.GetAsArray1D();
    }
}
