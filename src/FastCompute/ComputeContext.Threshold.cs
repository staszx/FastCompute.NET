using FastCompute.Gpu;
using ILGPU;
using ILGPU.Runtime;

namespace FastCompute;

public sealed partial class ComputeContext
{
    private Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, float>? thresholdKernel;

    internal float[] ExecuteThreshold(float[] source, float threshold)
    {
        ThrowIfDisposed();
        using MemoryBuffer1D<float, Stride1D.Dense> input = accelerator.Allocate1D(source);
        using MemoryBuffer1D<float, Stride1D.Dense> output = accelerator.Allocate1D<float>(source.Length);
        thresholdKernel ??= accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>, float>(ThresholdGpuKernels.Apply);
        thresholdKernel(accelerator.DefaultStream, source.Length, input.View, output.View, threshold);
        accelerator.Synchronize();
        return output.GetAsArray1D();
    }
}
