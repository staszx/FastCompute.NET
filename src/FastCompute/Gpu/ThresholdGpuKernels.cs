using System.ComponentModel;
using ILGPU;

namespace FastCompute.Gpu;

/// <summary>Contains generic threshold kernels required by ILGPU.</summary>
[EditorBrowsable(EditorBrowsableState.Never)]
public static class ThresholdGpuKernels
{
    /// <summary>Maps values at or above a threshold to one and all other values to zero.</summary>
    public static void Apply(Index1D index, ArrayView<float> source, ArrayView<float> destination, float threshold) =>
        destination[index] = source[index] >= threshold ? 1f : 0f;
}
