using System.Numerics;
using FastCompute.Backends.Gpu;

namespace FastCompute;

public static partial class Compute
{
    /// <summary>Maps values at or above a threshold to one and all other values to zero.</summary>
    public static float[] Threshold(ReadOnlySpan<float> source, float threshold, ComputeOptions? options = null)
    {
        if (!float.IsFinite(threshold)) throw new ArgumentOutOfRangeException(nameof(threshold));
        if (source.IsEmpty) return [];
        ComputeOptions effective = options ?? ComputeOptions.Default;
        ComputeBackendKind backend = ResolveThresholdBackend(effective, source.Length);
        float[] input = source.ToArray();
        if (backend == ComputeBackendKind.Gpu)
            return GpuComputeBackend.ResolveContext(effective).ExecuteThreshold(input, threshold);

        var result = GC.AllocateUninitializedArray<float>(input.Length);
        if (backend == ComputeBackendKind.ParallelCpu)
        {
            Parallel.For(0, input.Length, new ParallelOptions
            {
                CancellationToken = effective.CancellationToken,
                MaxDegreeOfParallelism = effective.MaxDegreeOfParallelism ?? -1
            }, index => result[index] = input[index] >= threshold ? 1f : 0f);
            return result;
        }
        int index = 0;
        if (backend == ComputeBackendKind.Simd)
        {
            int lanes = Vector<float>.Count;
            int vectorEnd = input.Length - (input.Length % lanes);
            var thresholdVector = new Vector<float>(threshold);
            for (; index < vectorEnd; index += lanes)
            {
                if ((index & 0xFFFF) == 0) effective.CancellationToken.ThrowIfCancellationRequested();
                Vector<int> mask = Vector.GreaterThanOrEqual(new Vector<float>(input, index), thresholdVector);
                Vector.ConditionalSelect(mask, Vector<float>.One, Vector<float>.Zero).CopyTo(result, index);
            }
        }
        for (; index < input.Length; index++)
        {
            if ((index & 0xFFFF) == 0) effective.CancellationToken.ThrowIfCancellationRequested();
            result[index] = input[index] >= threshold ? 1f : 0f;
        }
        return result;
    }

    private static ComputeBackendKind ResolveThresholdBackend(ComputeOptions options, int length)
    {
        ArgumentNullException.ThrowIfNull(options.Thresholds);
        if (options.GpuContext is not null && options.PreferredGpuAcceleratorIndex is not null)
            throw new ArgumentException("GpuContext and PreferredGpuAcceleratorIndex cannot be used together.", nameof(options));
        if (options.Backend == ComputeBackendKind.Simd && !Vector.IsHardwareAccelerated)
            throw new ComputeBackendUnavailableException(ComputeBackendKind.Simd);
        if (options.Backend != ComputeBackendKind.Auto) return options.Backend;
        if (length >= options.Thresholds.GpuSimpleThreshold && (options.GpuContext is not null || GpuComputeBackend.HasHardwareAccelerator)) return ComputeBackendKind.Gpu;
        if (length >= options.Thresholds.ParallelThreshold) return ComputeBackendKind.ParallelCpu;
        if (length >= options.Thresholds.SimdThreshold && Vector.IsHardwareAccelerated) return ComputeBackendKind.Simd;
        return ComputeBackendKind.Scalar;
    }
}
