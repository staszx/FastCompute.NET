using FastCompute.Gpu;
using ILGPU;
using ILGPU.Runtime;

namespace FastCompute;

public sealed partial class ComputeContext
{
    private Action<AcceleratorStream, Index1D, ArrayView<Complex32>, int, int, int, int>? fftBitReverseKernel;
    private Action<AcceleratorStream, Index1D, ArrayView<Complex32>, int, int, int, int, int, int>? fftStageKernel;
    private Action<AcceleratorStream, Index1D, ArrayView<Complex32>, float>? fftScaleKernel;
    private Action<AcceleratorStream, Index1D, ArrayView<Complex32>, ArrayView<float>>? phaseSpectrumKernel;

    internal float[] ExecutePhaseSpectrum(Complex32[] data)
    {
        ThrowIfDisposed();
        using MemoryBuffer1D<Complex32, Stride1D.Dense> input = accelerator.Allocate1D(data);
        using MemoryBuffer1D<float, Stride1D.Dense> output = accelerator.Allocate1D<float>(data.Length);
        phaseSpectrumKernel ??= accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<Complex32>, ArrayView<float>>(FourierGpuKernels.Phase);
        phaseSpectrumKernel(accelerator.DefaultStream, data.Length, input.View, output.View);
        accelerator.Synchronize();
        return output.GetAsArray1D();
    }

    internal void ExecuteFft(Complex32[] data, int width, int height, FourierDirection direction)
    {
        ThrowIfDisposed();
        using MemoryBuffer1D<Complex32, Stride1D.Dense> buffer = accelerator.Allocate1D(data);
        TransformFftDimension(buffer.View, width, height, width, 1, direction);
        if (height > 1) TransformFftDimension(buffer.View, height, width, 1, width, direction);
        if (direction == FourierDirection.Inverse)
            GetFourierKernel(ref fftScaleKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<Complex32>, float>(FourierGpuKernels.Scale))(accelerator.DefaultStream, data.Length, buffer.View, 1f / data.Length);
        accelerator.Synchronize();
        buffer.View.CopyToCPU(accelerator.DefaultStream, data);
        accelerator.Synchronize();
    }

    private void TransformFftDimension(ArrayView<Complex32> data, int dimensionLength, int transformCount, int baseStride, int elementStride, FourierDirection direction)
    {
        GetFourierKernel(ref fftBitReverseKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<Complex32>, int, int, int, int>(FourierGpuKernels.BitReverse))(
            accelerator.DefaultStream, checked(dimensionLength * transformCount), data, dimensionLength, transformCount, baseStride, elementStride);
        int butterflyCount = checked(transformCount * (dimensionLength >> 1));
        for (int size = 2; size <= dimensionLength; size <<= 1)
            GetFourierKernel(ref fftStageKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<Complex32>, int, int, int, int, int, int>(FourierGpuKernels.Stage))(
                accelerator.DefaultStream, butterflyCount, data, dimensionLength, transformCount, baseStride, elementStride, size, direction == FourierDirection.Inverse ? 1 : 0);
    }

    private IReadOnlyList<ComputeCompilationResult> PrecompileFourierKernels() =>
    [
        CompileFourierKernel(
            () => fftBitReverseKernel is not null,
            () => _ = GetFourierKernel(ref fftBitReverseKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<Complex32>, int, int, int, int>(FourierGpuKernels.BitReverse))),
        CompileFourierKernel(
            () => fftStageKernel is not null,
            () => _ = GetFourierKernel(ref fftStageKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<Complex32>, int, int, int, int, int, int>(FourierGpuKernels.Stage))),
        CompileFourierKernel(
            () => fftScaleKernel is not null,
            () => _ = GetFourierKernel(ref fftScaleKernel, () => accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<Complex32>, float>(FourierGpuKernels.Scale))),
        CompileFourierKernel(
            () => phaseSpectrumKernel is not null,
            () => phaseSpectrumKernel ??= accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<Complex32>, ArrayView<float>>(FourierGpuKernels.Phase))
    ];

    private ComputeCompilationResult CompileFourierKernel(Func<bool> isCached, Action compile)
    {
        bool cacheHit = isCached();
        long started = System.Diagnostics.Stopwatch.GetTimestamp();
        compile();
        return new ComputeCompilationResult(
            cacheHit,
            TimeSpan.Zero,
            cacheHit ? TimeSpan.Zero : System.Diagnostics.Stopwatch.GetElapsedTime(started),
            ComputeBackendKind.Gpu,
            accelerator.Name);
    }

    private static TKernel GetFourierKernel<TKernel>(ref TKernel? cache, Func<TKernel> compile) where TKernel : class
    {
        TKernel? existing = Volatile.Read(ref cache);
        if (existing is not null) return existing;
        TKernel candidate = compile();
        return Interlocked.CompareExchange(ref cache, candidate, null) ?? candidate;
    }
}
