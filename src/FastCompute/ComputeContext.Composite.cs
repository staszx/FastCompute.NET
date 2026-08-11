using FastCompute.Gpu;
using ILGPU;
using ILGPU.Runtime;

namespace FastCompute;

public sealed partial class ComputeContext
{
    private Action<
        AcceleratorStream,
        Index1D,
        ArrayView<float>,
        ArrayView<float>,
        ArrayView<GpuInstruction>,
        ArrayView<int>,
        ArrayView<int>,
        int,
        int>? compositeMapKernel;

    internal float[] ExecuteCompositeValue(
        float[] source,
        int valueCount,
        int sourceComponentCount,
        CompositeGpuProgram program)
    {
        ThrowIfDisposed();
        if (valueCount == 0)
        {
            return [];
        }

        int outputComponentCount = program.OutputOffsets.Length;
        using MemoryBuffer1D<float, Stride1D.Dense> sourceBuffer =
            accelerator.Allocate1D(source);
        using MemoryBuffer1D<float, Stride1D.Dense> destinationBuffer =
            accelerator.Allocate1D<float>(
                checked(valueCount * outputComponentCount));
        using MemoryBuffer1D<GpuInstruction, Stride1D.Dense> programBuffer =
            accelerator.Allocate1D(program.Instructions);
        using MemoryBuffer1D<int, Stride1D.Dense> offsetBuffer =
            accelerator.Allocate1D(program.OutputOffsets);
        using MemoryBuffer1D<int, Stride1D.Dense> countBuffer =
            accelerator.Allocate1D(program.OutputInstructionCounts);

        Action<
            AcceleratorStream,
            Index1D,
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<GpuInstruction>,
            ArrayView<int>,
            ArrayView<int>,
            int,
            int> kernel = GetCompositeMapKernel();

        kernel(
            accelerator.DefaultStream,
            valueCount,
            sourceBuffer.View,
            destinationBuffer.View,
            programBuffer.View,
            offsetBuffer.View,
            countBuffer.View,
            sourceComponentCount,
            outputComponentCount);
        accelerator.Synchronize();
        return destinationBuffer.GetAsArray1D();
    }

    private Action<
        AcceleratorStream,
        Index1D,
        ArrayView<float>,
        ArrayView<float>,
        ArrayView<GpuInstruction>,
        ArrayView<int>,
        ArrayView<int>,
        int,
        int> GetCompositeMapKernel()
    {
        Action<
            AcceleratorStream,
            Index1D,
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<GpuInstruction>,
            ArrayView<int>,
            ArrayView<int>,
            int,
            int>? cached = Volatile.Read(ref compositeMapKernel);
        if (cached is not null)
        {
            return cached;
        }

        var candidate = accelerator.LoadAutoGroupedKernel<
            Index1D,
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<GpuInstruction>,
            ArrayView<int>,
            ArrayView<int>,
            int,
            int>(GpuKernels.CompositeMap);
        return Interlocked.CompareExchange(
                   ref compositeMapKernel,
                   candidate,
                   null) ?? candidate;
    }
}
