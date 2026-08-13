using FastCompute.Gpu;
using ILGPU;
using ILGPU.Runtime;

namespace FastCompute;

public sealed partial class ComputeContext
{
    private Action<AcceleratorStream, Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<ByteGpuInstruction>, ArrayView<int>, ArrayView<int>, int, int>? byteCompositeMapKernel;
    private Action<AcceleratorStream, Index1D, ArrayView<byte>, ArrayView<float>, ArrayView<ByteGpuInstruction>, int, int>? byteCompositeProjectKernel;

    internal byte[] ExecuteByteCompositeMap(byte[] source, int valueCount, int sourceComponents, ByteCompositeGpuProgram program)
    {
        ThrowIfDisposed();
        int destinationComponents = program.OutputOffsets.Length;
        using var input = accelerator.Allocate1D(source);
        using var output = accelerator.Allocate1D<byte>(checked(valueCount * destinationComponents));
        using var instructions = accelerator.Allocate1D(program.Instructions);
        using var offsets = accelerator.Allocate1D(program.OutputOffsets);
        using var counts = accelerator.Allocate1D(program.OutputInstructionCounts);
        byteCompositeMapKernel ??= accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<ByteGpuInstruction>, ArrayView<int>, ArrayView<int>, int, int>(ByteCompositeGpuKernels.Map);
        byteCompositeMapKernel(accelerator.DefaultStream, valueCount, input.View, output.View, instructions.View, offsets.View, counts.View, sourceComponents, destinationComponents);
        accelerator.Synchronize();
        return output.GetAsArray1D();
    }

    internal float[] ExecuteByteCompositeProjection(byte[] source, int valueCount, int sourceComponents, ByteCompositeGpuProgram program)
    {
        ThrowIfDisposed();
        using var input = accelerator.Allocate1D(source);
        using var output = accelerator.Allocate1D<float>(valueCount);
        using var instructions = accelerator.Allocate1D(program.Instructions);
        byteCompositeProjectKernel ??= accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<byte>, ArrayView<float>, ArrayView<ByteGpuInstruction>, int, int>(ByteCompositeGpuKernels.Project);
        byteCompositeProjectKernel(accelerator.DefaultStream, valueCount, input.View, output.View, instructions.View, program.Instructions.Length, sourceComponents);
        accelerator.Synchronize();
        return output.GetAsArray1D();
    }

    private IReadOnlyList<ComputeCompilationResult> PrecompileByteCompositeKernels()
    {
        var results = new List<ComputeCompilationResult>(2);
        bool mapCached = byteCompositeMapKernel is not null;
        long started = System.Diagnostics.Stopwatch.GetTimestamp();
        byteCompositeMapKernel ??= accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<ByteGpuInstruction>, ArrayView<int>, ArrayView<int>, int, int>(ByteCompositeGpuKernels.Map);
        results.Add(new ComputeCompilationResult(mapCached, TimeSpan.Zero, mapCached ? TimeSpan.Zero : System.Diagnostics.Stopwatch.GetElapsedTime(started), ComputeBackendKind.Gpu, accelerator.Name));
        bool projectCached = byteCompositeProjectKernel is not null;
        started = System.Diagnostics.Stopwatch.GetTimestamp();
        byteCompositeProjectKernel ??= accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<byte>, ArrayView<float>, ArrayView<ByteGpuInstruction>, int, int>(ByteCompositeGpuKernels.Project);
        results.Add(new ComputeCompilationResult(projectCached, TimeSpan.Zero, projectCached ? TimeSpan.Zero : System.Diagnostics.Stopwatch.GetElapsedTime(started), ComputeBackendKind.Gpu, accelerator.Name));
        return results;
    }
}
