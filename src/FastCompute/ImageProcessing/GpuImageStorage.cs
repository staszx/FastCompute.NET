using ILGPU;
using ILGPU.Runtime;

namespace FastCompute.ImageProcessing;

internal abstract class GpuImageStorage : IDisposable
{
    protected GpuImageStorage(int scalarLength) => ScalarLength = scalarLength;

    internal int ScalarLength { get; }

    internal abstract bool IsFloat { get; }

    public abstract void Dispose();
}

internal sealed class GpuByteImageStorage(
    MemoryBuffer1D<byte, Stride1D.Dense> buffer)
    : GpuImageStorage((int)buffer.Length)
{
    internal MemoryBuffer1D<byte, Stride1D.Dense> Buffer { get; } = buffer;

    internal override bool IsFloat => false;

    public override void Dispose() => Buffer.Dispose();
}

internal sealed class GpuFloatImageStorage(
    MemoryBuffer1D<float, Stride1D.Dense> buffer)
    : GpuImageStorage((int)buffer.Length)
{
    internal MemoryBuffer1D<float, Stride1D.Dense> Buffer { get; } = buffer;

    internal override bool IsFloat => true;

    public override void Dispose() => Buffer.Dispose();
}
