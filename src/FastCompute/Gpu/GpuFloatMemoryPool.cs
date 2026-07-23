using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;

namespace FastCompute.Gpu;

internal sealed class GpuFloatMemoryPool : IDisposable
{
    private readonly Accelerator accelerator;
    private readonly ConcurrentDictionary<
        int,
        ConcurrentBag<MemoryBuffer1D<float, Stride1D.Dense>>> available = new();
    private readonly ConcurrentBag<
        MemoryBuffer1D<float, Stride1D.Dense>> allocated = new();
    private long allocatedCount;
    private long rentalCount;
    private long reuseCount;
    private int disposed;

    internal GpuFloatMemoryPool(Accelerator accelerator)
    {
        this.accelerator = accelerator;
    }

    internal ComputeMemoryPoolStatistics Statistics
    {
        get
        {
            int availableCount = 0;
            foreach (ConcurrentBag<
                         MemoryBuffer1D<float, Stride1D.Dense>> buffers
                     in available.Values)
            {
                availableCount += buffers.Count;
            }

            return new ComputeMemoryPoolStatistics(
                Interlocked.Read(ref allocatedCount),
                Interlocked.Read(ref rentalCount),
                Interlocked.Read(ref reuseCount),
                availableCount);
        }
    }

    internal Lease Rent(int length)
    {
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref disposed) != 0,
            this);

        Interlocked.Increment(ref rentalCount);
        ConcurrentBag<MemoryBuffer1D<float, Stride1D.Dense>> buffers =
            available.GetOrAdd(length, static _ => new());

        if (buffers.TryTake(out MemoryBuffer1D<float, Stride1D.Dense>? buffer))
        {
            Interlocked.Increment(ref reuseCount);
            return new Lease(this, buffer);
        }

        buffer = accelerator.Allocate1D<float>(length);
        allocated.Add(buffer);
        Interlocked.Increment(ref allocatedCount);
        return new Lease(this, buffer);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposed, 1) != 0)
        {
            return;
        }

        while (allocated.TryTake(
                   out MemoryBuffer1D<float, Stride1D.Dense>? buffer))
        {
            buffer.Dispose();
        }

        available.Clear();
    }

    private void Return(MemoryBuffer1D<float, Stride1D.Dense> buffer)
    {
        if (Volatile.Read(ref disposed) != 0)
        {
            return;
        }

        available
            .GetOrAdd(checked((int)buffer.Length), static _ => new())
            .Add(buffer);
    }

    internal sealed class Lease : IDisposable
    {
        private GpuFloatMemoryPool? pool;

        internal Lease(
            GpuFloatMemoryPool pool,
            MemoryBuffer1D<float, Stride1D.Dense> buffer)
        {
            this.pool = pool;
            Buffer = buffer;
        }

        internal MemoryBuffer1D<float, Stride1D.Dense> Buffer { get; }

        public void Dispose()
        {
            GpuFloatMemoryPool? owner =
                Interlocked.Exchange(ref pool, null);
            owner?.Return(Buffer);
        }
    }
}
