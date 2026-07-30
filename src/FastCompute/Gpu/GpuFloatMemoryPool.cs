using ILGPU;
using ILGPU.Runtime;

namespace FastCompute.Gpu;

internal sealed class GpuFloatMemoryPool : IDisposable
{
    private readonly Accelerator accelerator;
    private readonly long limitBytes;
    private readonly object syncRoot = new();
    private readonly Dictionary<int, LinkedList<Entry>> availableByLength =
        new();
    private readonly LinkedList<Entry> availableByAge = new();
    private readonly HashSet<Entry> live = [];
    private long allocatedCount;
    private long rentalCount;
    private long reuseCount;
    private long evictedCount;
    private long retainedBytes;
    private bool disposed;

    internal GpuFloatMemoryPool(Accelerator accelerator, long limitBytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(limitBytes);
        this.accelerator = accelerator;
        this.limitBytes = limitBytes;
    }

    internal ComputeMemoryPoolStatistics Statistics
    {
        get
        {
            lock (syncRoot)
            {
                return new ComputeMemoryPoolStatistics(
                    allocatedCount,
                    rentalCount,
                    reuseCount,
                    availableByAge.Count,
                    retainedBytes,
                    limitBytes,
                    evictedCount);
            }
        }
    }

    internal Lease Rent(int length)
    {
        lock (syncRoot)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            rentalCount++;
            if (availableByLength.TryGetValue(
                    length,
                    out LinkedList<Entry>? buffers) &&
                buffers.Last is not null)
            {
                Entry reused = buffers.Last.Value;
                RemoveAvailable(reused);
                reuseCount++;
                return new Lease(this, reused);
            }
        }

        MemoryBuffer1D<float, Stride1D.Dense> buffer =
            accelerator.Allocate1D<float>(length);
        var created = new Entry(buffer);
        lock (syncRoot)
        {
            if (disposed)
            {
                buffer.Dispose();
                throw new ObjectDisposedException(GetType().Name);
            }

            live.Add(created);
            allocatedCount++;
            return new Lease(this, created);
        }
    }

    public void Dispose()
    {
        Entry[] buffers;
        lock (syncRoot)
        {
            if (disposed)
            {
                return;
            }

            disposed = true;
            buffers = live.ToArray();
            live.Clear();
            availableByLength.Clear();
            availableByAge.Clear();
            retainedBytes = 0;
        }

        foreach (Entry entry in buffers)
        {
            entry.Buffer.Dispose();
        }
    }

    private void Return(Entry entry)
    {
        List<Entry>? toDispose = null;
        lock (syncRoot)
        {
            if (disposed)
            {
                if (live.Remove(entry))
                {
                    toDispose = [entry];
                }
            }
            else
            {
                LinkedList<Entry> sameSize =
                    availableByLength.GetValueOrDefault(entry.Length) ??
                    AddSizeClass(entry.Length);
                entry.SizeNode = sameSize.AddLast(entry);
                entry.AgeNode = availableByAge.AddLast(entry);
                retainedBytes += entry.SizeBytes;

                while (retainedBytes > limitBytes &&
                       availableByAge.First is not null)
                {
                    Entry oldest = availableByAge.First.Value;
                    RemoveAvailable(oldest);
                    live.Remove(oldest);
                    evictedCount++;
                    (toDispose ??= []).Add(oldest);
                }
            }
        }

        if (toDispose is not null)
        {
            foreach (Entry buffer in toDispose)
            {
                buffer.Buffer.Dispose();
            }
        }
    }

    private LinkedList<Entry> AddSizeClass(int length)
    {
        var buffers = new LinkedList<Entry>();
        availableByLength.Add(length, buffers);
        return buffers;
    }

    private void RemoveAvailable(Entry entry)
    {
        LinkedList<Entry> sizeList =
            availableByLength[entry.Length];
        sizeList.Remove(entry.SizeNode!);
        if (sizeList.Count == 0)
        {
            availableByLength.Remove(entry.Length);
        }

        availableByAge.Remove(entry.AgeNode!);
        retainedBytes -= entry.SizeBytes;
        entry.SizeNode = null;
        entry.AgeNode = null;
    }

    internal sealed class Entry
    {
        internal Entry(MemoryBuffer1D<float, Stride1D.Dense> buffer)
        {
            Buffer = buffer;
            Length = checked((int)buffer.Length);
            SizeBytes = checked((long)Length * sizeof(float));
        }

        internal MemoryBuffer1D<float, Stride1D.Dense> Buffer { get; }

        internal int Length { get; }

        internal long SizeBytes { get; }

        internal LinkedListNode<Entry>? SizeNode { get; set; }

        internal LinkedListNode<Entry>? AgeNode { get; set; }
    }

    internal sealed class Lease : IDisposable
    {
        private GpuFloatMemoryPool? pool;
        private readonly Entry entry;

        internal Lease(GpuFloatMemoryPool pool, Entry entry)
        {
            this.pool = pool;
            this.entry = entry;
        }

        internal MemoryBuffer1D<float, Stride1D.Dense> Buffer => entry.Buffer;

        public void Dispose()
        {
            GpuFloatMemoryPool? owner =
                Interlocked.Exchange(ref pool, null);
            owner?.Return(entry);
        }
    }
}
