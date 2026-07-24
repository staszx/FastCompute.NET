using FastCompute.Expressions;
using ILGPU;
using ILGPU.Runtime;

namespace FastCompute.Gpu;

internal abstract class ComputeBufferNode<T>
    where T : unmanaged
{
    private readonly object materializationLock = new();
    private MemoryBuffer1D<T, Stride1D.Dense>? materializedBuffer;
    private int referenceCount = 1;

    protected ComputeBufferNode(
        ComputeContext context,
        int length,
        MemoryBuffer1D<T, Stride1D.Dense>? materializedBuffer = null)
    {
        Context = context;
        Length = length;
        this.materializedBuffer = materializedBuffer;
    }

    internal ComputeContext Context { get; }

    internal int Length { get; }

    internal void Acquire()
    {
        while (true)
        {
            int current = Volatile.Read(ref referenceCount);
            if (current == 0)
            {
                throw new ObjectDisposedException(nameof(ComputeBuffer<T>));
            }

            if (Interlocked.CompareExchange(
                    ref referenceCount,
                    current + 1,
                    current) == current)
            {
                return;
            }
        }
    }

    internal void Release()
    {
        int remaining = Interlocked.Decrement(ref referenceCount);
        if (remaining > 0)
        {
            return;
        }

        if (remaining < 0)
        {
            throw new InvalidOperationException(
                "A compute-buffer graph node was released more than once.");
        }

        MemoryBuffer1D<T, Stride1D.Dense>? buffer;
        lock (materializationLock)
        {
            buffer = materializedBuffer;
            materializedBuffer = null;
            ReleaseDependencies();
        }

        buffer?.Dispose();
    }

    internal MemoryBuffer1D<T, Stride1D.Dense> GetBuffer()
    {
        Context.ThrowIfDisposed();

        lock (materializationLock)
        {
            if (Volatile.Read(ref referenceCount) == 0)
            {
                throw new ObjectDisposedException(nameof(ComputeBuffer<T>));
            }

            materializedBuffer ??= Materialize();
            return materializedBuffer;
        }
    }

    protected abstract MemoryBuffer1D<T, Stride1D.Dense> Materialize();

    protected virtual void ReleaseDependencies()
    {
    }
}

internal sealed class BufferSourceNode<T> : ComputeBufferNode<T>
    where T : unmanaged
{
    internal BufferSourceNode(
        ComputeContext context,
        int length)
        : base(context, length)
    {
    }

    internal BufferSourceNode(
        ComputeContext context,
        MemoryBuffer1D<T, Stride1D.Dense> buffer)
        : base(context, checked((int)buffer.Length), buffer)
    {
    }

    protected override MemoryBuffer1D<T, Stride1D.Dense> Materialize() =>
        throw new InvalidOperationException(
            "A source buffer must always be materialized.");
}

internal sealed class MapBufferNode<T> : ComputeBufferNode<T>
    where T : unmanaged
{
    private ComputeBufferNode<T>? source;
    private readonly ComputeExpressionPlan plan;

    internal MapBufferNode(
        ComputeContext context,
        ComputeBufferNode<T> source,
        ComputeExpressionPlan plan)
        : base(context, source.Length)
    {
        this.source = source;
        this.plan = plan;
    }

    protected override MemoryBuffer1D<T, Stride1D.Dense> Materialize()
    {
        ComputeBufferNode<T> dependency =
            source ??
            throw new ObjectDisposedException(nameof(ComputeBuffer<T>));
        MemoryBuffer1D<T, Stride1D.Dense> sourceBuffer =
            dependency.GetBuffer();
        MemoryBuffer1D<T, Stride1D.Dense> result =
            Context.ExecuteGraphMap(sourceBuffer, Length, plan);

        Interlocked.Exchange(ref source, null)?.Release();
        return result;
    }

    protected override void ReleaseDependencies() =>
        Interlocked.Exchange(ref source, null)?.Release();
}

internal sealed class ZipBufferNode<T> : ComputeBufferNode<T>
    where T : unmanaged
{
    private ComputeBufferNode<T>? left;
    private ComputeBufferNode<T>? right;
    private readonly ComputeExpressionPlan plan;

    internal ZipBufferNode(
        ComputeContext context,
        ComputeBufferNode<T> left,
        ComputeBufferNode<T> right,
        ComputeExpressionPlan plan)
        : base(context, left.Length)
    {
        this.left = left;
        this.right = right;
        this.plan = plan;
    }

    protected override MemoryBuffer1D<T, Stride1D.Dense> Materialize()
    {
        ComputeBufferNode<T> leftDependency =
            left ??
            throw new ObjectDisposedException(nameof(ComputeBuffer<T>));
        ComputeBufferNode<T> rightDependency =
            right ??
            throw new ObjectDisposedException(nameof(ComputeBuffer<T>));

        MemoryBuffer1D<T, Stride1D.Dense> result =
            Context.ExecuteGraphZip(
                leftDependency.GetBuffer(),
                rightDependency.GetBuffer(),
                Length,
                plan);

        Interlocked.Exchange(ref left, null)?.Release();
        Interlocked.Exchange(ref right, null)?.Release();
        return result;
    }

    protected override void ReleaseDependencies()
    {
        Interlocked.Exchange(ref left, null)?.Release();
        Interlocked.Exchange(ref right, null)?.Release();
    }
}
