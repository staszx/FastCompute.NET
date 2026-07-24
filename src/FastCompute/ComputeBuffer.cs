using System.Linq.Expressions;
using FastCompute.Gpu;
using ILGPU.Runtime;

namespace FastCompute;

/// <summary>
/// Owns an array stored in memory associated with a <see cref="ComputeContext"/>.
/// </summary>
public sealed class ComputeBuffer<T> : IDisposable
    where T : unmanaged
{
    private readonly ComputeContext context;
    private ComputeBufferNode<T>? node;

    internal ComputeBuffer(
        ComputeContext context,
        ComputeBufferNode<T> node)
    {
        this.context = context;
        this.node = node;
    }

    /// <summary>Releases an undisposed buffer handle.</summary>
    ~ComputeBuffer()
    {
        ReleaseNode();
    }

    /// <summary>Gets the number of elements in the buffer.</summary>
    public int Length
    {
        get
        {
            ComputeBufferNode<T> current = AcquireNode();
            try
            {
                return current.Length;
            }
            finally
            {
                current.Release();
            }
        }
    }

    /// <summary>Downloads the buffer to a new managed array.</summary>
    public T[] Download()
    {
        context.ThrowIfDisposed();
        ComputeBufferNode<T> current = AcquireNode();
        try
        {
            if (current.Length == 0)
            {
                return [];
            }

            return current.GetBuffer().GetAsArray1D();
        }
        finally
        {
            current.Release();
        }
    }

    /// <summary>Downloads the buffer into an existing destination span.</summary>
    /// <param name="destination">
    /// The destination whose length must be at least <see cref="Length"/>.
    /// </param>
    /// <exception cref="ArgumentException">The destination is too short.</exception>
    public void Download(Span<T> destination) =>
        context.Download(this, destination);

    /// <summary>Runs a map expression and keeps its result on the accelerator.</summary>
    public ComputeBuffer<T> Select(Expression<Func<T, T>> expression)
    {
        ArgumentNullException.ThrowIfNull(expression);
        return context.Select(this, expression);
    }

    /// <summary>
    /// Runs a zip expression against another buffer and keeps its result on the accelerator.
    /// </summary>
    public ComputeBuffer<T> Zip(
        ComputeBuffer<T> right,
        Expression<Func<T, T, T>> expression)
    {
        ArgumentNullException.ThrowIfNull(right);
        ArgumentNullException.ThrowIfNull(expression);
        return context.Zip(this, right, expression);
    }

    /// <summary>Computes the sum while keeping the input on the accelerator.</summary>
    public T Sum() => context.Reduce(this, ComputeReductionKind.Sum);

    /// <summary>Computes the minimum while keeping the input on the accelerator.</summary>
    /// <exception cref="InvalidOperationException">The buffer is empty.</exception>
    public T Min() => context.Reduce(this, ComputeReductionKind.Min);

    /// <summary>Computes the maximum while keeping the input on the accelerator.</summary>
    /// <exception cref="InvalidOperationException">The buffer is empty.</exception>
    public T Max() => context.Reduce(this, ComputeReductionKind.Max);

    /// <summary>Computes the arithmetic mean while keeping the input on the accelerator.</summary>
    /// <exception cref="InvalidOperationException">The buffer is empty.</exception>
    public T Average() => context.Reduce(this, ComputeReductionKind.Average);

    /// <inheritdoc />
    public void Dispose()
    {
        ReleaseNode();
        GC.SuppressFinalize(this);
    }

    private void ReleaseNode()
    {
        ComputeBufferNode<T>? owned =
            Interlocked.Exchange(ref node, null);
        owned?.Release();
    }

    internal ComputeContext Context => context;

    internal ComputeBufferNode<T> AcquireNode()
    {
        ComputeBufferNode<T> current =
            Volatile.Read(ref node) ??
            throw new ObjectDisposedException(GetType().Name);
        current.Acquire();
        return current;
    }
}
