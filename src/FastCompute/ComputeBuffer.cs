using System.Linq.Expressions;
using ILGPU;
using ILGPU.Runtime;

namespace FastCompute;

/// <summary>
/// Owns an array stored in memory associated with a <see cref="ComputeContext"/>.
/// </summary>
public sealed class ComputeBuffer<T> : IDisposable
    where T : unmanaged
{
    private readonly ComputeContext context;
    private MemoryBuffer1D<T, Stride1D.Dense>? buffer;

    internal ComputeBuffer(
        ComputeContext context,
        MemoryBuffer1D<T, Stride1D.Dense> buffer)
    {
        this.context = context;
        this.buffer = buffer;
    }

    /// <summary>Gets the number of elements in the buffer.</summary>
    public int Length => checked((int)GetBuffer().Length);

    /// <summary>Downloads the buffer to a new managed array.</summary>
    public T[] Download()
    {
        context.ThrowIfDisposed();
        return GetBuffer().GetAsArray1D();
    }

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

    /// <inheritdoc />
    public void Dispose()
    {
        MemoryBuffer1D<T, Stride1D.Dense>? owned =
            Interlocked.Exchange(ref buffer, null);
        owned?.Dispose();
    }

    internal ComputeContext Context => context;

    internal MemoryBuffer1D<T, Stride1D.Dense> GetBuffer() =>
        buffer ?? throw new ObjectDisposedException(GetType().Name);
}
