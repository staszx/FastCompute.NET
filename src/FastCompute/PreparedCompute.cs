using FastCompute.Gpu;

namespace FastCompute;

/// <summary>
/// Represents a planned and compiled reusable GPU map operation.
/// </summary>
public sealed class PreparedCompute<T>
    where T : unmanaged
{
    private readonly ComputeContext context;
    private readonly GpuProgram program;
    private readonly ComputeContext.CompiledKernel kernel;

    internal PreparedCompute(
        ComputeContext context,
        GpuProgram program,
        ComputeContext.CompiledKernel kernel)
    {
        this.context = context;
        this.program = program;
        this.kernel = kernel;
    }

    /// <summary>Runs the prepared operation.</summary>
    public T[] Run(T[] source)
    {
        ArgumentNullException.ThrowIfNull(source);
        context.ThrowIfDisposed();

        if (typeof(T) != typeof(float))
        {
            throw new NotSupportedException(
                $"GPU execution currently supports float, not '{typeof(T).Name}'.");
        }

        float[] result = context.ExecuteMap(
            (float[])(object)source,
            program,
            kernel,
            CancellationToken.None);
        return (T[])(object)result;
    }
}
