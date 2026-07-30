using FastCompute.Gpu;

namespace FastCompute;

/// <summary>
/// Represents a planned and compiled reusable GPU map operation.
/// </summary>
public sealed class PreparedCompute<T>
    where T : unmanaged
{
    private readonly ComputeContext context;
    private readonly Func<T[], T[]> runner;

    internal PreparedCompute(
        ComputeContext context,
        GpuProgram program,
        ComputeContext.CompiledKernel kernel)
    {
        this.context = context;
        runner = source =>
        {
            float[] result = context.ExecuteMap(
                (float[])(object)source,
                program,
                kernel,
                CancellationToken.None);
            return (T[])(object)result;
        };
    }

    internal PreparedCompute(
        ComputeContext context,
        Func<T[], T[]> runner)
    {
        this.context = context;
        this.runner = runner;
    }

    /// <summary>Runs the prepared operation.</summary>
    public T[] Run(T[] source)
    {
        ArgumentNullException.ThrowIfNull(source);
        context.ThrowIfDisposed();

        return runner(source);
    }
}
