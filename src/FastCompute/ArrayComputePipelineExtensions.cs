using System.Numerics;

namespace FastCompute;

/// <summary>
/// Creates lazy compute pipelines from supported numeric arrays.
/// </summary>
public static class ArrayComputePipelineExtensions
{
    /// <summary>
    /// Creates a lazy pipeline that selects its backend automatically.
    /// </summary>
    public static ComputePipeline<T> AsCompute<T>(this T[] source)
        where T : unmanaged, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(source);
        return new ComputePipeline<T>(
            source,
            ComputeOptions.Default);
    }

    /// <summary>
    /// Creates a lazy pipeline with operation-level execution settings.
    /// </summary>
    public static ComputePipeline<T> AsCompute<T>(
        this T[] source,
        ComputeOptions options)
        where T : unmanaged, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(options);
        return new ComputePipeline<T>(source, options);
    }

    /// <summary>
    /// Creates a lazy Auto pipeline that can reuse the supplied GPU context
    /// when GPU execution is selected.
    /// </summary>
    public static ComputePipeline<T> AsCompute<T>(
        this T[] source,
        ComputeContext context)
        where T : unmanaged, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(context);
        return new ComputePipeline<T>(
            source,
            new ComputeOptions
            {
                Backend = ComputeBackendKind.Auto,
                GpuContext = context
            });
    }
}
