using System.Linq.Expressions;

namespace FastCompute;

/// <summary>
/// Creates descriptors for batch GPU precompilation.
/// </summary>
public static class ComputeKernel
{
    /// <summary>Creates a map-kernel descriptor.</summary>
    public static ComputeKernelDescriptor Map<T>(Expression<Func<T, T>> expression)
        where T : unmanaged =>
        new(expression, ComputeKernelKind.Map, typeof(T));

    /// <summary>Creates a zip-kernel descriptor.</summary>
    public static ComputeKernelDescriptor Zip<T>(Expression<Func<T, T, T>> expression)
        where T : unmanaged =>
        new(expression, ComputeKernelKind.Zip, typeof(T));

    /// <summary>Creates a reduction-kernel descriptor.</summary>
    public static ComputeKernelDescriptor Reduction<T>(ComputeReductionKind reduction)
        where T : unmanaged =>
        new(reduction, typeof(T));
}

/// <summary>
/// Describes an expression to prepare with <see cref="ComputeContext.Precompile"/>.
/// </summary>
public sealed class ComputeKernelDescriptor
{
    internal ComputeKernelDescriptor(
        LambdaExpression expression,
        ComputeKernelKind kind,
        Type elementType)
    {
        Expression = expression;
        Kind = kind;
        ElementType = elementType;
    }

    internal ComputeKernelDescriptor(
        ComputeReductionKind reduction,
        Type elementType)
    {
        Reduction = reduction;
        Kind = ComputeKernelKind.Reduction;
        ElementType = elementType;
    }

    internal LambdaExpression? Expression { get; }

    internal ComputeKernelKind Kind { get; }

    internal Type ElementType { get; }

    internal ComputeReductionKind? Reduction { get; }
}

internal enum ComputeKernelKind
{
    Map,
    Zip,
    Reduction
}
