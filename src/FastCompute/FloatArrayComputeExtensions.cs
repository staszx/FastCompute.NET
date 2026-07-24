using System.Linq.Expressions;
using FastCompute.Diagnostics;

namespace FastCompute;

/// <summary>
/// Provides LINQ-style explicit compute operations for <see cref="float"/>
/// arrays.
/// </summary>
public static class FloatArrayComputeExtensions
{
    /// <summary>
    /// Applies an expression using the explicitly selected backend.
    /// </summary>
    /// <param name="source">The input array.</param>
    /// <param name="expression">The expression applied to every element.</param>
    /// <param name="backend">The backend that must execute the operation.</param>
    /// <returns>A new array containing the computed values.</returns>
    /// <exception cref="ArgumentException">
    /// <paramref name="backend"/> is <see cref="ComputeBackendKind.Auto"/>.
    /// </exception>
    public static float[] RunExplicit(
        this float[] source,
        Expression<Func<float, float>> expression,
        ComputeBackendKind backend) =>
        RunExplicit(
            source,
            expression,
            new ComputeOptions { Backend = backend });

    /// <summary>
    /// Applies an expression using the explicitly configured backend.
    /// </summary>
    /// <remarks>
    /// Automatic backend selection and automatic fallback are not used. GPU and
    /// SIMD expressions must still belong to their supported compute IR subset.
    /// </remarks>
    /// <param name="source">The input array.</param>
    /// <param name="expression">The expression applied to every element.</param>
    /// <param name="options">
    /// Execution settings containing a backend other than
    /// <see cref="ComputeBackendKind.Auto"/>.
    /// </param>
    /// <returns>A new array containing the computed values.</returns>
    /// <exception cref="ArgumentException">
    /// <see cref="ComputeOptions.Backend"/> is
    /// <see cref="ComputeBackendKind.Auto"/>.
    /// </exception>
    public static float[] RunExplicit(
        this float[] source,
        Expression<Func<float, float>> expression,
        ComputeOptions options)
    {
        ValidateArguments(source, expression, options);
        return Compute.Run(source, expression, options);
    }

    /// <summary>
    /// Applies an expression using the explicitly configured backend and
    /// returns execution diagnostics.
    /// </summary>
    /// <param name="source">The input array.</param>
    /// <param name="expression">The expression applied to every element.</param>
    /// <param name="options">
    /// Execution settings containing a backend other than
    /// <see cref="ComputeBackendKind.Auto"/>.
    /// </param>
    /// <returns>The computed array and collected diagnostics.</returns>
    public static ComputeResult<float[]> RunExplicitWithDiagnostics(
        this float[] source,
        Expression<Func<float, float>> expression,
        ComputeOptions options)
    {
        ValidateArguments(source, expression, options);
        return Compute.RunWithDiagnostics(source, expression, options);
    }

    private static void ValidateArguments(
        float[] source,
        Expression<Func<float, float>> expression,
        ComputeOptions options)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(expression);
        ArgumentNullException.ThrowIfNull(options);

        if (options.Backend == ComputeBackendKind.Auto)
        {
            throw new ArgumentException(
                $"{nameof(RunExplicit)} requires Scalar, ParallelCpu, Simd, " +
                "or Gpu. Auto selection is intentionally unavailable.",
                nameof(options));
        }
    }
}
