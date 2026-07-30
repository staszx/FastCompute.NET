using System.Linq.Expressions;

namespace FastCompute;

/// <summary>
/// Provides LINQ-style explicit compute operations for double and integer
/// arrays.
/// </summary>
public static class NumericArrayComputeExtensions
{
    /// <summary>Runs a double expression on the explicitly selected backend.</summary>
    public static double[] RunExplicit(
        this double[] source,
        Expression<Func<double, double>> expression,
        ComputeBackendKind backend) =>
        RunExplicit(
            source,
            expression,
            new ComputeOptions { Backend = backend });

    /// <summary>Runs a double expression with explicit backend options.</summary>
    public static double[] RunExplicit(
        this double[] source,
        Expression<Func<double, double>> expression,
        ComputeOptions options)
    {
        ValidateArguments(source, expression, options);
        return Compute.Run(source, expression, options);
    }

    /// <summary>Runs an integer expression on the explicitly selected backend.</summary>
    public static int[] RunExplicit(
        this int[] source,
        Expression<Func<int, int>> expression,
        ComputeBackendKind backend) =>
        RunExplicit(
            source,
            expression,
            new ComputeOptions { Backend = backend });

    /// <summary>Runs an integer expression with explicit backend options.</summary>
    public static int[] RunExplicit(
        this int[] source,
        Expression<Func<int, int>> expression,
        ComputeOptions options)
    {
        ValidateArguments(source, expression, options);
        return Compute.Run(source, expression, options);
    }

    private static void ValidateArguments<T>(
        T[] source,
        Expression<Func<T, T>> expression,
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
