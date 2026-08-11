using System.Linq.Expressions;
using FastCompute.Backends;

namespace FastCompute;

/// <summary>
/// Represents a lazy computation over an unmanaged composite value.
/// </summary>
/// <typeparam name="T">The composite element type.</typeparam>
public sealed class ComputeValuePipeline<T>
    where T : unmanaged, IComputeValue<T>
{
    private readonly T[] source;
    private readonly ComputeOptions options;
    private readonly IReadOnlyList<Expression<Func<T, T>>> operations;

    internal ComputeValuePipeline(T[] source, ComputeOptions options)
        : this(source, options, Array.Empty<Expression<Func<T, T>>>())
    {
    }

    private ComputeValuePipeline(
        T[] source,
        ComputeOptions options,
        IReadOnlyList<Expression<Func<T, T>>> operations)
    {
        this.source = source;
        this.options = options;
        this.operations = operations;
    }

    /// <summary>Gets the source element count.</summary>
    public int Length => source.Length;

    /// <summary>Gets the number of recorded operations.</summary>
    public int OperationCount => operations.Count;

    /// <summary>Records a composite value transformation.</summary>
    public ComputeValuePipeline<T> Select(Expression<Func<T, T>> expression) =>
        Append(expression);

    /// <summary>
    /// Records a composite value transformation whose terminal operation may
    /// reuse the source buffer.
    /// </summary>
    public ComputeValuePipeline<T> SelectInPlace(
        Expression<Func<T, T>> expression) => Append(expression);

    /// <summary>
    /// Records a projection from the composite value to a floating-point value.
    /// </summary>
    public ComputeValueProjectionPipeline<T> Select(
        Expression<Func<T, float>> expression)
    {
        ArgumentNullException.ThrowIfNull(expression);
        return new ComputeValueProjectionPipeline<T>(
            source,
            options,
            ComposeProjection(expression));
    }

    /// <summary>
    /// Records a transformation into another registered unmanaged value type.
    /// Homogeneous float or byte layouts use native backends. Mixed physical
    /// component types currently execute on Scalar or ParallelCpu.
    /// </summary>
    public ComputeValueConversionPipeline<T, TDestination>
        Select<TDestination>(Expression<Func<T, TDestination>> expression)
        where TDestination : unmanaged, IComputeValue<TDestination>
    {
        ArgumentNullException.ThrowIfNull(expression);
        var parameter = Expression.Parameter(typeof(T), "value");
        Expression input = ApplyOperations(parameter);
        Expression body = new ParameterReplacer(
            expression.Parameters[0],
            input).Visit(expression.Body)!;
        return new ComputeValueConversionPipeline<T, TDestination>(
            source,
            options,
            Expression.Lambda<Func<T, TDestination>>(body, parameter));
    }

    /// <summary>Executes the pipeline into a new array.</summary>
    public T[] ToArray()
    {
        Expression<Func<T, T>>? expression = ComposeOperations();
        return expression is null
            ? source.ToArray()
            : ComputeValueExecutor.Map(source, expression, options, inPlace: false);
    }

    /// <summary>Executes the pipeline by overwriting and returning the source array.</summary>
    public T[] ToArrayInPlace()
    {
        Expression<Func<T, T>>? expression = ComposeOperations();
        return expression is null
            ? source
            : ComputeValueExecutor.Map(source, expression, options, inPlace: true);
    }

    private ComputeValuePipeline<T> Append(Expression<Func<T, T>> expression)
    {
        ArgumentNullException.ThrowIfNull(expression);
        return new ComputeValuePipeline<T>(
            source,
            options,
            [.. operations, expression]);
    }

    private Expression<Func<T, float>> ComposeProjection(
        Expression<Func<T, float>> projection)
    {
        var parameter = Expression.Parameter(typeof(T), "value");
        Expression body = ApplyOperations(parameter);
        body = new ParameterReplacer(projection.Parameters[0], body)
            .Visit(projection.Body)!;
        return Expression.Lambda<Func<T, float>>(body, parameter);
    }

    private Expression<Func<T, T>>? ComposeOperations()
    {
        if (operations.Count == 0)
        {
            return null;
        }

        var parameter = Expression.Parameter(typeof(T), "value");
        return Expression.Lambda<Func<T, T>>(
            ApplyOperations(parameter),
            parameter);
    }

    private Expression ApplyOperations(Expression input)
    {
        Expression body = input;
        foreach (Expression<Func<T, T>> operation in operations)
        {
            body = new ParameterReplacer(operation.Parameters[0], body)
                .Visit(operation.Body)!;
        }

        return body;
    }

    private sealed class ParameterReplacer(
        ParameterExpression parameter,
        Expression replacement) : ExpressionVisitor
    {
        protected override Expression VisitParameter(ParameterExpression node) =>
            node == parameter ? replacement : base.VisitParameter(node);
    }
}

/// <summary>Represents a lazy conversion between registered compute values.</summary>
public sealed class ComputeValueConversionPipeline<TSource, TDestination>
    where TSource : unmanaged, IComputeValue<TSource>
    where TDestination : unmanaged, IComputeValue<TDestination>
{
    private readonly TSource[] source;
    private readonly ComputeOptions options;
    private readonly Expression<Func<TSource, TDestination>> expression;

    internal ComputeValueConversionPipeline(
        TSource[] source,
        ComputeOptions options,
        Expression<Func<TSource, TDestination>> expression)
    {
        this.source = source;
        this.options = options;
        this.expression = expression;
    }

    /// <summary>Gets the source element count.</summary>
    public int Length => source.Length;

    /// <summary>Executes the conversion into a new array.</summary>
    public TDestination[] ToArray() =>
        ComputeValueConversionExecutor.Transform(source, expression, options);
}

/// <summary>
/// Represents a lazy projection from a composite value to a floating-point
/// array.
/// </summary>
/// <typeparam name="T">The composite source type.</typeparam>
public sealed class ComputeValueProjectionPipeline<T>
    where T : unmanaged, IComputeValue<T>
{
    private readonly T[] source;
    private readonly ComputeOptions options;
    private readonly Expression<Func<T, float>> expression;

    internal ComputeValueProjectionPipeline(
        T[] source,
        ComputeOptions options,
        Expression<Func<T, float>> expression)
    {
        this.source = source;
        this.options = options;
        this.expression = expression;
    }

    /// <summary>Gets the source element count.</summary>
    public int Length => source.Length;

    /// <summary>Executes the projection and returns its results.</summary>
    public float[] ToArray() =>
        ComputeValueExecutor.Project(source, expression, options);
}

/// <summary>
/// Creates lazy compute pipelines for registered unmanaged composite values.
/// </summary>
public static class ComputeValueArrayExtensions
{
    /// <summary>Creates an automatically planned composite value pipeline.</summary>
    public static ComputeValuePipeline<T> AsCompute<T>(this T[] source)
        where T : unmanaged, IComputeValue<T>
    {
        ArgumentNullException.ThrowIfNull(source);
        return new ComputeValuePipeline<T>(source, ComputeOptions.Default);
    }

    /// <summary>Creates a composite value pipeline with explicit options.</summary>
    public static ComputeValuePipeline<T> AsCompute<T>(
        this T[] source,
        ComputeOptions options)
        where T : unmanaged, IComputeValue<T>
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(options);
        return new ComputeValuePipeline<T>(source, options);
    }
}
