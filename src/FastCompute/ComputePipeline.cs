using System.Linq.Expressions;
using System.Numerics;

namespace FastCompute;

/// <summary>
/// Represents a lazy, branch-safe array computation.
/// </summary>
/// <remarks>
/// Select operations are recorded and fused into one expression when a
/// terminal operation is called. <see cref="ToArray"/> never changes the
/// source array. Use <see cref="ToArrayInPlace"/> to explicitly permit source
/// replacement.
/// </remarks>
public sealed class ComputePipeline<T>
    where T : unmanaged, INumber<T>
{
    private readonly T[] source;
    private readonly ComputeOptions options;
    private readonly ComputePipelineNode<T>? tail;
    private readonly ComputePipelineZipNode<T>? zipNode;

    internal ComputePipeline(
        T[] source,
        ComputeOptions options)
        : this(source, options, tail: null, zipNode: null)
    {
    }

    private ComputePipeline(
        T[] source,
        ComputeOptions options,
        ComputePipelineNode<T>? tail,
        ComputePipelineZipNode<T>? zipNode)
    {
        this.source = source;
        this.options = options;
        this.tail = tail;
        this.zipNode = zipNode;
    }

    /// <summary>Gets the source array length without executing the pipeline.</summary>
    public int Length => source.Length;

    /// <summary>Gets the number of recorded operations.</summary>
    public int OperationCount =>
        (zipNode?.LeftTail?.OperationCount ?? 0) +
        (zipNode is null ? 0 : 1) +
        (tail?.OperationCount ?? 0);

    /// <summary>
    /// Records a mapping operation without executing it.
    /// </summary>
    public ComputePipeline<T> Select(
        Expression<Func<T, T>> expression) =>
        Append(expression, allowBufferReuse: false);

    /// <summary>
    /// Records a mapping operation whose intermediate storage may be reused.
    /// </summary>
    /// <remarks>
    /// This is an optimizer hint and does not mutate the source array.
    /// <see cref="ToArrayInPlace"/> is the explicit source-mutating terminal.
    /// </remarks>
    public ComputePipeline<T> SelectInPlace(
        Expression<Func<T, T>> expression) =>
        Append(expression, allowBufferReuse: true);

    /// <summary>
    /// Records a binary operation with a second array without executing it.
    /// </summary>
    /// <remarks>
    /// Selectors recorded before and after this operation are fused into the
    /// resulting binary expression. A pipeline currently supports one Zip
    /// node because additional Zip nodes require more than two source arrays.
    /// </remarks>
    public ComputePipeline<T> Zip(
        T[] right,
        Expression<Func<T, T, T>> expression)
    {
        ArgumentNullException.ThrowIfNull(right);
        ArgumentNullException.ThrowIfNull(expression);
        if (zipNode is not null)
        {
            throw new NotSupportedException(
                "A lazy pipeline currently supports one Zip operation.");
        }

        return new ComputePipeline<T>(
            source,
            options,
            tail: null,
            zipNode: new ComputePipelineZipNode<T>(tail, right, expression));
    }

    /// <summary>
    /// Optimizes and executes the pipeline and returns a new managed array.
    /// </summary>
    public T[] ToArray()
    {
        if (zipNode is not null)
        {
            return ExecuteZip(
                ComputePipelineOptimizer.Optimize(zipNode, tail),
                inPlace: false);
        }

        Expression<Func<T, T>>? expression =
            ComputePipelineOptimizer.Optimize(tail);
        if (expression is null)
        {
            return source.ToArray();
        }

        return ExecuteMap(expression, inPlace: false);
    }

    /// <summary>
    /// Optimizes and executes the pipeline by replacing the source array.
    /// </summary>
    /// <returns>The same array instance that was passed to <c>AsCompute</c>.</returns>
    public T[] ToArrayInPlace()
    {
        if (zipNode is not null)
        {
            return ExecuteZip(
                ComputePipelineOptimizer.Optimize(zipNode, tail),
                inPlace: true);
        }

        Expression<Func<T, T>>? expression =
            ComputePipelineOptimizer.Optimize(tail);
        if (expression is null)
        {
            return source;
        }

        return ExecuteMap(expression, inPlace: true);
    }

    /// <summary>Executes the pipeline and computes its sum.</summary>
    public T Sum() => Reduce(ComputeReductionKind.Sum);

    /// <summary>Executes the pipeline and computes its minimum.</summary>
    public T Min() => Reduce(ComputeReductionKind.Min);

    /// <summary>Executes the pipeline and computes its maximum.</summary>
    public T Max() => Reduce(ComputeReductionKind.Max);

    /// <summary>Executes the pipeline and computes its arithmetic mean.</summary>
    public T Average() => Reduce(ComputeReductionKind.Average);

    private ComputePipeline<T> Append(
        Expression<Func<T, T>> expression,
        bool allowBufferReuse)
    {
        ArgumentNullException.ThrowIfNull(expression);
        return new ComputePipeline<T>(
            source,
            options,
            new ComputePipelineNode<T>(
                tail,
                expression,
                allowBufferReuse),
            zipNode);
    }

    private T[] ExecuteMap(
        Expression<Func<T, T>> expression,
        bool inPlace)
    {
        if (typeof(T) == typeof(float))
        {
            var floatSource = (float[])(object)source;
            var floatExpression =
                (Expression<Func<float, float>>)(object)expression;
            float[] result = inPlace
                ? Compute.RunInPlace(
                    floatSource,
                    floatExpression,
                    options)
                : Compute.Run(
                    floatSource,
                    floatExpression,
                    options);
            return (T[])(object)result;
        }

        return inPlace
            ? Compute.RunInPlace(source, expression, options)
            : Compute.Run(source, expression, options);
    }

    private T[] ExecuteZip(
        Expression<Func<T, T, T>> expression,
        bool inPlace)
    {
        T[] right = zipNode!.Right;
        if (typeof(T) == typeof(float))
        {
            var floatSource = (float[])(object)source;
            var floatRight = (float[])(object)right;
            var floatExpression =
                (Expression<Func<float, float, float>>)(object)expression;
            float[] result = inPlace
                ? Compute.ZipInPlace(
                    floatSource,
                    floatRight,
                    floatExpression,
                    options)
                : Compute.Zip(
                    floatSource,
                    floatRight,
                    floatExpression,
                    options);
            return (T[])(object)result;
        }

        return inPlace
            ? Compute.ZipInPlace(source, right, expression, options)
            : Compute.Zip(source, right, expression, options);
    }

    private T Reduce(ComputeReductionKind reduction)
    {
        if (zipNode is not null)
        {
            Expression<Func<T, T, T>> zippedExpression =
                ComputePipelineOptimizer.Optimize(zipNode, tail);
            if (typeof(T) == typeof(float))
            {
                float result = Compute.ReduceZipped(
                    (float[])(object)source,
                    (float[])(object)zipNode.Right,
                    (Expression<Func<float, float, float>>)(object)zippedExpression,
                    reduction,
                    options);
                return (T)(object)result;
            }

            return Compute.ReduceZipped(
                source,
                zipNode.Right,
                zippedExpression,
                reduction,
                options);
        }

        Expression<Func<T, T>>? expression =
            ComputePipelineOptimizer.Optimize(tail);
        if (typeof(T) == typeof(float))
        {
            var floatSource = (float[])(object)source;
            float result = expression is null
                ? ReduceFloat(floatSource, reduction)
                : Compute.ReduceMapped(
                    floatSource,
                    (Expression<Func<float, float>>)(object)expression,
                    reduction,
                    options);
            return (T)(object)result;
        }

        return expression is null
            ? ReduceNumeric(source, reduction)
            : Compute.ReduceMapped(source, expression, reduction, options);
    }

    private float ReduceFloat(
        float[] input,
        ComputeReductionKind reduction) => reduction switch
    {
        ComputeReductionKind.Sum => Compute.Sum(input, options),
        ComputeReductionKind.Min => Compute.Min(input, options),
        ComputeReductionKind.Max => Compute.Max(input, options),
        ComputeReductionKind.Average => Compute.Average(input, options),
        _ => throw new ArgumentOutOfRangeException(nameof(reduction))
    };

    private T ReduceNumeric(
        T[] input,
        ComputeReductionKind reduction) => reduction switch
        {
            ComputeReductionKind.Sum =>
                Compute.Sum(input, options),
            ComputeReductionKind.Min =>
                Compute.Min(input, options),
            ComputeReductionKind.Max =>
                Compute.Max(input, options),
            ComputeReductionKind.Average =>
                Compute.Average(input, options),
            _ => throw new ArgumentOutOfRangeException(nameof(reduction))
        };
}

internal sealed class ComputePipelineZipNode<T>
    where T : unmanaged, INumber<T>
{
    internal ComputePipelineZipNode(
        ComputePipelineNode<T>? leftTail,
        T[] right,
        Expression<Func<T, T, T>> expression)
    {
        LeftTail = leftTail;
        Right = right;
        Expression = expression;
    }

    internal ComputePipelineNode<T>? LeftTail { get; }

    internal T[] Right { get; }

    internal Expression<Func<T, T, T>> Expression { get; }
}

internal sealed class ComputePipelineNode<T>
    where T : unmanaged, INumber<T>
{
    internal ComputePipelineNode(
        ComputePipelineNode<T>? previous,
        Expression<Func<T, T>> expression,
        bool allowBufferReuse)
    {
        Previous = previous;
        Expression = expression;
        AllowBufferReuse = allowBufferReuse;
        OperationCount = checked((previous?.OperationCount ?? 0) + 1);
    }

    internal ComputePipelineNode<T>? Previous { get; }

    internal Expression<Func<T, T>> Expression { get; }

    internal bool AllowBufferReuse { get; }

    internal int OperationCount { get; }
}

internal static class ComputePipelineOptimizer
{
    internal static Expression<Func<T, T>>? Optimize<T>(
        ComputePipelineNode<T>? tail)
        where T : unmanaged, INumber<T>
    {
        if (tail is null)
        {
            return null;
        }

        var nodes = new ComputePipelineNode<T>[tail.OperationCount];
        ComputePipelineNode<T>? current = tail;
        for (int index = nodes.Length - 1; index >= 0; index--)
        {
            nodes[index] = current!;
            current = current!.Previous;
        }

        var parameter = Expression.Parameter(typeof(T), "value");
        Expression body = parameter;
        foreach (ComputePipelineNode<T> node in nodes)
        {
            body = new PipelineParameterReplacer(
                    node.Expression.Parameters[0],
                    body)
                .Visit(node.Expression.Body)!;
        }

        return Expression.Lambda<Func<T, T>>(body, parameter);
    }

    internal static Expression<Func<T, T, T>> Optimize<T>(
        ComputePipelineZipNode<T> zipNode,
        ComputePipelineNode<T>? tail)
        where T : unmanaged, INumber<T>
    {
        var left = Expression.Parameter(typeof(T), "left");
        var right = Expression.Parameter(typeof(T), "right");
        Expression leftBody = Apply(zipNode.LeftTail, left);
        Expression body = new PipelineParameterReplacer(
                zipNode.Expression.Parameters[0],
                leftBody)
            .Visit(zipNode.Expression.Body)!;
        body = new PipelineParameterReplacer(
                zipNode.Expression.Parameters[1],
                right)
            .Visit(body)!;
        body = Apply(tail, body);
        return Expression.Lambda<Func<T, T, T>>(body, left, right);
    }

    private static Expression Apply<T>(
        ComputePipelineNode<T>? tail,
        Expression body)
        where T : unmanaged, INumber<T>
    {
        if (tail is null)
        {
            return body;
        }

        var nodes = new ComputePipelineNode<T>[tail.OperationCount];
        ComputePipelineNode<T>? current = tail;
        for (int index = nodes.Length - 1; index >= 0; index--)
        {
            nodes[index] = current!;
            current = current!.Previous;
        }

        foreach (ComputePipelineNode<T> node in nodes)
        {
            body = new PipelineParameterReplacer(
                    node.Expression.Parameters[0],
                    body)
                .Visit(node.Expression.Body)!;
        }

        return body;
    }

    private sealed class PipelineParameterReplacer(
        ParameterExpression parameter,
        Expression replacement) : ExpressionVisitor
    {
        protected override Expression VisitParameter(
            ParameterExpression node) =>
            node == parameter
                ? replacement
                : base.VisitParameter(node);
    }
}
