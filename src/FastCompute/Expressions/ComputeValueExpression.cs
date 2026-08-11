using System.Linq.Expressions;
using System.Reflection;

namespace FastCompute.Expressions;

internal enum ComputeValueBinaryOperation
{
    Add,
    Subtract,
    Multiply,
    Divide
}

internal abstract record ComputeValueExpressionNode;

internal sealed record ComputeValueComponentNode(int Index)
    : ComputeValueExpressionNode;

internal sealed record ComputeValueConstantNode(float Value)
    : ComputeValueExpressionNode;

internal sealed record ComputeValueNegateNode(ComputeValueExpressionNode Operand)
    : ComputeValueExpressionNode;

internal sealed record ComputeValueBinaryNode(
    ComputeValueBinaryOperation Operation,
    ComputeValueExpressionNode Left,
    ComputeValueExpressionNode Right)
    : ComputeValueExpressionNode;

internal sealed record ComputeValueExpressionProgram(
    IReadOnlyList<ComputeValueExpressionNode> Outputs);

internal static class ComputeValueExpressionParser
{
    internal static ComputeValueExpressionProgram ParseProjection<T>(
        Expression<Func<T, float>> expression,
        ComputeValueDescriptor<T> descriptor)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(expression);
        return new ComputeValueExpressionProgram(
            [ParseScalar(expression.Body, expression.Parameters[0], descriptor)]);
    }

    internal static ComputeValueExpressionProgram ParseMap<T>(
        Expression<Func<T, T>> expression,
        ComputeValueDescriptor<T> descriptor)
        where T : unmanaged =>
        ParseMap(expression, descriptor, descriptor);

    internal static ComputeValueExpressionProgram ParseMap<TSource, TDestination>(
        Expression<Func<TSource, TDestination>> expression,
        ComputeValueDescriptor<TSource> sourceDescriptor,
        ComputeValueDescriptor<TDestination> destinationDescriptor)
        where TSource : unmanaged
        where TDestination : unmanaged
    {
        ArgumentNullException.ThrowIfNull(expression);
        Expression body = expression.Body;
        if (typeof(TSource) == typeof(TDestination) &&
            body == expression.Parameters[0])
        {
            return new ComputeValueExpressionProgram(
                Enumerable.Range(0, sourceDescriptor.ComponentCount)
                    .Select(index =>
                        (ComputeValueExpressionNode)new ComputeValueComponentNode(index))
                    .ToArray());
        }

        if (body is NewExpression construction &&
            construction.Arguments.Count == destinationDescriptor.ComponentCount)
        {
            return new ComputeValueExpressionProgram(
                construction.Arguments
                    .Select(argument =>
                        ParseScalar(
                            argument,
                            expression.Parameters[0],
                            sourceDescriptor))
                    .ToArray());
        }

        if (body is MemberInitExpression initializer)
        {
            var outputs =
                new ComputeValueExpressionNode?[destinationDescriptor.ComponentCount];
            foreach (MemberBinding binding in initializer.Bindings)
            {
                if (binding is not MemberAssignment assignment)
                {
                    throw Unsupported(
                        body,
                        "Only direct component assignments are supported.");
                }

                int index = destinationDescriptor.GetComponentIndex(
                    assignment.Member);
                outputs[index] = ParseScalar(
                    assignment.Expression,
                    expression.Parameters[0],
                    sourceDescriptor);
            }

            if (outputs.Any(output => output is null))
            {
                throw Unsupported(
                    body,
                    "A composite initializer must assign every registered component.");
            }

            return new ComputeValueExpressionProgram(outputs!);
        }

        throw Unsupported(
            body,
            "Composite results must use the registered constructor order or assign every component.");
    }

    private static ComputeValueExpressionNode ParseScalar<T>(
        Expression expression,
        ParameterExpression parameter,
        ComputeValueDescriptor<T> descriptor)
        where T : unmanaged
    {
        if (expression.Type != typeof(float))
        {
            throw Unsupported(expression, "Composite compute operations currently use Single components.");
        }

        if (expression is MemberExpression member && member.Expression == parameter)
        {
            return new ComputeValueComponentNode(
                descriptor.GetComponentIndex(member.Member));
        }

        if (expression is MemberExpression projectedMember &&
            projectedMember.Expression is NewExpression construction &&
            construction.Arguments.Count == descriptor.ComponentCount)
        {
            int componentIndex = descriptor.GetComponentIndex(projectedMember.Member);
            return ParseScalar(
                construction.Arguments[componentIndex],
                parameter,
                descriptor);
        }

        if (expression is MemberExpression initializedMember &&
            initializedMember.Expression is MemberInitExpression initializer)
        {
            int componentIndex = descriptor.GetComponentIndex(initializedMember.Member);
            MemberInfo componentMember = descriptor.Components[componentIndex];
            MemberAssignment? assignment = initializer.Bindings
                .OfType<MemberAssignment>()
                .SingleOrDefault(binding => binding.Member == componentMember);
            if (assignment is not null)
            {
                return ParseScalar(assignment.Expression, parameter, descriptor);
            }
        }

        if (TryReadConstant(expression, out float constant))
        {
            return new ComputeValueConstantNode(constant);
        }

        return expression switch
        {
            BinaryExpression binary => ParseBinary(binary, parameter, descriptor),
            UnaryExpression { NodeType: ExpressionType.Negate } unary =>
                new ComputeValueNegateNode(
                    ParseScalar(unary.Operand, parameter, descriptor)),
            _ => throw Unsupported(
                expression,
                "Use registered components, constants, and arithmetic operators.")
        };
    }

    private static ComputeValueExpressionNode ParseBinary<T>(
        BinaryExpression expression,
        ParameterExpression parameter,
        ComputeValueDescriptor<T> descriptor)
        where T : unmanaged
    {
        ComputeValueBinaryOperation operation = expression.NodeType switch
        {
            ExpressionType.Add => ComputeValueBinaryOperation.Add,
            ExpressionType.Subtract => ComputeValueBinaryOperation.Subtract,
            ExpressionType.Multiply => ComputeValueBinaryOperation.Multiply,
            ExpressionType.Divide => ComputeValueBinaryOperation.Divide,
            _ => throw Unsupported(
                expression,
                $"Binary operator '{expression.NodeType}' is not supported.")
        };

        return new ComputeValueBinaryNode(
            operation,
            ParseScalar(expression.Left, parameter, descriptor),
            ParseScalar(expression.Right, parameter, descriptor));
    }

    private static bool TryReadConstant(Expression expression, out float value)
    {
        if (expression is ConstantExpression { Value: float direct })
        {
            value = direct;
            return true;
        }

        if (expression is MemberExpression member &&
            member.Expression is ConstantExpression)
        {
            try
            {
                value = Expression.Lambda<Func<float>>(member).Compile()();
                return true;
            }
            catch (Exception exception) when (
                exception is ArgumentException or InvalidOperationException)
            {
                // Report the original expression below.
            }
        }

        value = default;
        return false;
    }

    private static NotSupportedException Unsupported(
        Expression expression,
        string description) =>
        new(
            $"The composite compute expression '{expression}' is not supported. " +
            description);
}
