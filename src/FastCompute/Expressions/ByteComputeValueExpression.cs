using System.Linq.Expressions;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace FastCompute.Expressions;

internal enum ByteComputeOperation
{
    Add,
    Subtract,
    Multiply,
    Divide
}

internal abstract record ByteComputeNode;
internal sealed record ByteComponentNode(int Index) : ByteComputeNode;
internal sealed record ByteConstantNode(int Value) : ByteComputeNode;
internal sealed record ByteNegateNode(ByteComputeNode Operand) : ByteComputeNode;
internal sealed record ByteNarrowNode(ByteComputeNode Operand) : ByteComputeNode;
internal sealed record ByteBinaryNode(ByteComputeOperation Operation, ByteComputeNode Left, ByteComputeNode Right) : ByteComputeNode;
internal sealed record ByteComputeProgram(IReadOnlyList<ByteComputeNode> Outputs);

internal static class ByteComputeParser
{
    internal static ByteComputeProgram ParseMap<TSource, TDestination>(
        Expression<Func<TSource, TDestination>> expression,
        ComputeValueDescriptor<TSource> sourceDescriptor,
        ComputeValueDescriptor<TDestination> destinationDescriptor)
        where TSource : unmanaged
        where TDestination : unmanaged
    {
        Expression body = expression.Body;
        if (typeof(TSource) == typeof(TDestination) && body == expression.Parameters[0])
            return new ByteComputeProgram(Enumerable.Range(0, sourceDescriptor.ComponentCount).Select(index => (ByteComputeNode)new ByteComponentNode(index)).ToArray());
        if (body is NewExpression construction && construction.Arguments.Count == destinationDescriptor.ComponentCount)
            return new ByteComputeProgram(construction.Arguments.Select(argument => Parse(argument, expression.Parameters[0], sourceDescriptor)).ToArray());
        if (body is MemberInitExpression initializer)
        {
            var outputs = new ByteComputeNode?[destinationDescriptor.ComponentCount];
            foreach (MemberAssignment assignment in initializer.Bindings.OfType<MemberAssignment>())
                outputs[destinationDescriptor.GetComponentIndex(assignment.Member)] = Parse(assignment.Expression, expression.Parameters[0], sourceDescriptor);
            if (outputs.Any(output => output is null))
                throw Unsupported(body, "A byte initializer must assign every registered component.");
            return new ByteComputeProgram(outputs!);
        }
        throw Unsupported(body, "Byte results must use the registered constructor order or initialize every component.");
    }

    internal static ByteComputeProgram ParseProjection<T>(
        Expression<Func<T, float>> expression,
        ComputeValueDescriptor<T> descriptor)
        where T : unmanaged =>
        new([Parse(expression.Body, expression.Parameters[0], descriptor)]);

    private static ByteComputeNode Parse<T>(Expression expression, ParameterExpression parameter, ComputeValueDescriptor<T> descriptor)
        where T : unmanaged
    {
        if (expression is UnaryExpression { NodeType: ExpressionType.Convert } conversion)
        {
            if (conversion.Type == typeof(byte)) return new ByteNarrowNode(Parse(conversion.Operand, parameter, descriptor));
            if (conversion.Type is { } type && (type == typeof(int) || type == typeof(float))) return Parse(conversion.Operand, parameter, descriptor);
        }
        if (expression is UnaryExpression { NodeType: ExpressionType.ConvertChecked })
            throw Unsupported(expression, "Checked byte conversions are not supported on GPU.");
        if (expression is MemberExpression member && member.Expression == parameter && member.Type == typeof(byte))
            return new ByteComponentNode(descriptor.GetComponentIndex(member.Member));
        if (expression is MemberExpression projected && projected.Expression is NewExpression construction)
        {
            int index = descriptor.GetComponentIndex(projected.Member);
            return Parse(construction.Arguments[index], parameter, descriptor);
        }
        if (TryConstant(expression, out int constant)) return new ByteConstantNode(constant);
        if (expression is UnaryExpression { NodeType: ExpressionType.Negate } negate)
            return new ByteNegateNode(Parse(negate.Operand, parameter, descriptor));
        if (expression is BinaryExpression binary)
        {
            ByteComputeOperation operation = binary.NodeType switch
            {
                ExpressionType.Add => ByteComputeOperation.Add,
                ExpressionType.Subtract => ByteComputeOperation.Subtract,
                ExpressionType.Multiply => ByteComputeOperation.Multiply,
                ExpressionType.Divide => ByteComputeOperation.Divide,
                _ => throw Unsupported(binary, $"Integer operator '{binary.NodeType}' is not supported.")
            };
            return new ByteBinaryNode(operation, Parse(binary.Left, parameter, descriptor), Parse(binary.Right, parameter, descriptor));
        }
        throw Unsupported(expression, "Use byte components, integer constants, arithmetic, and explicit byte narrowing.");
    }

    private static bool TryConstant(Expression expression, out int value)
    {
        if (expression is ConstantExpression constant)
        {
            if (constant.Value is int integer) { value = integer; return true; }
            if (constant.Value is byte octet) { value = octet; return true; }
        }
        if (expression is MemberExpression
            {
                Expression: ConstantExpression owner,
                Member: FieldInfo field
            } && owner.Value is not null &&
            owner.Value.GetType().IsDefined(typeof(CompilerGeneratedAttribute), inherit: false))
        {
            object? captured = field.GetValue(owner.Value);
            if (captured is int integer) { value = integer; return true; }
            if (captured is byte octet) { value = octet; return true; }
        }
        value = 0;
        return false;
    }

    private static NotSupportedException Unsupported(Expression expression, string reason) =>
        new($"The byte compute expression '{expression}' is not supported. {reason}");
}
