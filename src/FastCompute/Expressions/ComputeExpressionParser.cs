using System.Linq.Expressions;

namespace FastCompute.Expressions;

internal static class ComputeExpressionParser
{
    private static readonly IReadOnlyDictionary<string, (ComputeFunction Function, int Arity)> SupportedFunctions =
        new Dictionary<string, (ComputeFunction, int)>(StringComparer.Ordinal)
        {
            [nameof(GpuMath.Abs)] = (ComputeFunction.Abs, 1),
            [nameof(GpuMath.Min)] = (ComputeFunction.Min, 2),
            [nameof(GpuMath.Max)] = (ComputeFunction.Max, 2),
            [nameof(GpuMath.Clamp)] = (ComputeFunction.Clamp, 3),
            [nameof(GpuMath.Sqrt)] = (ComputeFunction.Sqrt, 1),
            [nameof(GpuMath.Sin)] = (ComputeFunction.Sin, 1),
            [nameof(GpuMath.Cos)] = (ComputeFunction.Cos, 1),
            [nameof(GpuMath.Tan)] = (ComputeFunction.Tan, 1),
            [nameof(GpuMath.Exp)] = (ComputeFunction.Exp, 1),
            [nameof(GpuMath.Log)] = (ComputeFunction.Log, 1),
            [nameof(GpuMath.Log10)] = (ComputeFunction.Log10, 1),
            [nameof(GpuMath.Pow)] = (ComputeFunction.Pow, 2),
            [nameof(GpuMath.Floor)] = (ComputeFunction.Floor, 1),
            [nameof(GpuMath.Ceiling)] = (ComputeFunction.Ceiling, 1),
            [nameof(GpuMath.Round)] = (ComputeFunction.Round, 1)
        };

    internal static ComputeExpressionPlan Parse(LambdaExpression expression)
    {
        ArgumentNullException.ThrowIfNull(expression);

        if (expression.Parameters.Count is < 1 or > 2 ||
            expression.Parameters.Any(parameter => parameter.Type != typeof(float)) ||
            expression.ReturnType != typeof(float))
        {
            throw Unsupported(
                expression,
                "Only expressions with one or two float parameters and a float result are supported.");
        }

        var parameterIndexes = new Dictionary<ParameterExpression, int>(expression.Parameters.Count);
        for (int index = 0; index < expression.Parameters.Count; index++)
        {
            parameterIndexes.Add(expression.Parameters[index], index);
        }

        ComputeNode root = ParseNode(expression.Body, parameterIndexes);
        return new ComputeExpressionPlan(expression.Parameters.Count, root);
    }

    private static ComputeNode ParseNode(
        Expression expression,
        IReadOnlyDictionary<ParameterExpression, int> parameterIndexes)
    {
        if (expression.Type != typeof(float))
        {
            throw Unsupported(expression, "Every value inside a compute expression must have type float.");
        }

        return expression switch
        {
            ParameterExpression parameter => ParseParameter(parameter, parameterIndexes),
            ConstantExpression constant => new ConstantNode((float)constant.Value!),
            BinaryExpression binary => ParseBinary(binary, parameterIndexes),
            UnaryExpression unary => ParseUnary(unary, parameterIndexes),
            MethodCallExpression call => ParseMethodCall(call, parameterIndexes),
            MemberExpression => throw Unsupported(
                expression,
                "Captured values and access to object members are not supported in this implementation stage."),
            _ => throw Unsupported(
                expression,
                $"Expression node '{expression.NodeType}' is not supported.")
        };
    }

    private static ParameterNode ParseParameter(
        ParameterExpression parameter,
        IReadOnlyDictionary<ParameterExpression, int> parameterIndexes)
    {
        if (!parameterIndexes.TryGetValue(parameter, out int index))
        {
            throw Unsupported(parameter, "The parameter does not belong to the compute expression.");
        }

        return new ParameterNode(index);
    }

    private static BinaryNode ParseBinary(
        BinaryExpression expression,
        IReadOnlyDictionary<ParameterExpression, int> parameterIndexes)
    {
        BinaryOperation operation = expression.NodeType switch
        {
            ExpressionType.Add => BinaryOperation.Add,
            ExpressionType.Subtract => BinaryOperation.Subtract,
            ExpressionType.Multiply => BinaryOperation.Multiply,
            ExpressionType.Divide => BinaryOperation.Divide,
            _ => throw Unsupported(
                expression,
                $"Binary operator '{expression.NodeType}' is not supported.")
        };

        return new BinaryNode(
            operation,
            ParseNode(expression.Left, parameterIndexes),
            ParseNode(expression.Right, parameterIndexes));
    }

    private static UnaryNode ParseUnary(
        UnaryExpression expression,
        IReadOnlyDictionary<ParameterExpression, int> parameterIndexes)
    {
        if (expression.NodeType != ExpressionType.Negate)
        {
            throw Unsupported(
                expression,
                $"Unary operator '{expression.NodeType}' is not supported.");
        }

        return new UnaryNode(
            UnaryOperation.Negate,
            ParseNode(expression.Operand, parameterIndexes));
    }

    private static FunctionNode ParseMethodCall(
        MethodCallExpression expression,
        IReadOnlyDictionary<ParameterExpression, int> parameterIndexes)
    {
        if (expression.Method.DeclaringType != typeof(GpuMath) ||
            !SupportedFunctions.TryGetValue(expression.Method.Name, out var function) ||
            expression.Arguments.Count != function.Arity)
        {
            throw Unsupported(
                expression,
                $"Method call '{expression.Method.DeclaringType?.Name}.{expression.Method.Name}' is not supported.");
        }

        var arguments = new ComputeNode[expression.Arguments.Count];
        for (int index = 0; index < arguments.Length; index++)
        {
            arguments[index] = ParseNode(expression.Arguments[index], parameterIndexes);
        }

        return new FunctionNode(function.Function, arguments);
    }

    private static GpuExpressionNotSupportedException Unsupported(
        Expression expression,
        string description) =>
        new(
            expression.NodeType,
            expression.ToString(),
            description,
            ["Only arithmetic operators and methods declared in GpuMath are allowed."]);
}
