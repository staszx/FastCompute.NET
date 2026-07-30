using System.Linq.Expressions;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace FastCompute.Expressions;

internal enum NumericOpCode
{
    Parameter0,
    Parameter1,
    Constant,
    Negate,
    Add,
    Subtract,
    Multiply,
    Divide,
    Abs,
    Min,
    Max,
    Clamp,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Exp,
    Log,
    Log10,
    Pow,
    Floor,
    Ceiling,
    Round
}

internal readonly record struct NumericInstruction<T>(
    NumericOpCode OpCode,
    T Operand)
    where T : unmanaged;

internal sealed record NumericExpressionProgram<T>(
    NumericInstruction<T>[] Instructions,
    int ParameterCount,
    int MaximumStackDepth)
    where T : unmanaged;

internal static class NumericExpressionParser
{
    internal static NumericExpressionProgram<T> Parse<T>(
        LambdaExpression expression)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(expression);
        if (typeof(T) != typeof(double) && typeof(T) != typeof(int))
        {
            throw new NotSupportedException(
                $"Numeric expressions support double and int, not '{typeof(T).Name}'.");
        }

        if (expression.Parameters.Count is < 1 or > 2 ||
            expression.Parameters.Any(parameter => parameter.Type != typeof(T)) ||
            expression.ReturnType != typeof(T))
        {
            throw Unsupported(
                expression,
                $"The expression must have one or two {typeof(T).Name} parameters and a {typeof(T).Name} result.");
        }

        var parameterIndexes =
            new Dictionary<ParameterExpression, int>(
                expression.Parameters.Count);
        for (int index = 0; index < expression.Parameters.Count; index++)
        {
            parameterIndexes.Add(expression.Parameters[index], index);
        }

        var instructions = new List<NumericInstruction<T>>();
        int depth = Emit(
            expression.Body,
            parameterIndexes,
            instructions,
            currentDepth: 0,
            out int maximumDepth);
        if (depth != 1)
        {
            throw new InvalidOperationException(
                "The numeric expression compiler produced an invalid stack.");
        }

        return new NumericExpressionProgram<T>(
            instructions.ToArray(),
            expression.Parameters.Count,
            maximumDepth);
    }

    private static int Emit<T>(
        Expression expression,
        IReadOnlyDictionary<ParameterExpression, int> parameterIndexes,
        List<NumericInstruction<T>> instructions,
        int currentDepth,
        out int maximumDepth)
        where T : unmanaged
    {
        if (TryReadConstant(expression, out T constant))
        {
            instructions.Add(
                new NumericInstruction<T>(
                    NumericOpCode.Constant,
                    constant));
            maximumDepth = currentDepth + 1;
            return currentDepth + 1;
        }

        switch (expression)
        {
            case ParameterExpression parameter:
                if (!parameterIndexes.TryGetValue(parameter, out int index))
                {
                    throw Unsupported(
                        expression,
                        "The parameter does not belong to this expression.");
                }

                instructions.Add(
                    new NumericInstruction<T>(
                        index == 0
                            ? NumericOpCode.Parameter0
                            : NumericOpCode.Parameter1,
                        default));
                maximumDepth = currentDepth + 1;
                return currentDepth + 1;

            case UnaryExpression
                {
                    NodeType: ExpressionType.Negate or
                        ExpressionType.NegateChecked
                } unary:
            {
                int depth = Emit(
                    unary.Operand,
                    parameterIndexes,
                    instructions,
                    currentDepth,
                    out maximumDepth);
                instructions.Add(
                    new NumericInstruction<T>(
                        NumericOpCode.Negate,
                        default));
                return depth;
            }

            case BinaryExpression binary:
            {
                NumericOpCode operation = binary.NodeType switch
                {
                    ExpressionType.Add or ExpressionType.AddChecked =>
                        NumericOpCode.Add,
                    ExpressionType.Subtract or ExpressionType.SubtractChecked =>
                        NumericOpCode.Subtract,
                    ExpressionType.Multiply or ExpressionType.MultiplyChecked =>
                        NumericOpCode.Multiply,
                    ExpressionType.Divide => NumericOpCode.Divide,
                    _ => throw Unsupported(
                        expression,
                        $"Binary operator '{binary.NodeType}' is not supported.")
                };
                int leftDepth = Emit(
                    binary.Left,
                    parameterIndexes,
                    instructions,
                    currentDepth,
                    out int leftMaximum);
                int rightDepth = Emit(
                    binary.Right,
                    parameterIndexes,
                    instructions,
                    leftDepth,
                    out int rightMaximum);
                instructions.Add(
                    new NumericInstruction<T>(operation, default));
                maximumDepth = Math.Max(leftMaximum, rightMaximum);
                return rightDepth - 1;
            }

            case MethodCallExpression call:
                return EmitCall(
                    call,
                    parameterIndexes,
                    instructions,
                    currentDepth,
                    out maximumDepth);

            default:
                throw Unsupported(
                    expression,
                    $"Expression node '{expression.NodeType}' is not supported.");
        }
    }

    private static int EmitCall<T>(
        MethodCallExpression expression,
        IReadOnlyDictionary<ParameterExpression, int> parameterIndexes,
        List<NumericInstruction<T>> instructions,
        int currentDepth,
        out int maximumDepth)
        where T : unmanaged
    {
        NumericOpCode operation = GetMethodOperation<T>(expression);
        int depth = currentDepth;
        int maximum = currentDepth;
        foreach (Expression argument in expression.Arguments)
        {
            depth = Emit(
                argument,
                parameterIndexes,
                instructions,
                depth,
                out int argumentMaximum);
            maximum = Math.Max(maximum, argumentMaximum);
        }

        instructions.Add(new NumericInstruction<T>(operation, default));
        maximumDepth = maximum;
        return depth - expression.Arguments.Count + 1;
    }

    private static NumericOpCode GetMethodOperation<T>(
        MethodCallExpression expression)
        where T : unmanaged
    {
        Type expectedType = typeof(T) == typeof(double)
            ? typeof(Math)
            : typeof(Math);
        if (expression.Method.DeclaringType != expectedType)
        {
            throw Unsupported(
                expression,
                $"Only arithmetic operators and supported System.Math methods for {typeof(T).Name} are allowed.");
        }

        NumericOpCode operation = expression.Method.Name switch
        {
            nameof(Math.Abs) => NumericOpCode.Abs,
            nameof(Math.Min) => NumericOpCode.Min,
            nameof(Math.Max) => NumericOpCode.Max,
            nameof(Math.Clamp) => NumericOpCode.Clamp,
            nameof(Math.Sqrt) when typeof(T) == typeof(double) =>
                NumericOpCode.Sqrt,
            nameof(Math.Sin) when typeof(T) == typeof(double) =>
                NumericOpCode.Sin,
            nameof(Math.Cos) when typeof(T) == typeof(double) =>
                NumericOpCode.Cos,
            nameof(Math.Tan) when typeof(T) == typeof(double) =>
                NumericOpCode.Tan,
            nameof(Math.Exp) when typeof(T) == typeof(double) =>
                NumericOpCode.Exp,
            nameof(Math.Log) when typeof(T) == typeof(double) =>
                NumericOpCode.Log,
            nameof(Math.Log10) when typeof(T) == typeof(double) =>
                NumericOpCode.Log10,
            nameof(Math.Pow) when typeof(T) == typeof(double) =>
                NumericOpCode.Pow,
            nameof(Math.Floor) when typeof(T) == typeof(double) =>
                NumericOpCode.Floor,
            nameof(Math.Ceiling) when typeof(T) == typeof(double) =>
                NumericOpCode.Ceiling,
            nameof(Math.Round) when typeof(T) == typeof(double) =>
                NumericOpCode.Round,
            _ => throw Unsupported(
                expression,
                $"Method '{expression.Method.Name}' is not supported for {typeof(T).Name}.")
        };

        int expectedArity = operation switch
        {
            NumericOpCode.Min or NumericOpCode.Max or NumericOpCode.Pow => 2,
            NumericOpCode.Clamp => 3,
            _ => 1
        };
        if (expression.Arguments.Count != expectedArity)
        {
            throw Unsupported(
                expression,
                $"Method '{expression.Method.Name}' has an unsupported overload.");
        }

        return operation;
    }

    private static bool TryReadConstant<T>(
        Expression expression,
        out T result)
        where T : unmanaged
    {
        object? value = expression switch
        {
            ConstantExpression constant => constant.Value,
            MemberExpression
            {
                Expression: ConstantExpression owner,
                Member: FieldInfo field
            } when owner.Value is not null &&
                   owner.Value.GetType().IsDefined(
                       typeof(CompilerGeneratedAttribute),
                       inherit: false) =>
                field.GetValue(owner.Value),
            _ => null
        };

        if (value is T typed)
        {
            result = typed;
            return true;
        }

        result = default;
        return false;
    }

    private static GpuExpressionNotSupportedException Unsupported(
        Expression expression,
        string description) =>
        new(
            expression.NodeType,
            expression.ToString(),
            description,
            [
                "Use arithmetic operators and the supported System.Math overloads."
            ]);
}
