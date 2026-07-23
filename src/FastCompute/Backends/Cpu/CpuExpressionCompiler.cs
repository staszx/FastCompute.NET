using System.Linq.Expressions;
using System.Reflection;
using FastCompute.Expressions;

namespace FastCompute.Backends.Cpu;

internal static class CpuExpressionCompiler
{
    private static readonly IReadOnlyDictionary<ComputeFunction, MethodInfo> FunctionMethods =
        CreateFunctionMethods();

    internal static Func<float, float> CompileUnary(ComputeExpressionPlan plan)
    {
        var parameter = Expression.Parameter(typeof(float), "value");
        Expression body = Build(plan.Root, [parameter]);
        return Expression.Lambda<Func<float, float>>(body, parameter).Compile();
    }

    internal static Func<float, float, float> CompileBinary(ComputeExpressionPlan plan)
    {
        var left = Expression.Parameter(typeof(float), "left");
        var right = Expression.Parameter(typeof(float), "right");
        Expression body = Build(plan.Root, [left, right]);
        return Expression.Lambda<Func<float, float, float>>(body, left, right).Compile();
    }

    private static Expression Build(ComputeNode node, IReadOnlyList<ParameterExpression> parameters) =>
        node switch
        {
            ParameterNode parameter => parameters[parameter.Index],
            ConstantNode constant => Expression.Constant(constant.Value),
            UnaryNode unary => BuildUnary(unary, parameters),
            BinaryNode binary => BuildBinary(binary, parameters),
            FunctionNode function => BuildFunction(function, parameters),
            _ => throw new ComputeCompilationException(
                $"Unsupported IR node '{node.GetType().Name}'.",
                new NotSupportedException(node.GetType().FullName))
        };

    private static Expression BuildUnary(
        UnaryNode unary,
        IReadOnlyList<ParameterExpression> parameters) =>
        unary.Operation switch
        {
            UnaryOperation.Negate => Expression.Negate(Build(unary.Operand, parameters)),
            _ => throw new ArgumentOutOfRangeException(nameof(unary))
        };

    private static Expression BuildBinary(
        BinaryNode binary,
        IReadOnlyList<ParameterExpression> parameters)
    {
        Expression left = Build(binary.Left, parameters);
        Expression right = Build(binary.Right, parameters);

        return binary.Operation switch
        {
            BinaryOperation.Add => Expression.Add(left, right),
            BinaryOperation.Subtract => Expression.Subtract(left, right),
            BinaryOperation.Multiply => Expression.Multiply(left, right),
            BinaryOperation.Divide => Expression.Divide(left, right),
            _ => throw new ArgumentOutOfRangeException(nameof(binary))
        };
    }

    private static Expression BuildFunction(
        FunctionNode function,
        IReadOnlyList<ParameterExpression> parameters)
    {
        var arguments = new Expression[function.Arguments.Count];
        for (int index = 0; index < arguments.Length; index++)
        {
            arguments[index] = Build(function.Arguments[index], parameters);
        }

        return Expression.Call(FunctionMethods[function.Function], arguments);
    }

    private static IReadOnlyDictionary<ComputeFunction, MethodInfo> CreateFunctionMethods()
    {
        var result = new Dictionary<ComputeFunction, MethodInfo>();

        foreach (ComputeFunction function in Enum.GetValues<ComputeFunction>())
        {
            MethodInfo? method = typeof(GpuMath).GetMethod(
                function.ToString(),
                BindingFlags.Public | BindingFlags.Static);

            result.Add(
                function,
                method ?? throw new InvalidOperationException(
                    $"GpuMath does not define the expected method '{function}'."));
        }

        return result;
    }
}
