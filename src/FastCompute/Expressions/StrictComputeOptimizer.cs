namespace FastCompute.Expressions;

internal static class StrictComputeOptimizer
{
    internal static ComputeExpressionPlan Optimize(ComputeExpressionPlan plan) =>
        plan with { Root = OptimizeNode(plan.Root) };

    private static ComputeNode OptimizeNode(ComputeNode node) =>
        node switch
        {
            UnaryNode unary => OptimizeUnary(unary),
            BinaryNode binary => OptimizeBinary(binary),
            FunctionNode function => OptimizeFunction(function),
            _ => node
        };

    private static ComputeNode OptimizeUnary(UnaryNode unary)
    {
        ComputeNode operand = OptimizeNode(unary.Operand);

        if (operand is ConstantNode constant)
        {
            return new ConstantNode(-constant.Value);
        }

        if (operand is UnaryNode { Operation: UnaryOperation.Negate } nested)
        {
            return nested.Operand;
        }

        return unary with { Operand = operand };
    }

    private static ComputeNode OptimizeBinary(BinaryNode binary)
    {
        ComputeNode left = OptimizeNode(binary.Left);
        ComputeNode right = OptimizeNode(binary.Right);

        if (left is ConstantNode leftConstant && right is ConstantNode rightConstant)
        {
            return new ConstantNode(EvaluateBinary(binary.Operation, leftConstant.Value, rightConstant.Value));
        }

        if (binary.Operation == BinaryOperation.Multiply)
        {
            if (IsOne(right))
            {
                return left;
            }

            if (IsOne(left))
            {
                return right;
            }
        }

        if (binary.Operation == BinaryOperation.Divide && IsOne(right))
        {
            return left;
        }

        if (binary.Operation == BinaryOperation.Subtract && IsPositiveZero(right))
        {
            return left;
        }

        return binary with { Left = left, Right = right };
    }

    private static ComputeNode OptimizeFunction(FunctionNode function)
    {
        var arguments = new ComputeNode[function.Arguments.Count];
        bool allConstants = true;

        for (int index = 0; index < arguments.Length; index++)
        {
            arguments[index] = OptimizeNode(function.Arguments[index]);
            allConstants &= arguments[index] is ConstantNode;
        }

        if (!allConstants)
        {
            return function with { Arguments = arguments };
        }

        var values = new float[arguments.Length];
        for (int index = 0; index < values.Length; index++)
        {
            values[index] = ((ConstantNode)arguments[index]).Value;
        }

        return new ConstantNode(EvaluateFunction(function.Function, values));
    }

    private static float EvaluateBinary(BinaryOperation operation, float left, float right) =>
        operation switch
        {
            BinaryOperation.Add => left + right,
            BinaryOperation.Subtract => left - right,
            BinaryOperation.Multiply => left * right,
            BinaryOperation.Divide => left / right,
            _ => throw new ArgumentOutOfRangeException(nameof(operation))
        };

    private static float EvaluateFunction(ComputeFunction function, IReadOnlyList<float> values) =>
        function switch
        {
            ComputeFunction.Abs => GpuMath.Abs(values[0]),
            ComputeFunction.Min => GpuMath.Min(values[0], values[1]),
            ComputeFunction.Max => GpuMath.Max(values[0], values[1]),
            ComputeFunction.Clamp => GpuMath.Clamp(values[0], values[1], values[2]),
            ComputeFunction.Sqrt => GpuMath.Sqrt(values[0]),
            ComputeFunction.Sin => GpuMath.Sin(values[0]),
            ComputeFunction.Cos => GpuMath.Cos(values[0]),
            ComputeFunction.Tan => GpuMath.Tan(values[0]),
            ComputeFunction.Exp => GpuMath.Exp(values[0]),
            ComputeFunction.Log => GpuMath.Log(values[0]),
            ComputeFunction.Log10 => GpuMath.Log10(values[0]),
            ComputeFunction.Pow => GpuMath.Pow(values[0], values[1]),
            ComputeFunction.Floor => GpuMath.Floor(values[0]),
            ComputeFunction.Ceiling => GpuMath.Ceiling(values[0]),
            ComputeFunction.Round => GpuMath.Round(values[0]),
            _ => throw new ArgumentOutOfRangeException(nameof(function))
        };

    private static bool IsOne(ComputeNode node) =>
        node is ConstantNode { Value: 1.0f };

    private static bool IsPositiveZero(ComputeNode node) =>
        node is ConstantNode constant &&
        BitConverter.SingleToInt32Bits(constant.Value) == 0;
}
