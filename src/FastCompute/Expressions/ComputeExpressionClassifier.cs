namespace FastCompute.Expressions;

internal enum ComputeExpressionComplexity
{
    Simple,
    Medium,
    Heavy
}

internal static class ComputeExpressionClassifier
{
    internal static int GetGpuThreshold(
        ComputeExpressionPlan? plan,
        ComputeThresholdOptions thresholds)
    {
        return Classify(plan) switch
        {
            ComputeExpressionComplexity.Simple => thresholds.GpuSimpleThreshold,
            ComputeExpressionComplexity.Medium => thresholds.GpuMediumThreshold,
            ComputeExpressionComplexity.Heavy => thresholds.GpuHeavyThreshold,
            _ => throw new ArgumentOutOfRangeException(nameof(plan))
        };
    }

    internal static ComputeExpressionComplexity Classify(
        ComputeExpressionPlan? plan)
    {
        if (plan is null)
        {
            return ComputeExpressionComplexity.Medium;
        }

        if (ContainsHeavyFunction(plan.Root))
        {
            return ComputeExpressionComplexity.Heavy;
        }

        return ContainsFunction(plan.Root)
            ? ComputeExpressionComplexity.Medium
            : ComputeExpressionComplexity.Simple;
    }

    private static bool ContainsHeavyFunction(ComputeNode node) => node switch
    {
        FunctionNode function when function.Function is
            ComputeFunction.Sin or
            ComputeFunction.Cos or
            ComputeFunction.Tan or
            ComputeFunction.Exp or
            ComputeFunction.Log or
            ComputeFunction.Log10 or
            ComputeFunction.Pow => true,
        UnaryNode unary => ContainsHeavyFunction(unary.Operand),
        BinaryNode binary =>
            ContainsHeavyFunction(binary.Left) ||
            ContainsHeavyFunction(binary.Right),
        FunctionNode function =>
            function.Arguments.Any(ContainsHeavyFunction),
        _ => false
    };

    private static bool ContainsFunction(ComputeNode node) => node switch
    {
        FunctionNode => true,
        UnaryNode unary => ContainsFunction(unary.Operand),
        BinaryNode binary =>
            ContainsFunction(binary.Left) ||
            ContainsFunction(binary.Right),
        _ => false
    };
}
