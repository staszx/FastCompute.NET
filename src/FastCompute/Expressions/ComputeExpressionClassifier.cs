namespace FastCompute.Expressions;

internal static class ComputeExpressionClassifier
{
    internal static int GetGpuThreshold(
        ComputeExpressionPlan? plan,
        ComputeThresholdOptions thresholds)
    {
        if (plan is null)
        {
            return thresholds.GpuMediumThreshold;
        }

        if (ContainsHeavyFunction(plan.Root))
        {
            return thresholds.GpuHeavyThreshold;
        }

        return ContainsFunction(plan.Root)
            ? thresholds.GpuMediumThreshold
            : thresholds.GpuSimpleThreshold;
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
