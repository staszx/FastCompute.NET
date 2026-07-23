using FastCompute.Expressions;

namespace FastCompute.Backends;

internal interface IComputeBackend
{
    ComputeBackendKind Kind { get; }

    bool IsAvailable { get; }

    bool Supports(ComputeExpressionPlan plan);

    ComputeBackendExecution<float[]> ExecuteMap(
        float[] source,
        ComputeExpressionPlan plan,
        ComputeExecutionContext context);

    ComputeBackendExecution<float[]> ExecuteZip(
        float[] left,
        float[] right,
        ComputeExpressionPlan plan,
        ComputeExecutionContext context);

    ComputeBackendExecution<float> Reduce(
        float[] source,
        ComputeReductionKind reduction,
        ComputeExecutionContext context);
}
