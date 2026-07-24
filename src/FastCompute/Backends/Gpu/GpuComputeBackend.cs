using FastCompute.Expressions;

namespace FastCompute.Backends.Gpu;

internal sealed class GpuComputeBackend : IComputeBackend
{
    private static readonly Lazy<ComputeContext> SharedContext =
        new(
            () => ComputeContext.Create(),
            LazyThreadSafetyMode.ExecutionAndPublication);
    private static readonly Lazy<bool> HardwareAcceleratorAvailable =
        new(
            () => ComputeContext.GetAccelerators().Any(
                device => !string.Equals(
                    device.AcceleratorType,
                    "CPU",
                    StringComparison.OrdinalIgnoreCase)),
            LazyThreadSafetyMode.ExecutionAndPublication);

    internal static GpuComputeBackend Instance { get; } = new();

    internal static bool HasHardwareAccelerator =>
        HardwareAcceleratorAvailable.Value;

    internal static bool TryGetAutomaticMemoryBudget(
        ComputeContext? context,
        long? requestedBudget,
        out long budget)
    {
        if (context is not null)
        {
            budget = context.GetAutomaticMemoryBudget(requestedBudget);
            return true;
        }

        if (!HasHardwareAccelerator)
        {
            budget = 0;
            return false;
        }

        budget = SharedContext.Value.GetAutomaticMemoryBudget(requestedBudget);
        return true;
    }

    private GpuComputeBackend()
    {
    }

    public ComputeBackendKind Kind => ComputeBackendKind.Gpu;

    public bool IsAvailable => true;

    public bool Supports(ComputeExpressionPlan plan) => true;

    public ComputeBackendExecution<float[]> ExecuteMap(
        float[] source,
        ComputeExpressionPlan plan,
        ComputeExecutionContext context)
    {
        if (context.GpuContext is not null)
        {
            return context.GpuContext.ExecuteMapPlan(source, plan, context);
        }

        return SharedContext.Value.ExecuteMapPlan(source, plan, context);
    }

    public ComputeBackendExecution<float[]> ExecuteZip(
        float[] left,
        float[] right,
        ComputeExpressionPlan plan,
        ComputeExecutionContext context)
    {
        if (context.GpuContext is not null)
        {
            return context.GpuContext.ExecuteZipPlan(left, right, plan, context);
        }

        return SharedContext.Value.ExecuteZipPlan(left, right, plan, context);
    }

    public ComputeBackendExecution<float> Reduce(
        float[] source,
        ComputeReductionKind reduction,
        ComputeExecutionContext context)
    {
        if (context.GpuContext is not null)
        {
            return context.GpuContext.ExecuteReduction(
                source,
                reduction,
                context);
        }

        return SharedContext.Value.ExecuteReduction(source, reduction, context);
    }
}
