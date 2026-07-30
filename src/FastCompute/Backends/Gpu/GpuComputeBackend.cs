using System.Collections.Concurrent;
using FastCompute.Expressions;

namespace FastCompute.Backends.Gpu;

internal sealed class GpuComputeBackend : IComputeBackend
{
    private static readonly ConcurrentDictionary<int, Lazy<ComputeContext>>
        PreferredContexts = new();
    private static readonly Lazy<ComputeContext> SharedContext =
        new(
            () => ComputeContext.Create(),
            LazyThreadSafetyMode.ExecutionAndPublication);
    private static readonly Lazy<IReadOnlyList<ComputeDeviceInfo>> Devices =
        new(
            ComputeContext.GetAccelerators,
            LazyThreadSafetyMode.ExecutionAndPublication);

    internal static GpuComputeBackend Instance { get; } = new();

    internal static bool HasHardwareAccelerator =>
        Devices.Value.Any(IsHardwareAccelerator);

    internal static bool TryGetAutomaticMemoryBudget(
        ComputeContext? context,
        int? preferredAcceleratorIndex,
        long? requestedBudget,
        out long budget)
    {
        if (context is not null)
        {
            budget = context.GetAutomaticMemoryBudget(requestedBudget);
            return true;
        }

        if (preferredAcceleratorIndex is int index)
        {
            if (!TryGetPreferredContext(index, out ComputeContext? preferred))
            {
                budget = 0;
                return false;
            }

            budget = preferred!.GetAutomaticMemoryBudget(requestedBudget);
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

    internal static long GetExplicitMemoryBudget(
        ComputeContext? context,
        int? preferredAcceleratorIndex,
        long? requestedBudget)
    {
        ComputeContext effectiveContext =
            ResolveContext(context, preferredAcceleratorIndex);
        return effectiveContext.GetAutomaticMemoryBudget(requestedBudget);
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
        => ResolveContext(context).ExecuteMapPlan(source, plan, context);

    internal ComputeBackendExecution<float[]> ExecuteMapInPlace(
        float[] source,
        ComputeExpressionPlan plan,
        ComputeExecutionContext context)
        => ResolveContext(context).ExecuteMapInPlacePlan(
            source,
            plan,
            context);

    internal ComputeBackendExecution<float[]> ExecuteZipInPlace(
        float[] target,
        float[] right,
        ComputeExpressionPlan plan,
        ComputeExecutionContext context)
        => ResolveContext(context).ExecuteZipInPlacePlan(
            target,
            right,
            plan,
            context);

    public ComputeBackendExecution<float[]> ExecuteZip(
        float[] left,
        float[] right,
        ComputeExpressionPlan plan,
        ComputeExecutionContext context)
        => ResolveContext(context).ExecuteZipPlan(left, right, plan, context);

    public ComputeBackendExecution<float> Reduce(
        float[] source,
        ComputeReductionKind reduction,
        ComputeExecutionContext context)
        => ResolveContext(context).ExecuteReduction(
            source,
            reduction,
            context);

    internal ComputeBackendExecution<int[]> ExecuteHistogram(
        float[] source,
        int binCount,
        float minimum,
        float maximum,
        HistogramOutOfRangeMode outOfRangeMode,
        ComputeExecutionContext context)
        => ResolveContext(context).ExecuteHistogram(
            source,
            binCount,
            minimum,
            maximum,
            outOfRangeMode,
            context);

    internal static ComputeContext ResolveContext(
        ComputeExecutionContext context) =>
        ResolveContext(
            context.GpuContext,
            context.PreferredGpuAcceleratorIndex);

    private static ComputeContext ResolveContext(
        ComputeContext? context,
        int? preferredAcceleratorIndex)
    {
        if (context is not null)
        {
            return context;
        }

        return preferredAcceleratorIndex is int index
            ? GetPreferredContext(index)
            : SharedContext.Value;
    }

    private static ComputeContext GetPreferredContext(int index)
    {
        ComputeDeviceInfo? device =
            Devices.Value.SingleOrDefault(item => item.Index == index);
        if (device is null || !IsHardwareAccelerator(device))
        {
            throw new ArgumentOutOfRangeException(
                nameof(index),
                index,
                "The preferred accelerator index must identify a hardware GPU.");
        }

        return PreferredContexts.GetOrAdd(
            index,
            static acceleratorIndex =>
                new Lazy<ComputeContext>(
                    () => ComputeContext.Create(
                        new ComputeContextOptions
                        {
                            AcceleratorIndex = acceleratorIndex
                        }),
                    LazyThreadSafetyMode.ExecutionAndPublication)).Value;
    }

    private static bool TryGetPreferredContext(
        int index,
        out ComputeContext? context)
    {
        ComputeDeviceInfo? device =
            Devices.Value.SingleOrDefault(item => item.Index == index);
        if (device is null || !IsHardwareAccelerator(device))
        {
            context = null;
            return false;
        }

        try
        {
            context = GetPreferredContext(index);
            return true;
        }
        catch (Exception)
        {
            context = null;
            return false;
        }
    }

    private static bool IsHardwareAccelerator(ComputeDeviceInfo device) =>
        !string.Equals(
            device.AcceleratorType,
            "CPU",
            StringComparison.OrdinalIgnoreCase);
}
