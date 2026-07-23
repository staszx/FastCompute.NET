namespace FastCompute;

/// <summary>
/// Describes an accelerator visible to FastCompute.
/// </summary>
/// <param name="Index">The index accepted by <see cref="ComputeContext.Create"/>.</param>
/// <param name="Name">The device name reported by ILGPU.</param>
/// <param name="AcceleratorType">The ILGPU accelerator type.</param>
public sealed record ComputeDeviceInfo(int Index, string Name, string AcceleratorType);
