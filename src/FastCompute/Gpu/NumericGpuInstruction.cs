using System.ComponentModel;

namespace FastCompute.Gpu;

/// <summary>
/// Represents one double-precision instruction consumed by ILGPU runtime
/// kernels. This is infrastructure, not a user-facing API.
/// </summary>
[EditorBrowsable(EditorBrowsableState.Never)]
public readonly record struct DoubleGpuInstruction(
    int OpCode,
    double Operand);

/// <summary>
/// Represents one integer instruction consumed by ILGPU runtime kernels.
/// This is infrastructure, not a user-facing API.
/// </summary>
[EditorBrowsable(EditorBrowsableState.Never)]
public readonly record struct IntGpuInstruction(
    int OpCode,
    int Operand);
