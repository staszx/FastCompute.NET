namespace FastCompute.Diagnostics;

/// <summary>
/// Contains a compute value and diagnostics collected for its execution.
/// </summary>
/// <typeparam name="T">The computed value type.</typeparam>
/// <param name="Value">The computed value.</param>
/// <param name="Diagnostics">The collected diagnostics.</param>
public sealed record ComputeResult<T>(
    T Value,
    ComputeDiagnostics Diagnostics);
