namespace FastCompute;

/// <summary>
/// Controls which expression optimizations may be applied.
/// </summary>
public enum ComputeOptimizationMode
{
    /// <summary>Preserves observable IEEE 754 behavior.</summary>
    Strict,

    /// <summary>Allows transformations that may relax IEEE 754 behavior.</summary>
    Fast
}
