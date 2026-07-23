namespace FastCompute;

/// <summary>
/// Identifies a reduction operation.
/// </summary>
public enum ComputeReductionKind
{
    /// <summary>Computes the sum of all elements.</summary>
    Sum,

    /// <summary>Computes the minimum element.</summary>
    Min,

    /// <summary>Computes the maximum element.</summary>
    Max,

    /// <summary>Computes the arithmetic mean.</summary>
    Average
}
