namespace FastCompute;

/// <summary>
/// Identifies an unmanaged composite value whose homogeneous scalar components
/// can be processed natively by FastCompute backends.
/// </summary>
/// <typeparam name="TSelf">The implementing value type.</typeparam>
public interface IComputeValue<TSelf>
    where TSelf : unmanaged, IComputeValue<TSelf>
{
    /// <summary>
    /// Gets the validated component layout used by FastCompute.
    /// </summary>
    static abstract ComputeValueDescriptor<TSelf> ComputeDescriptor { get; }
}
