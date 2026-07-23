namespace FastCompute;

/// <summary>
/// Describes transient float-buffer reuse in a GPU context.
/// </summary>
/// <param name="AllocatedBuffers">Number of device buffers allocated by the pool.</param>
/// <param name="Rentals">Total number of buffer rentals.</param>
/// <param name="Reuses">Rentals satisfied by an existing buffer.</param>
/// <param name="AvailableBuffers">Buffers currently available for reuse.</param>
public sealed record ComputeMemoryPoolStatistics(
    long AllocatedBuffers,
    long Rentals,
    long Reuses,
    int AvailableBuffers);
