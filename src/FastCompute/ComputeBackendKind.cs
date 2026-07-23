namespace FastCompute;

/// <summary>
/// Identifies an execution backend.
/// </summary>
public enum ComputeBackendKind
{
    /// <summary>Lets FastCompute choose an available backend.</summary>
    Auto,

    /// <summary>Executes operations in a single-threaded scalar CPU loop.</summary>
    Scalar,

    /// <summary>Executes operations on multiple CPU threads.</summary>
    ParallelCpu,

    /// <summary>Executes operations with CPU SIMD instructions.</summary>
    Simd,

    /// <summary>Executes operations on a GPU.</summary>
    Gpu
}
