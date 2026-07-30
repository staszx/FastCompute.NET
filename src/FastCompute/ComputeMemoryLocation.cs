namespace FastCompute;

/// <summary>Identifies where a resident compute buffer is stored.</summary>
public enum ComputeMemoryLocation
{
    /// <summary>The buffer is owned by a CPU accelerator in host memory.</summary>
    Host,

    /// <summary>The buffer is owned by a hardware accelerator in device memory.</summary>
    Device
}
