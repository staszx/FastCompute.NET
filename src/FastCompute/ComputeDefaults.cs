namespace FastCompute;

/// <summary>
/// Configures process-wide defaults used when an operation does not provide
/// an equivalent setting through <see cref="ComputeOptions"/>.
/// </summary>
public static class ComputeDefaults
{
    private const int NoAccelerator = -1;
    private static int preferredGpuAcceleratorIndex = NoAccelerator;

    /// <summary>
    /// Gets or sets the preferred hardware GPU accelerator index.
    /// </summary>
    /// <remarks>
    /// The setting is used by Auto and explicit GPU operations only when
    /// neither <see cref="ComputeOptions.GpuContext"/> nor
    /// <see cref="ComputeOptions.PreferredGpuAcceleratorIndex"/> is supplied
    /// for that operation. In Auto mode the setting selects which GPU is
    /// considered but does not force GPU execution. Set it to
    /// <see langword="null"/> to restore automatic accelerator selection.
    /// </remarks>
    public static int? PreferredGpuAcceleratorIndex
    {
        get
        {
            int index = Volatile.Read(
                ref preferredGpuAcceleratorIndex);
            return index == NoAccelerator ? null : index;
        }
        set
        {
            if (value is < 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(value),
                    value,
                    "PreferredGpuAcceleratorIndex cannot be negative.");
            }

            Volatile.Write(
                ref preferredGpuAcceleratorIndex,
                value ?? NoAccelerator);
        }
    }

    internal static ComputeOptions Apply(ComputeOptions options)
    {
        int? acceleratorIndex = PreferredGpuAcceleratorIndex;
        if (acceleratorIndex is null ||
            options.Backend is not (
                ComputeBackendKind.Auto or
                ComputeBackendKind.Gpu) ||
            options.GpuContext is not null ||
            options.PreferredGpuAcceleratorIndex is not null)
        {
            return options;
        }

        return options.WithPreferredGpuAcceleratorIndex(
            acceleratorIndex.Value);
    }
}
