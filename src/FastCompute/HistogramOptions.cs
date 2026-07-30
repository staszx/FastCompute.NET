namespace FastCompute;

/// <summary>
/// Defines how Histogram handles finite values outside the requested range.
/// </summary>
public enum HistogramOutOfRangeMode
{
    /// <summary>Ignores values below the minimum or above the maximum.</summary>
    Ignore,

    /// <summary>
    /// Counts values below the minimum in the first bin and values above the
    /// maximum in the last bin.
    /// </summary>
    Clamp
}

/// <summary>Configures Histogram-specific behavior.</summary>
public sealed class HistogramOptions
{
    /// <summary>
    /// Gets the behavior for finite values outside the requested range.
    /// NaN values are always ignored.
    /// </summary>
    public HistogramOutOfRangeMode OutOfRangeMode { get; init; } =
        HistogramOutOfRangeMode.Clamp;
}
