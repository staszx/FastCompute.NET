namespace FastCompute.ImageProcessing;

/// <summary>Contains measured energy in radial image-frequency bands.</summary>
public readonly struct FrequencyBandEnergy(double low, double mid, double high)
{
    /// <summary>Gets energy below the low-frequency boundary.</summary>
    public double Low { get; } = low;

    /// <summary>Gets energy in the middle-frequency band.</summary>
    public double Mid { get; } = mid;

    /// <summary>Gets energy above the middle-frequency boundary.</summary>
    public double High { get; } = high;

    /// <summary>Gets total measured energy.</summary>
    public double Total => Low + Mid + High;
}
