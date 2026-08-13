namespace FastCompute.ImageProcessing;

/// <summary>Contains measurable local peak facts from a two-dimensional spectrum.</summary>
public readonly struct SpectrumPeakMetrics(float maximumRatio, int strongPeakCount)
{
    /// <summary>Gets the maximum power-to-neighbourhood ratio.</summary>
    public float MaximumRatio { get; } = maximumRatio;

    /// <summary>Gets the number of ratios meeting the requested strong-peak threshold.</summary>
    public int StrongPeakCount { get; } = strongPeakCount;
}
