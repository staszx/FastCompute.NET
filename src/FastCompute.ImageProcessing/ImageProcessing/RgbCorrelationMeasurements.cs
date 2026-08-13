namespace FastCompute.ImageProcessing;

/// <summary>Contains measurable RGB channel and neighbour correlations.</summary>
public readonly struct RgbCorrelationMeasurements(float[] channelCorrelations, float[] neighbourCorrelations)
{
    /// <summary>Gets same-pixel RG, RB, and GB correlations.</summary>
    public float[] ChannelCorrelations { get; } = channelCorrelations;

    /// <summary>Gets RG-right, RG-down, BG-right, and BG-down correlations.</summary>
    public float[] NeighbourCorrelations { get; } = neighbourCorrelations;
}
