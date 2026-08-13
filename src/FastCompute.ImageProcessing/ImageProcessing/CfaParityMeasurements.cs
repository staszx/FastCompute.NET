namespace FastCompute.ImageProcessing;

/// <summary>Contains mean interpolation residuals for four parity classes and three RGB channels.</summary>
public sealed class CfaParityMeasurements
{
    /// <summary>Gets mean absolute interpolation residuals indexed by parity then RGB channel.</summary>
    public double[,] MeanResiduals { get; init; } = new double[4, 3];
}
