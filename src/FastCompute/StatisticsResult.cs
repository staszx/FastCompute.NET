namespace FastCompute;

/// <summary>Contains population distribution moments calculated together.</summary>
public readonly struct StatisticsResult
{
    internal StatisticsResult(double mean, double variance, double skewness, double kurtosis)
    {
        Mean = mean;
        Variance = variance;
        StandardDeviation = Math.Sqrt(variance);
        Skewness = skewness;
        Kurtosis = kurtosis;
    }

    /// <summary>Gets the arithmetic mean.</summary>
    public double Mean { get; }
    /// <summary>Gets the population variance.</summary>
    public double Variance { get; }
    /// <summary>Gets the population standard deviation.</summary>
    public double StandardDeviation { get; }
    /// <summary>Gets the standardized third central moment.</summary>
    public double Skewness { get; }
    /// <summary>Gets excess kurtosis.</summary>
    public double Kurtosis { get; }
}
