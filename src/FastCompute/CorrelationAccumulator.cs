namespace FastCompute;

/// <summary>Accumulates paired samples for a streaming Pearson correlation.</summary>
public struct CorrelationAccumulator
{
    private long count;
    private double sumX;
    private double sumY;
    private double sumXX;
    private double sumYY;
    private double sumXY;

    /// <summary>Adds one pair of samples.</summary>
    public void Add(float x, float y)
    {
        count++;
        sumX += x;
        sumY += y;
        sumXX += (double)x * x;
        sumYY += (double)y * y;
        sumXY += (double)x * y;
    }

    /// <summary>Gets the accumulated Pearson correlation coefficient.</summary>
    public readonly double Correlation
    {
        get
        {
            if (count < 2) return 0d;
            double covariance = (count * sumXY) - (sumX * sumY);
            double varianceX = (count * sumXX) - (sumX * sumX);
            double varianceY = (count * sumYY) - (sumY * sumY);
            double denominator = Math.Sqrt(Math.Max(0d, varianceX * varianceY));
            return denominator > 1e-30 ? Math.Clamp(covariance / denominator, -1d, 1d) : 0d;
        }
    }
}
