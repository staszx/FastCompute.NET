namespace AiImageForensics.Statistics;

internal static class StatisticsMath
{
    public static DistributionStatistics Calculate(ReadOnlySpan<float> data)
    {
        if (data.IsEmpty) return new DistributionStatistics();
        double mean = 0;
        for (int i = 0; i < data.Length; i++) mean += data[i];
        mean /= data.Length;

        double m2 = 0, m3 = 0, m4 = 0;
        for (int i = 0; i < data.Length; i++)
        {
            double d = data[i] - mean;
            double d2 = d * d;
            m2 += d2;
            m3 += d2 * d;
            m4 += d2 * d2;
        }
        double variance = m2 / data.Length;
        double standardDeviation = Math.Sqrt(variance);
        double skewness = standardDeviation > 1e-15 ? (m3 / data.Length) / Math.Pow(standardDeviation, 3) : 0;
        double kurtosis = variance > 1e-15 ? ((m4 / data.Length) / (variance * variance)) - 3 : 0;
        return new DistributionStatistics { Mean = mean, Variance = variance, StandardDeviation = standardDeviation, Skewness = skewness, Kurtosis = kurtosis };
    }

    public static double CalculateCorrelation(ReadOnlySpan<float> data, int width, int height, int offsetX, int offsetY)
    {
        if (width <= 0 || height <= 0 || data.Length < checked(width * height)) throw new ArgumentException("Invalid dimensions.");
        int startX = Math.Max(0, -offsetX), endX = Math.Min(width, width - offsetX);
        int startY = Math.Max(0, -offsetY), endY = Math.Min(height, height - offsetY);
        long count = (long)(endX - startX) * (endY - startY);
        if (count <= 1) return 0;

        double sumA = 0, sumB = 0;
        for (int y = startY; y < endY; y++)
        for (int x = startX; x < endX; x++)
        {
            sumA += data[(y * width) + x];
            sumB += data[((y + offsetY) * width) + x + offsetX];
        }
        double meanA = sumA / count, meanB = sumB / count;
        double covariance = 0, varianceA = 0, varianceB = 0;
        for (int y = startY; y < endY; y++)
        for (int x = startX; x < endX; x++)
        {
            double a = data[(y * width) + x] - meanA;
            double b = data[((y + offsetY) * width) + x + offsetX] - meanB;
            covariance += a * b;
            varianceA += a * a;
            varianceB += b * b;
        }
        double denominator = Math.Sqrt(varianceA * varianceB);
        return denominator > 1e-20 ? Math.Clamp(covariance / denominator, -1, 1) : 0;
    }

    public static double CalculateCorrelation(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length || a.IsEmpty) return 0;
        double meanA = 0, meanB = 0;
        for (int i = 0; i < a.Length; i++) { meanA += a[i]; meanB += b[i]; }
        meanA /= a.Length; meanB /= b.Length;
        double covariance = 0, varianceA = 0, varianceB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double da = a[i] - meanA, db = b[i] - meanB;
            covariance += da * db; varianceA += da * da; varianceB += db * db;
        }
        double denominator = Math.Sqrt(varianceA * varianceB);
        return denominator > 1e-20 ? Math.Clamp(covariance / denominator, -1, 1) : 0;
    }
}
