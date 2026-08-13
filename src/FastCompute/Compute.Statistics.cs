namespace FastCompute;

public static partial class Compute
{
    /// <summary>Calculates population moments for single-precision values.</summary>
    public static StatisticsResult CalculateStatistics(ReadOnlySpan<float> values, ComputeOptions? options = null) =>
        CalculateStatistics(ToDoubles(values), options);

    /// <summary>Calculates population moments for double-precision values.</summary>
    public static StatisticsResult CalculateStatistics(ReadOnlySpan<double> values, ComputeOptions? options = null)
    {
        if (values.IsEmpty) return default;
        double[] data = values.ToArray();
        double mean = Average(data, options);
        double variance = ReduceMapped(data, value => (value - mean) * (value - mean), ComputeReductionKind.Sum, options) / data.Length;
        if (variance <= 1e-30) return new StatisticsResult(mean, Math.Max(0, variance), 0, 0);
        double m3 = ReduceMapped(data, value => (value - mean) * (value - mean) * (value - mean), ComputeReductionKind.Sum, options) / data.Length;
        double m4 = ReduceMapped(data, value => (value - mean) * (value - mean) * (value - mean) * (value - mean), ComputeReductionKind.Sum, options) / data.Length;
        double deviation = Math.Sqrt(variance);
        return new StatisticsResult(mean, variance, m3 / (deviation * deviation * deviation), (m4 / (variance * variance)) - 3d);
    }

    /// <summary>Calculates the mean of single-precision values.</summary>
    public static double Mean(ReadOnlySpan<float> values, ComputeOptions? options = null) =>
        values.IsEmpty ? 0d : Average(ToDoubles(values), options);

    /// <summary>Calculates the mean of double-precision values.</summary>
    public static double Mean(ReadOnlySpan<double> values, ComputeOptions? options = null) =>
        values.IsEmpty ? 0d : Average(values.ToArray(), options);

    /// <summary>Calculates population variance.</summary>
    public static double Variance(ReadOnlySpan<float> values, ComputeOptions? options = null) => CalculateStatistics(values, options).Variance;

    /// <summary>Calculates population variance.</summary>
    public static double Variance(ReadOnlySpan<double> values, ComputeOptions? options = null) => CalculateStatistics(values, options).Variance;

    /// <summary>Calculates population standard deviation.</summary>
    public static double StandardDeviation(ReadOnlySpan<float> values, ComputeOptions? options = null) => CalculateStatistics(values, options).StandardDeviation;

    /// <summary>Calculates population standard deviation.</summary>
    public static double StandardDeviation(ReadOnlySpan<double> values, ComputeOptions? options = null) => CalculateStatistics(values, options).StandardDeviation;

    /// <summary>Calculates skewness.</summary>
    public static double Skewness(ReadOnlySpan<float> values, ComputeOptions? options = null) => CalculateStatistics(values, options).Skewness;

    /// <summary>Calculates excess kurtosis.</summary>
    public static double Kurtosis(ReadOnlySpan<float> values, ComputeOptions? options = null) => CalculateStatistics(values, options).Kurtosis;

    /// <summary>Calculates the sum of squared values.</summary>
    public static double SumOfSquares(ReadOnlySpan<float> values, ComputeOptions? options = null) =>
        values.IsEmpty ? 0d : ReduceMapped(ToDoubles(values), value => value * value, ComputeReductionKind.Sum, options);

    /// <summary>Calculates covariance between equally sized sequences.</summary>
    public static double Covariance(ReadOnlySpan<float> x, ReadOnlySpan<float> y, ComputeOptions? options = null) =>
        Covariance(ToDoubles(x), ToDoubles(y), options);

    /// <summary>Calculates covariance between equally sized sequences.</summary>
    public static double Covariance(ReadOnlySpan<double> x, ReadOnlySpan<double> y, ComputeOptions? options = null)
    {
        ValidatePairs(x.Length, y.Length);
        if (x.IsEmpty) return 0d;
        double[] left = x.ToArray();
        double[] right = y.ToArray();
        double meanX = Average(left, options);
        double meanY = Average(right, options);
        return ReduceZipped(left, right, (a, b) => (a - meanX) * (b - meanY), ComputeReductionKind.Sum, options) / left.Length;
    }

    /// <summary>Calculates the Pearson correlation coefficient.</summary>
    public static double Correlation(ReadOnlySpan<float> x, ReadOnlySpan<float> y, ComputeOptions? options = null) =>
        Correlation(ToDoubles(x), ToDoubles(y), options);

    /// <summary>Calculates the Pearson correlation coefficient.</summary>
    public static double Correlation(ReadOnlySpan<double> x, ReadOnlySpan<double> y, ComputeOptions? options = null)
    {
        ValidatePairs(x.Length, y.Length);
        if (x.IsEmpty) return 0d;
        double[] left = x.ToArray();
        double[] right = y.ToArray();
        double meanX = Average(left, options);
        double meanY = Average(right, options);
        double covariance = ReduceZipped(left, right, (a, b) => (a - meanX) * (b - meanY), ComputeReductionKind.Sum, options);
        double varianceX = ReduceMapped(left, value => (value - meanX) * (value - meanX), ComputeReductionKind.Sum, options);
        double varianceY = ReduceMapped(right, value => (value - meanY) * (value - meanY), ComputeReductionKind.Sum, options);
        double denominator = Math.Sqrt(Math.Max(0d, varianceX * varianceY));
        return denominator > 1e-30 ? Math.Clamp(covariance / denominator, -1d, 1d) : 0d;
    }

    /// <summary>Calculates correlation between a sequence and a lagged copy.</summary>
    public static double AutoCorrelation(ReadOnlySpan<float> values, int lag, ComputeOptions? options = null)
    {
        if (lag < 0 || lag >= values.Length) throw new ArgumentOutOfRangeException(nameof(lag));
        return lag == 0
            ? Correlation(values, values, options)
            : Correlation(values[..^lag], values[lag..], options);
    }

    /// <summary>Fits <c>y = slope * x + intercept</c> using ordinary least squares.</summary>
    public static LinearRegressionResult LinearRegression(ReadOnlySpan<double> x, ReadOnlySpan<double> y, ComputeOptions? options = null)
    {
        ValidatePairs(x.Length, y.Length);
        if (x.Length < 2) return default;
        double meanX = Mean(x, options);
        double meanY = Mean(y, options);
        double covariance = Covariance(x, y, options);
        double varianceX = Variance(x, options);
        double slope = varianceX > 1e-30 ? covariance / varianceX : 0d;
        double intercept = meanY - (slope * meanX);
        double[] observed = y.ToArray();
        double ssResidual = ReduceZipped(x.ToArray(), observed, (a, b) => (b - ((slope * a) + intercept)) * (b - ((slope * a) + intercept)), ComputeReductionKind.Sum, options);
        double ssTotal = ReduceMapped(observed, value => (value - meanY) * (value - meanY), ComputeReductionKind.Sum, options);
        double rSquared = ssTotal > 1e-30 ? 1d - (ssResidual / ssTotal) : 0d;
        return new LinearRegressionResult(slope, intercept, Math.Clamp(rSquared, 0d, 1d));
    }

    /// <summary>Calculates Shannon entropy from non-negative histogram counts.</summary>
    public static double ShannonEntropy(ReadOnlySpan<int> histogram)
    {
        long total = 0;
        for (int index = 0; index < histogram.Length; index++)
        {
            if (histogram[index] < 0) throw new ArgumentOutOfRangeException(nameof(histogram));
            total += histogram[index];
        }
        if (total == 0) return 0d;
        double entropy = 0d;
        for (int index = 0; index < histogram.Length; index++)
        {
            if (histogram[index] == 0) continue;
            double probability = (double)histogram[index] / total;
            entropy -= probability * Math.Log2(probability);
        }
        return entropy;
    }

    private static double[] ToDoubles(ReadOnlySpan<float> values)
    {
        var result = GC.AllocateUninitializedArray<double>(values.Length);
        for (int index = 0; index < values.Length; index++) result[index] = values[index];
        return result;
    }

    private static void ValidatePairs(int xLength, int yLength)
    {
        if (xLength != yLength) throw new ArgumentException("Sequences must have equal lengths.");
    }

}
