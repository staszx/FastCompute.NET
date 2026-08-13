namespace FastCompute.Tests;

public sealed class StatisticsTests
{
    private static readonly ComputeBackendKind[] Backends =
    [
        ComputeBackendKind.Scalar,
        ComputeBackendKind.ParallelCpu,
        ComputeBackendKind.Simd,
        ComputeBackendKind.Gpu
    ];

    [TestCaseSource(nameof(Backends))]
    public void CalculateStatistics_ProducesBackendParity(ComputeBackendKind backend)
    {
        float[] values = [1, 2, 3, 4];
        StatisticsResult result = Compute.CalculateStatistics(
            values,
            new ComputeOptions { Backend = backend });

        Assert.Multiple(() =>
        {
            Assert.That(result.Mean, Is.EqualTo(2.5).Within(1e-10));
            Assert.That(result.Variance, Is.EqualTo(1.25).Within(1e-10));
            Assert.That(result.StandardDeviation, Is.EqualTo(Math.Sqrt(1.25)).Within(1e-10));
            Assert.That(result.Skewness, Is.Zero.Within(1e-10));
            Assert.That(result.Kurtosis, Is.EqualTo(-1.36).Within(1e-10));
        });
    }

    [TestCaseSource(nameof(Backends))]
    public void CorrelationAndRegression_ProduceBackendParity(ComputeBackendKind backend)
    {
        double[] x = [1, 2, 3, 4, 5];
        double[] y = [3, 5, 7, 9, 11];
        var options = new ComputeOptions { Backend = backend };

        double correlation = Compute.Correlation(x, y, options);
        LinearRegressionResult regression = Compute.LinearRegression(x, y, options);

        Assert.Multiple(() =>
        {
            Assert.That(correlation, Is.EqualTo(1).Within(1e-10));
            Assert.That(regression.Slope, Is.EqualTo(2).Within(1e-10));
            Assert.That(regression.Intercept, Is.EqualTo(1).Within(1e-10));
            Assert.That(regression.RSquared, Is.EqualTo(1).Within(1e-10));
        });
    }

    [Test]
    public void AutoCorrelationAndEntropy_HandleDegenerateInput()
    {
        float[] values = [1, 2, 3, 4];

        Assert.Multiple(() =>
        {
            Assert.That(Compute.AutoCorrelation(values, 0), Is.EqualTo(1).Within(1e-12));
            Assert.That(Compute.ShannonEntropy(new[] { 1, 1, 1, 1 }), Is.EqualTo(2).Within(1e-12));
            Assert.That(Compute.ShannonEntropy(new int[4]), Is.Zero);
        });
    }

    [Test]
    public void PercentilesPeaksAndWindows_AreGenericSignalOperations()
    {
        float[] percentileValues = [4, 1, 3, 2];
        float[] signal = [1, 5, 1, 4, 1];
        float[] window = [1, 1, 1, 1, 1];

        float median = Compute.Median(percentileValues);
        SignalPeak[] peaks = Compute.FindPeaks(signal);
        Compute.ApplyHannWindow(window, new ComputeOptions { Backend = ComputeBackendKind.Simd });

        Assert.Multiple(() =>
        {
            Assert.That(median, Is.EqualTo(2.5f));
            Assert.That(peaks.Select(peak => peak.Index), Is.EqualTo(new[] { 1, 3 }));
            Assert.That(window, Is.EqualTo(new[] { 0f, 0.5f, 1f, 0.5f, 0f }).Within(1e-5f));
        });
    }

    [TestCaseSource(nameof(Backends))]
    public void NormalizeAndSafeDivide_ProduceBackendParity(ComputeBackendKind backend)
    {
        var options = new ComputeOptions { Backend = backend };
        float[] normalized = Compute.Normalize(new float[] { 2f, 4f, 6f }, options);
        float[] divided = Compute.SafeDivide(
            new float[] { 4f, 5f, 6f },
            new float[] { 2f, 0f, -3f },
            zeroResult: -1f,
            options: options);

        Assert.Multiple(() =>
        {
            Assert.That(normalized, Is.EqualTo(new[] { 0f, 0.5f, 1f }).Within(1e-6f));
            Assert.That(divided, Is.EqualTo(new[] { 2f, -1f, -2f }).Within(1e-6f));
        });
    }
}
