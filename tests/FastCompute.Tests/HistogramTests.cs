namespace FastCompute.Tests;

public sealed class HistogramTests
{
    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    public void Histogram_UsesInclusiveRangeAndIgnoresInvalidValues(
        ComputeBackendKind backend)
    {
        float[] source =
        [
            -0.1f,
            0.0f,
            0.1f,
            0.249f,
            0.25f,
            0.5f,
            0.75f,
            0.999f,
            1.0f,
            1.1f,
            float.NaN
        ];
        var options = new ComputeOptions
        {
            Backend = backend,
            MaxDegreeOfParallelism = 2
        };

        int[] result =
            Compute.Histogram(
                source,
                4,
                0.0f,
                1.0f,
                new HistogramOptions
                {
                    OutOfRangeMode = HistogramOutOfRangeMode.Ignore
                },
                options);

        Assert.That(result, Is.EqualTo(new[] { 3, 1, 1, 3 }));
    }

    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    public void Histogram_ClampsOutOfRangeValuesByDefault(
        ComputeBackendKind backend)
    {
        int[] result = Compute.Histogram(
            [-2.0f, -0.1f, 0.25f, 0.75f, 1.1f, 2.0f, float.NaN],
            2,
            0.0f,
            1.0f,
            new ComputeOptions { Backend = backend });

        Assert.That(result, Is.EqualTo(new[] { 3, 3 }));
    }

    [Test]
    public void Histogram_ParallelMatchesScalar()
    {
        float[] source = Enumerable.Range(0, 25_003)
            .Select(index => (index % 2_000 - 500) / 1_000.0f)
            .ToArray();

        int[] scalar = Compute.Histogram(
            source,
            128,
            0.0f,
            1.0f,
            new ComputeOptions { Backend = ComputeBackendKind.Scalar });
        int[] parallel = Compute.Histogram(
            source,
            128,
            0.0f,
            1.0f,
            new ComputeOptions
            {
                Backend = ComputeBackendKind.ParallelCpu,
                MaxDegreeOfParallelism = 4
            });

        Assert.That(parallel, Is.EqualTo(scalar));
    }

    [Test]
    public void Histogram_EmptySourceReturnsZeroBins()
    {
        int[] result = Compute.Histogram(
            [],
            8,
            -1.0f,
            1.0f,
            new ComputeOptions { Backend = ComputeBackendKind.Scalar });

        Assert.That(result, Is.EqualTo(new int[8]));
    }

    [Test]
    public void HistogramWithDiagnostics_ReportsSelectedBackend()
    {
        var result = Compute.HistogramWithDiagnostics(
            [0.1f, 0.2f, 0.8f],
            2,
            0.0f,
            1.0f,
            new ComputeOptions { Backend = ComputeBackendKind.Scalar });

        Assert.Multiple(() =>
        {
            Assert.That(result.Value, Is.EqualTo(new[] { 2, 1 }));
            Assert.That(
                result.Diagnostics.Backend,
                Is.EqualTo(ComputeBackendKind.Scalar));
        });
    }

    [Test]
    public void Histogram_ValidatesArgumentsAndBackend()
    {
        Assert.Multiple(() =>
        {
            Assert.Throws<ArgumentNullException>(
                () => Compute.Histogram(null!, 4, 0.0f, 1.0f));
            Assert.Throws<ArgumentOutOfRangeException>(
                () => Compute.Histogram([], 0, 0.0f, 1.0f));
            Assert.Throws<ArgumentOutOfRangeException>(
                () => Compute.Histogram(
                    [],
                    4,
                    float.NegativeInfinity,
                    1.0f));
            Assert.Throws<ArgumentOutOfRangeException>(
                () => Compute.Histogram([], 4, 1.0f, 1.0f));
            Assert.Throws<ArgumentOutOfRangeException>(
                () => Compute.Histogram(
                    [],
                    4,
                    0.0f,
                    float.PositiveInfinity));
            Assert.Throws<ComputeBackendNotSupportedException>(
                () => Compute.Histogram(
                    [0.5f],
                    4,
                    0.0f,
                    1.0f,
                    new ComputeOptions
                    {
                        Backend = ComputeBackendKind.Simd
                    }));
        });
    }

    [Test]
    public void Histogram_CancellationBeforeExecutionDoesNotProcessInput()
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();

        Assert.Throws<OperationCanceledException>(
            () => Compute.Histogram(
                [0.1f, 0.2f],
                4,
                0.0f,
                1.0f,
                new ComputeOptions
                {
                    Backend = ComputeBackendKind.ParallelCpu,
                    CancellationToken = cancellation.Token
                }));
    }
}
