using FastCompute.Diagnostics;

namespace FastCompute.Tests;

public sealed class DiagnosticsTests
{
    [Test]
    public void RunWithDiagnostics_ReturnsResultAndSeparatedTimings()
    {
        float[] source = [1.0f, 2.0f, 3.0f];
        var options = new ComputeOptions { Backend = ComputeBackendKind.Scalar };

        ComputeResult<float[]> result =
            Compute.RunWithDiagnostics(source, value => value * 2.0f, options);

        Assert.Multiple(() =>
        {
            Assert.That(result.Value, Is.EqualTo(new[] { 2.0f, 4.0f, 6.0f }));
            Assert.That(result.Diagnostics.Backend, Is.EqualTo(ComputeBackendKind.Scalar));
            Assert.That(result.Diagnostics.PlanningTime, Is.GreaterThanOrEqualTo(TimeSpan.Zero));
            Assert.That(result.Diagnostics.CompilationTime, Is.GreaterThanOrEqualTo(TimeSpan.Zero));
            Assert.That(result.Diagnostics.ExecutionTime, Is.GreaterThanOrEqualTo(TimeSpan.Zero));
            Assert.That(result.Diagnostics.UploadTime, Is.EqualTo(TimeSpan.Zero));
            Assert.That(result.Diagnostics.DownloadTime, Is.EqualTo(TimeSpan.Zero));
            Assert.That(result.Diagnostics.KernelCacheHit, Is.False);
            Assert.That(result.Diagnostics.DeviceName, Is.Null);
        });
    }

    [Test]
    public void RunWithDiagnostics_ReportsForcedParallelBackend()
    {
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.ParallelCpu,
            MaxDegreeOfParallelism = 2
        };

        ComputeResult<float[]> result =
            Compute.RunWithDiagnostics([1.0f, 2.0f], value => value, options);

        Assert.That(result.Diagnostics.Backend, Is.EqualTo(ComputeBackendKind.ParallelCpu));
    }

    [TestCase(0)]
    [TestCase(-1)]
    public void InvalidMaximumParallelism_IsRejected(int maximumParallelism)
    {
        var options = new ComputeOptions
        {
            MaxDegreeOfParallelism = maximumParallelism
        };

        Assert.Throws<ArgumentOutOfRangeException>(
            () => Compute.Run([1.0f], value => value, options));
    }

    [Test]
    public void NegativeThreshold_IsRejected()
    {
        var options = new ComputeOptions
        {
            Thresholds = new ComputeThresholdOptions { ParallelThreshold = -1 }
        };

        Assert.Throws<ArgumentOutOfRangeException>(
            () => Compute.Run([1.0f], value => value, options));
    }

    [Test]
    public void NullThresholdOptions_AreRejected()
    {
        var options = new ComputeOptions { Thresholds = null! };

        Assert.Throws<ArgumentNullException>(
            () => Compute.Run([1.0f], value => value, options));
    }
}
