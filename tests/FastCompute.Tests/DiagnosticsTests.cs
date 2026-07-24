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

    [Test]
    public void Auto_DoesNotSelectGpuForSimpleCpuResidentExpressionByDefault()
    {
        float[] source = new float[4_096];

        ComputeResult<float[]> result =
            Compute.RunWithDiagnostics(source, value => value + 2.0f);

        Assert.Multiple(() =>
        {
            Assert.That(result.Diagnostics.Backend, Is.Not.EqualTo(ComputeBackendKind.Gpu));
            Assert.That(
                result.Diagnostics.BackendSelectionReason,
                Does.Contain("disabled by default"));
            Assert.That(result.Diagnostics.EstimatedGpuMemoryBytes, Is.Null);
        });
    }

    [Test]
    public void GpuWorkingSetEstimate_RejectsFourGibibyteOneShotMapOnFourGibibyteGpu()
    {
        const int elementCount = 1024 * 1024 * 1024;

        long workingSet =
            Compute.EstimateGpuWorkingSetBytes(
                parameterCount: 1,
                elementCount);

        Assert.That(workingSet, Is.GreaterThan(8L * 1024 * 1024 * 1024));
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

    [TestCase(0L)]
    [TestCase(-1L)]
    public void InvalidGpuMemoryBudget_IsRejected(long memoryBudget)
    {
        var options = new ComputeOptions
        {
            GpuMemoryBudgetBytes = memoryBudget
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
