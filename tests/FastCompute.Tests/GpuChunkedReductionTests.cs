namespace FastCompute.Tests;

[TestFixture]
[Category("GPU")]
public sealed class GpuChunkedReductionTests
{
    private const int NvidiaAcceleratorIndex = 2;
    private const long PlanningOverheadBytes = 1024 * 1024;

    [Test]
    public void Reductions_GpuProcessSequentialChunksAndReportTransfers()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(2_503);
        long budget =
            PlanningOverheadBytes + 2L * sizeof(float) * 1_000;
        var gpuOptions = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context,
            GpuMemoryBudgetBytes = budget
        };
        var scalarOptions = new ComputeOptions
        {
            Backend = ComputeBackendKind.Scalar
        };

        var sum = Compute.SumWithDiagnostics(source, gpuOptions);
        var minimum = Compute.MinWithDiagnostics(source, gpuOptions);
        var maximum = Compute.MaxWithDiagnostics(source, gpuOptions);
        var average = Compute.AverageWithDiagnostics(source, gpuOptions);

        TestContext.Out.WriteLine(
            $"Chunked reduction accelerator: {sum.Diagnostics.DeviceName}");
        Assert.Multiple(() =>
        {
            Assert.That(
                sum.Value,
                Is.EqualTo(Compute.Sum(source, scalarOptions)).Within(2e-3f));
            Assert.That(
                minimum.Value,
                Is.EqualTo(Compute.Min(source, scalarOptions)));
            Assert.That(
                maximum.Value,
                Is.EqualTo(Compute.Max(source, scalarOptions)));
            Assert.That(
                average.Value,
                Is.EqualTo(Compute.Average(source, scalarOptions))
                    .Within(1e-6f));
            AssertReductionDiagnostics(sum.Diagnostics, source.Length);
            AssertReductionDiagnostics(minimum.Diagnostics, source.Length);
            AssertReductionDiagnostics(maximum.Diagnostics, source.Length);
            AssertReductionDiagnostics(average.Diagnostics, source.Length);
        });
    }

    [Test]
    public void Reductions_GpuPropagateNaNAcrossChunkBoundaries()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(2_503);
        source[1_500] = float.NaN;
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context,
            GpuChunkElementCount = 1_000
        };

        Assert.Multiple(() =>
        {
            Assert.That(
                float.IsNaN(Compute.Min(source, options)),
                Is.True);
            Assert.That(
                float.IsNaN(Compute.Max(source, options)),
                Is.True);
            Assert.That(
                float.IsNaN(Compute.Sum(source, options)),
                Is.True);
            Assert.That(
                float.IsNaN(Compute.Average(source, options)),
                Is.True);
        });
    }

    [Test]
    public void Reduction_AutoSelectsChunkedGpuAboveFullMemoryBudget()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(2_503);
        long budget =
            PlanningOverheadBytes + 2L * sizeof(float) * 1_000;
        var options = new ComputeOptions
        {
            GpuContext = context,
            GpuMemoryBudgetBytes = budget,
            Thresholds = new ComputeThresholdOptions
            {
                GpuMediumThreshold = 0
            }
        };

        var result = Compute.MaxWithDiagnostics(source, options);

        Assert.Multiple(() =>
        {
            Assert.That(
                result.Diagnostics.Backend,
                Is.EqualTo(ComputeBackendKind.Gpu));
            Assert.That(result.Diagnostics.IsChunked, Is.True);
            Assert.That(
                result.Diagnostics.BackendSelectionReason,
                Does.Contain("chunked"));
            Assert.That(
                result.Diagnostics.EstimatedGpuMemoryBytes,
                Is.GreaterThan(result.Diagnostics.GpuMemoryBudgetBytes));
        });
    }

    [Test]
    public void Reduction_ExplicitGpuRejectsBudgetWhenChunkingDisabled()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(2_503);
        long budget =
            PlanningOverheadBytes + 2L * sizeof(float) * 1_000;
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context,
            GpuMemoryBudgetBytes = budget,
            EnableGpuChunking = false
        };

        Assert.Multiple(() =>
        {
            Assert.Throws<ComputeGpuMemoryBudgetExceededException>(
                () => Compute.Sum(source, options));
            Assert.That(
                context.MemoryPoolStatistics.AllocatedBuffers,
                Is.Zero);
        });
    }

    [Test]
    public void Reduction_EmptySumDiagnosticsRemainDefined()
    {
        using ComputeContext context = CreateCudaContext();
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context,
            GpuChunkElementCount = 1_000
        };

        var result = Compute.SumWithDiagnostics([], options);

        Assert.Multiple(() =>
        {
            Assert.That(result.Value, Is.Zero);
            Assert.That(result.Diagnostics.ChunkCount, Is.Zero);
            Assert.That(result.Diagnostics.UploadedBytes, Is.Zero);
            Assert.That(result.Diagnostics.DownloadedBytes, Is.Zero);
        });
    }

    private static void AssertReductionDiagnostics(
        FastCompute.Diagnostics.ComputeDiagnostics diagnostics,
        int sourceLength)
    {
        Assert.That(diagnostics.Backend, Is.EqualTo(ComputeBackendKind.Gpu));
        Assert.That(diagnostics.ChunkCount, Is.EqualTo(3));
        Assert.That(diagnostics.ChunkElementCount, Is.EqualTo(1_000));
        Assert.That(
            diagnostics.UploadedBytes,
            Is.EqualTo((long)sourceLength * sizeof(float)));
        Assert.That(
            diagnostics.DownloadedBytes,
            Is.EqualTo(3L * sizeof(float)));
        Assert.That(diagnostics.DeviceName, Does.Contain("NVIDIA"));
    }

    private static ComputeContext CreateCudaContext()
    {
        ComputeDeviceInfo device = ComputeContext.GetAccelerators()
            .Single(item => item.Index == NvidiaAcceleratorIndex);
        Assert.That(
            device.AcceleratorType,
            Does.Contain("Cuda").IgnoreCase);
        return ComputeContext.Create(new ComputeContextOptions
        {
            AcceleratorIndex = NvidiaAcceleratorIndex
        });
    }

    private static float[] CreateSource(int count)
    {
        var source = new float[count];
        for (int index = 0; index < count; index++)
        {
            source[index] = (index - count / 2) / 10_000.0f;
        }

        return source;
    }
}
