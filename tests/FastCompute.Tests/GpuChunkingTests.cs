namespace FastCompute.Tests;

[TestFixture]
[Category("GPU")]
public sealed class GpuChunkingTests
{
    private const int NvidiaAcceleratorIndex = 2;
    private const long PlanningOverheadBytes = 1024 * 1024;

    [Test]
    public void Run_GpuUsesConfiguredChunksWithoutGapsOrOverlap()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = Enumerable.Range(0, 10_003)
            .Select(index => (float)index)
            .ToArray();
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context,
            GpuChunkElementCount = 1_024
        };

        var result = Compute.RunWithDiagnostics(
            source,
            value => value * 2.0f + 3.0f,
            options);

        TestContext.Out.WriteLine(
            $"Chunked Map accelerator: {result.Diagnostics.DeviceName}");
        Assert.Multiple(() =>
        {
            Assert.That(
                result.Value,
                Is.EqualTo(source.Select(value => value * 2.0f + 3.0f)));
            Assert.That(result.Diagnostics.ChunkCount, Is.EqualTo(10));
            Assert.That(
                result.Diagnostics.ChunkElementCount,
                Is.EqualTo(1_024));
            Assert.That(result.Diagnostics.IsChunked, Is.True);
            Assert.That(
                result.Diagnostics.UploadedBytes,
                Is.EqualTo((long)source.Length * sizeof(float)));
            Assert.That(
                result.Diagnostics.DownloadedBytes,
                Is.EqualTo((long)source.Length * sizeof(float)));
        });
    }

    [Test]
    public void Run_GpuDerivesChunkSizeFromMemoryBudget()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(2_503);
        long budget = PlanningOverheadBytes + 2L * sizeof(float) * 1_000;
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context,
            GpuMemoryBudgetBytes = budget
        };

        var result = Compute.RunWithDiagnostics(
            source,
            value => ComputeMath.Sin(value) + value * value,
            options);
        float[] expected = source
            .Select(value => MathF.Sin(value) + value * value)
            .ToArray();

        Assert.Multiple(() =>
        {
            Assert.That(result.Value, Is.EqualTo(expected).Within(2e-4f));
            Assert.That(result.Diagnostics.ChunkCount, Is.EqualTo(3));
            Assert.That(
                result.Diagnostics.ChunkElementCount,
                Is.EqualTo(1_000));
            Assert.That(
                result.Diagnostics.GpuMemoryBudgetBytes,
                Is.EqualTo(budget));
        });
    }

    [Test]
    public void Zip_GpuProcessesSequentialChunksAndReportsTransfers()
    {
        using ComputeContext context = CreateCudaContext();
        float[] left = CreateSource(2_503);
        float[] right = Enumerable.Range(0, left.Length)
            .Select(index => index / 100.0f)
            .ToArray();
        long budget = PlanningOverheadBytes + 3L * sizeof(float) * 1_000;
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context,
            GpuMemoryBudgetBytes = budget
        };

        var result = Compute.ZipWithDiagnostics(
            left,
            right,
            (x, y) => x * 2.0f - y,
            options);
        float[] expected = left
            .Zip(right, (x, y) => x * 2.0f - y)
            .ToArray();

        TestContext.Out.WriteLine(
            $"Chunked Zip accelerator: {result.Diagnostics.DeviceName}");
        Assert.Multiple(() =>
        {
            Assert.That(result.Value, Is.EqualTo(expected).Within(1e-5f));
            Assert.That(result.Diagnostics.ChunkCount, Is.EqualTo(3));
            Assert.That(
                result.Diagnostics.ChunkElementCount,
                Is.EqualTo(1_000));
            Assert.That(
                result.Diagnostics.UploadedBytes,
                Is.EqualTo((long)left.Length * sizeof(float) * 2));
            Assert.That(
                result.Diagnostics.DownloadedBytes,
                Is.EqualTo((long)left.Length * sizeof(float)));
        });
    }

    [Test]
    public void Run_ExplicitGpuRejectsOversizedWorkingSetWhenChunkingDisabled()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(2_503);
        long budget = PlanningOverheadBytes + 2L * sizeof(float) * 1_000;
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context,
            GpuMemoryBudgetBytes = budget,
            EnableGpuChunking = false
        };

        var exception = Assert.Throws<ComputeGpuMemoryBudgetExceededException>(
            () => Compute.Run(source, value => value + 1.0f, options));

        Assert.Multiple(() =>
        {
            Assert.That(exception!.BudgetBytes, Is.EqualTo(budget));
            Assert.That(exception.EstimatedBytes, Is.GreaterThan(budget));
        });
    }

    [Test]
    public void Run_ExplicitGpuRejectsConfiguredChunkThatExceedsBudget()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(2_503);
        long budget = PlanningOverheadBytes + 2L * sizeof(float) * 1_000;
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context,
            GpuMemoryBudgetBytes = budget,
            GpuChunkElementCount = 1_001
        };

        Assert.Throws<ComputeGpuMemoryBudgetExceededException>(
            () => Compute.Run(source, value => value + 1.0f, options));
    }

    [Test]
    public void Run_AutoSelectsChunkedGpuWhenFullWorkingSetExceedsBudget()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(2_503);
        long budget = PlanningOverheadBytes + 2L * sizeof(float) * 1_000;
        var options = new ComputeOptions
        {
            GpuContext = context,
            GpuMemoryBudgetBytes = budget,
            Thresholds = new ComputeThresholdOptions
            {
                GpuHeavyThreshold = 0
            }
        };

        var result = Compute.RunWithDiagnostics(
            source,
            value => ComputeMath.Sin(value) + ComputeMath.Exp(value),
            options);

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
    public void Run_AutoDoesNotSelectGpuAboveBudgetWhenChunkingDisabled()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(2_503);
        long budget = PlanningOverheadBytes + 2L * sizeof(float) * 1_000;
        var options = new ComputeOptions
        {
            GpuContext = context,
            GpuMemoryBudgetBytes = budget,
            EnableGpuChunking = false,
            Thresholds = new ComputeThresholdOptions
            {
                GpuHeavyThreshold = 0
            }
        };

        var result = Compute.RunWithDiagnostics(
            source,
            value => ComputeMath.Sin(value) + ComputeMath.Exp(value),
            options);

        Assert.Multiple(() =>
        {
            Assert.That(
                result.Diagnostics.Backend,
                Is.Not.EqualTo(ComputeBackendKind.Gpu));
            Assert.That(result.Diagnostics.IsChunked, Is.False);
            Assert.That(
                result.Diagnostics.BackendSelectionReason,
                Does.Contain("exceeds"));
        });
    }

    [Test]
    public void Options_RejectNonPositiveGpuChunkElementCount()
    {
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Scalar,
            GpuChunkElementCount = 0
        };

        Assert.Throws<ArgumentOutOfRangeException>(
            () => Compute.Run([1.0f], value => value, options));
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
        for (int index = 0; index < source.Length; index++)
        {
            source[index] = (index - count / 2) / 10_000.0f;
        }

        return source;
    }
}
