namespace FastCompute.Tests;

[TestFixture]
[Category("GPU")]
public sealed class GpuHistogramTests
{
    private const int NvidiaAcceleratorIndex = 2;
    private const long PlanningOverheadBytes = 1024 * 1024;

    [Test]
    public void Histogram_GpuMatchesScalarAndUsesAtomicCounters()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = Enumerable.Repeat(0.5f, 25_003).ToArray();
        source[0] = 0.0f;
        source[^1] = 1.0f;
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context
        };

        var result = Compute.HistogramWithDiagnostics(
            source,
            16,
            0.0f,
            1.0f,
            options);
        int[] expected = Compute.Histogram(
            source,
            16,
            0.0f,
            1.0f,
            new ComputeOptions { Backend = ComputeBackendKind.Scalar });

        TestContext.Out.WriteLine(
            $"GPU Histogram accelerator: {result.Diagnostics.DeviceName}");
        Assert.Multiple(() =>
        {
            Assert.That(result.Value, Is.EqualTo(expected));
            Assert.That(result.Diagnostics.ChunkCount, Is.EqualTo(1));
            Assert.That(
                result.Diagnostics.UploadedBytes,
                Is.EqualTo((long)source.Length * sizeof(float)));
            Assert.That(
                result.Diagnostics.DownloadedBytes,
                Is.EqualTo(16L * sizeof(int)));
            Assert.That(result.Diagnostics.DeviceName, Does.Contain("NVIDIA"));
        });
    }

    [Test]
    public void Histogram_GpuDerivesChunkSizeFromBudget()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(2_503);
        const int binCount = 32;
        long budget =
            PlanningOverheadBytes +
            binCount * sizeof(int) +
            1_000L * sizeof(float);
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context,
            GpuMemoryBudgetBytes = budget
        };

        var result = Compute.HistogramWithDiagnostics(
            source,
            binCount,
            -0.2f,
            0.2f,
            options);
        int[] expected = Compute.Histogram(
            source,
            binCount,
            -0.2f,
            0.2f,
            new ComputeOptions { Backend = ComputeBackendKind.Scalar });

        Assert.Multiple(() =>
        {
            Assert.That(result.Value, Is.EqualTo(expected));
            Assert.That(result.Diagnostics.ChunkCount, Is.EqualTo(3));
            Assert.That(
                result.Diagnostics.ChunkElementCount,
                Is.EqualTo(1_000));
            Assert.That(
                result.Diagnostics.EstimatedGpuMemoryBytes,
                Is.EqualTo(
                    Compute.EstimateGpuHistogramWorkingSetBytes(
                        source.Length,
                        binCount)));
            Assert.That(result.Diagnostics.IsChunked, Is.True);
        });
    }

    [Test]
    public void Histogram_AutoSelectsChunkedGpu()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(2_503);
        const int binCount = 32;
        long budget =
            PlanningOverheadBytes +
            binCount * sizeof(int) +
            1_000L * sizeof(float);
        var options = new ComputeOptions
        {
            GpuContext = context,
            GpuMemoryBudgetBytes = budget,
            Thresholds = new ComputeThresholdOptions
            {
                GpuHistogramThreshold = 0
            }
        };

        var result = Compute.HistogramWithDiagnostics(
            source,
            binCount,
            -0.2f,
            0.2f,
            options);

        Assert.Multiple(() =>
        {
            Assert.That(
                result.Diagnostics.Backend,
                Is.EqualTo(ComputeBackendKind.Gpu));
            Assert.That(result.Diagnostics.IsChunked, Is.True);
            Assert.That(
                result.Diagnostics.BackendSelectionReason,
                Does.Contain("chunked Histogram"));
        });
    }

    [Test]
    public void Histogram_AutoDoesNotSelectGpuByDefault()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(300_000);
        var options = new ComputeOptions
        {
            GpuContext = context
        };

        var result = Compute.HistogramWithDiagnostics(
            source,
            32,
            -0.2f,
            0.2f,
            options);

        Assert.Multiple(() =>
        {
            Assert.That(
                result.Diagnostics.Backend,
                Is.Not.EqualTo(ComputeBackendKind.Gpu));
            Assert.That(
                result.Diagnostics.BackendSelectionReason,
                Does.Contain("disabled by default"));
        });
    }

    [Test]
    public void Histogram_ExplicitGpuRejectsBudgetWhenChunkingDisabled()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(2_503);
        const int binCount = 32;
        long budget =
            PlanningOverheadBytes +
            binCount * sizeof(int) +
            1_000L * sizeof(float);
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
                () => Compute.Histogram(
                    source,
                    binCount,
                    -0.2f,
                    0.2f,
                    options));
            Assert.That(
                context.MemoryPoolStatistics.AllocatedBuffers,
                Is.Zero);
        });
    }

    [Test]
    public void PrecompileHistogram_CachesTemplate()
    {
        using ComputeContext context = CreateCudaContext();

        ComputeCompilationResult first =
            context.PrecompileHistogram<float>();
        ComputeCompilationResult second =
            context.PrecompileHistogram<float>();

        Assert.Multiple(() =>
        {
            Assert.That(first.CacheHit, Is.False);
            Assert.That(second.CacheHit, Is.True);
        });
    }

    [Test]
    public void Histogram_GpuHandlesEmptyInput()
    {
        using ComputeContext context = CreateCudaContext();
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context
        };

        var result = Compute.HistogramWithDiagnostics(
            [],
            8,
            0.0f,
            1.0f,
            options);

        Assert.Multiple(() =>
        {
            Assert.That(result.Value, Is.EqualTo(new int[8]));
            Assert.That(result.Diagnostics.ChunkCount, Is.Zero);
            Assert.That(result.Diagnostics.UploadedBytes, Is.Zero);
            Assert.That(result.Diagnostics.DownloadedBytes, Is.Zero);
        });
    }

    [Test]
    public void Histogram_GpuSupportsIgnoreOutOfRangeMode()
    {
        using ComputeContext context = CreateCudaContext();
        int[] result = Compute.Histogram(
            [-1.0f, 0.25f, 0.75f, 2.0f],
            2,
            0.0f,
            1.0f,
            new HistogramOptions
            {
                OutOfRangeMode = HistogramOutOfRangeMode.Ignore
            },
            new ComputeOptions
            {
                Backend = ComputeBackendKind.Gpu,
                GpuContext = context
            });

        Assert.That(result, Is.EqualTo(new[] { 1, 1 }));
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

        source[10] = float.NaN;
        source[20] = -1.0f;
        source[30] = 1.0f;
        return source;
    }
}
