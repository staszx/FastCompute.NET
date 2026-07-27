namespace FastCompute.Tests;

[TestFixture]
[Category("GPU")]
public sealed class GpuStreamingTests
{
    private const int NvidiaAcceleratorIndex = 2;

    [Test]
    public void DoubleBufferedMap_MatchesSequentialGpuWithPartialLastChunk()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(2_503);
        float[] original = (float[])source.Clone();
        const int chunkElementCount = 1_000;
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context,
            GpuChunkElementCount = chunkElementCount
        };

        float[] expected = Compute.Run(
            source,
            value =>
                GpuMath.Sin(value) *
                GpuMath.Exp(-value * value),
            options);
        float[] actual = Compute.Run(
            source,
            value =>
                GpuMath.Sin(value) *
                GpuMath.Exp(-value * value),
            new ComputeOptions
            {
                Backend = ComputeBackendKind.Gpu,
                GpuContext = context,
                GpuChunkElementCount = chunkElementCount,
                EnableGpuStreaming = true
            });

        Assert.Multiple(() =>
        {
            Assert.That(actual, Is.EqualTo(expected).Within(1e-5f));
            Assert.That(source, Is.EqualTo(original));
            Assert.That(context.DeviceName, Does.Contain("NVIDIA"));
        });
    }

    [Test]
    public void DoubleBufferedMap_ReusesFourDeviceBuffers()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(4_096);

        _ = Compute.Run(
            source,
            value => value * 2.0f + 1.0f,
            new ComputeOptions
            {
                Backend = ComputeBackendKind.Gpu,
                GpuContext = context,
                GpuChunkElementCount = 1_024,
                EnableGpuStreaming = true
            });
        ComputeMemoryPoolStatistics afterFirst =
            context.MemoryPoolStatistics;
        _ = Compute.Run(
            source,
            value => value * 3.0f - 1.0f,
            new ComputeOptions
            {
                Backend = ComputeBackendKind.Gpu,
                GpuContext = context,
                GpuChunkElementCount = 1_024,
                EnableGpuStreaming = true
            });
        ComputeMemoryPoolStatistics afterSecond =
            context.MemoryPoolStatistics;

        Assert.Multiple(() =>
        {
            Assert.That(afterFirst.AllocatedBuffers, Is.EqualTo(4));
            Assert.That(
                afterSecond.AllocatedBuffers,
                Is.EqualTo(afterFirst.AllocatedBuffers));
            Assert.That(
                afterSecond.Reuses - afterFirst.Reuses,
                Is.EqualTo(4));
        });
    }

    [Test]
    public void DoubleBufferedMap_ValidatesChunkSizeAndHandlesEmptyInput()
    {
        using ComputeContext context = CreateCudaContext();

        Assert.Multiple(() =>
        {
            Assert.That(
                Compute.Run(
                    [],
                    value => value + 1.0f,
                    new ComputeOptions
                    {
                        Backend = ComputeBackendKind.Gpu,
                        GpuContext = context,
                        GpuChunkElementCount = 128,
                        EnableGpuStreaming = true
                    }),
                Is.Empty);
            Assert.Throws<ArgumentOutOfRangeException>(
                () => Compute.Run(
                    [1.0f],
                    value => value + 1.0f,
                    new ComputeOptions
                    {
                        Backend = ComputeBackendKind.Gpu,
                        GpuContext = context,
                        GpuChunkElementCount = 0,
                        EnableGpuStreaming = true
                    }));
        });
    }

    [Test]
    public void StreamingMap_PublicApiReportsTwoStreamsAndPhysicalTransfers()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(2_503);
        const int chunkElementCount = 1_000;
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context,
            GpuChunkElementCount = chunkElementCount,
            EnableGpuStreaming = true
        };

        var result = Compute.RunWithDiagnostics(
            source,
            value =>
                GpuMath.Sin(value) *
                GpuMath.Exp(-value * value),
            options);
        float[] expected = source
            .Select(
                value =>
                    MathF.Sin(value) *
                    MathF.Exp(-value * value))
            .ToArray();

        Assert.Multiple(() =>
        {
            Assert.That(result.Value, Is.EqualTo(expected).Within(1e-5f));
            Assert.That(result.Diagnostics.IsStreaming, Is.True);
            Assert.That(result.Diagnostics.StreamCount, Is.EqualTo(2));
            Assert.That(result.Diagnostics.ChunkCount, Is.EqualTo(3));
            Assert.That(
                result.Diagnostics.UploadedBytes,
                Is.EqualTo(3L * chunkElementCount * sizeof(float)));
            Assert.That(
                result.Diagnostics.DownloadedBytes,
                Is.EqualTo(3L * chunkElementCount * sizeof(float)));
            Assert.That(
                result.Diagnostics.BackendSelectionReason,
                Does.Contain("double-buffered streaming"));
        });
    }

    [Test]
    public void StreamingMap_DefaultChunkingRemainsSequential()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(2_503);
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context,
            GpuChunkElementCount = 1_000
        };

        var result = Compute.RunWithDiagnostics(
            source,
            value => value * 2.0f,
            options);

        Assert.Multiple(() =>
        {
            Assert.That(result.Diagnostics.IsChunked, Is.True);
            Assert.That(result.Diagnostics.IsStreaming, Is.False);
            Assert.That(result.Diagnostics.StreamCount, Is.Zero);
        });
    }

    [Test]
    public void StreamingMap_UsesOneShotExecutionWhenChunkingIsUnnecessary()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(128);
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context,
            EnableGpuStreaming = true
        };

        var result = Compute.RunWithDiagnostics(
            source,
            value => value * 2.0f,
            options);

        Assert.Multiple(() =>
        {
            Assert.That(result.Diagnostics.IsChunked, Is.False);
            Assert.That(result.Diagnostics.IsStreaming, Is.False);
            Assert.That(result.Diagnostics.StreamCount, Is.Zero);
        });
    }

    [Test]
    public void StreamingMap_HonorsPreCanceledToken()
    {
        using ComputeContext context = CreateCudaContext();
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context,
            GpuChunkElementCount = 128,
            EnableGpuStreaming = true,
            CancellationToken = cancellation.Token
        };

        Assert.Throws<OperationCanceledException>(
            () => Compute.Run(
                CreateSource(1_024),
                value => value * 2.0f,
                options));
    }

    [Test]
    public void StreamingMap_AccountsForFourDeviceBuffers()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(2_503);
        long insufficientBudget =
            GpuChunkPlan.PlanningOverheadBytes +
            1_000L * sizeof(float) * 2;
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context,
            GpuChunkElementCount = 1_000,
            GpuMemoryBudgetBytes = insufficientBudget,
            EnableGpuStreaming = true
        };

        Assert.Throws<ComputeGpuMemoryBudgetExceededException>(
            () => Compute.Run(
                source,
                value => value * 2.0f,
                options));
    }

    [Test]
    public void StreamingMap_RejectsAutoAndUnsupportedOperations()
    {
        using ComputeContext context = CreateCudaContext();
        var autoOptions = new ComputeOptions
        {
            GpuContext = context,
            GpuChunkElementCount = 2,
            EnableGpuStreaming = true
        };
        var gpuOptions = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context,
            GpuChunkElementCount = 2,
            EnableGpuStreaming = true
        };

        Assert.Multiple(() =>
        {
            Assert.Throws<ArgumentException>(
                () => Compute.Run(
                    [1.0f, 2.0f, 3.0f],
                    value => value + 1.0f,
                    autoOptions));
            Assert.Throws<NotSupportedException>(
                () => Compute.Zip(
                    [1.0f, 2.0f, 3.0f],
                    [3.0f, 2.0f, 1.0f],
                    (left, right) => left + right,
                    gpuOptions));
            Assert.Throws<NotSupportedException>(
                () => Compute.RunInPlace(
                    [1.0f, 2.0f, 3.0f],
                    value => value + 1.0f,
                    gpuOptions));
        });
    }

    private static ComputeContext CreateCudaContext()
    {
        ComputeDeviceInfo device = ComputeContext.GetAccelerators()
            .Single(item => item.Index == NvidiaAcceleratorIndex);
        Assert.That(
            device.AcceleratorType,
            Does.Contain("Cuda").IgnoreCase);
        ComputeContext context =
            ComputeContext.Create(
                new ComputeContextOptions
                {
                    AcceleratorIndex = NvidiaAcceleratorIndex
                });
        TestContext.Out.WriteLine(
            $"GPU streaming accelerator: {context.DeviceName}");
        return context;
    }

    private static float[] CreateSource(int count)
    {
        var source = new float[count];
        for (int index = 0; index < count; index++)
        {
            source[index] = (index % 10_000) / 10_000.0f;
        }

        return source;
    }
}
