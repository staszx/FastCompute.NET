namespace FastCompute.Tests;

[TestFixture]
[Category("GPU")]
public sealed class GpuInPlaceChunkingTests
{
    private const int NvidiaAcceleratorIndex = 2;
    private const long PlanningOverheadBytes = 1024 * 1024;

    [Test]
    public void RunInPlace_GpuDerivesOneBufferChunksFromBudget()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(2_503);
        float[] original = (float[])source.Clone();
        long budget = PlanningOverheadBytes + sizeof(float) * 1_000L;
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context,
            GpuMemoryBudgetBytes = budget
        };

        var result = Compute.RunInPlaceWithDiagnostics(
            source,
            value => ComputeMath.Sin(value) + value * value,
            options);
        float[] expected = original
            .Select(value => MathF.Sin(value) + value * value)
            .ToArray();

        TestContext.Out.WriteLine(
            $"Chunked in-place Map accelerator: " +
            $"{result.Diagnostics.DeviceName}");
        Assert.Multiple(() =>
        {
            Assert.That(result.Value, Is.SameAs(source));
            Assert.That(source, Is.EqualTo(expected).Within(2e-4f));
            Assert.That(result.Diagnostics.IsInPlace, Is.True);
            Assert.That(result.Diagnostics.ChunkCount, Is.EqualTo(3));
            Assert.That(
                result.Diagnostics.ChunkElementCount,
                Is.EqualTo(1_000));
            Assert.That(
                result.Diagnostics.UploadedBytes,
                Is.EqualTo((long)source.Length * sizeof(float)));
            Assert.That(
                result.Diagnostics.DownloadedBytes,
                Is.EqualTo((long)source.Length * sizeof(float)));
        });
    }

    [Test]
    public void RunInPlace_AutoSelectsChunkedGpuForHeavyExpression()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(2_503);
        long budget = PlanningOverheadBytes + sizeof(float) * 1_000L;
        var options = new ComputeOptions
        {
            GpuContext = context,
            GpuMemoryBudgetBytes = budget,
            Thresholds = new ComputeThresholdOptions
            {
                GpuHeavyThreshold = 0
            }
        };

        var result = Compute.RunInPlaceWithDiagnostics(
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
                Does.Contain("chunked in-place Map"));
        });
    }

    [Test]
    public void ZipInPlace_GpuProcessesTwoBufferChunks()
    {
        using ComputeContext context = CreateCudaContext();
        float[] target = CreateSource(2_503);
        float[] original = (float[])target.Clone();
        float[] right = CreateRight(target.Length);
        long budget =
            PlanningOverheadBytes + 2L * sizeof(float) * 1_000;
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context,
            GpuMemoryBudgetBytes = budget
        };

        var result = Compute.ZipInPlaceWithDiagnostics(
            target,
            right,
            (left, value) =>
                left * value + ComputeMath.Sqrt(ComputeMath.Abs(left - value)),
            options);
        float[] expected = original
            .Zip(
                right,
                (left, value) =>
                    left * value + MathF.Sqrt(MathF.Abs(left - value)))
            .ToArray();

        TestContext.Out.WriteLine(
            $"Chunked in-place Zip accelerator: " +
            $"{result.Diagnostics.DeviceName}");
        Assert.Multiple(() =>
        {
            Assert.That(result.Value, Is.SameAs(target));
            Assert.That(target, Is.EqualTo(expected).Within(2e-4f));
            Assert.That(result.Diagnostics.ChunkCount, Is.EqualTo(3));
            Assert.That(
                result.Diagnostics.ChunkElementCount,
                Is.EqualTo(1_000));
            Assert.That(
                result.Diagnostics.EstimatedGpuMemoryBytes,
                Is.EqualTo(
                    Compute.EstimateGpuZipInPlaceWorkingSetBytes(
                        target.Length)));
            Assert.That(
                result.Diagnostics.UploadedBytes,
                Is.EqualTo((long)target.Length * sizeof(float) * 2));
            Assert.That(
                result.Diagnostics.DownloadedBytes,
                Is.EqualTo((long)target.Length * sizeof(float)));
        });
    }

    [Test]
    public void ZipInPlace_AutoSelectsChunkedGpuForHeavyExpression()
    {
        using ComputeContext context = CreateCudaContext();
        float[] target = CreateSource(2_503);
        float[] right = CreateRight(target.Length);
        long budget =
            PlanningOverheadBytes + 2L * sizeof(float) * 1_000;
        var options = new ComputeOptions
        {
            GpuContext = context,
            GpuMemoryBudgetBytes = budget,
            Thresholds = new ComputeThresholdOptions
            {
                GpuHeavyThreshold = 0
            }
        };

        var result = Compute.ZipInPlaceWithDiagnostics(
            target,
            right,
            (left, value) =>
                ComputeMath.Sin(left) + ComputeMath.Exp(value),
            options);

        Assert.Multiple(() =>
        {
            Assert.That(
                result.Diagnostics.Backend,
                Is.EqualTo(ComputeBackendKind.Gpu));
            Assert.That(result.Diagnostics.IsChunked, Is.True);
            Assert.That(
                result.Diagnostics.BackendSelectionReason,
                Does.Contain("chunked in-place Zip"));
        });
    }

    [Test]
    public void ZipInPlace_ExplicitGpuRejectsBudgetBeforeMutationWhenDisabled()
    {
        using ComputeContext context = CreateCudaContext();
        float[] target = CreateSource(2_503);
        float[] original = (float[])target.Clone();
        float[] right = CreateRight(target.Length);
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
                () => Compute.ZipInPlace(
                    target,
                    right,
                    (left, value) => left + value,
                    options));
            Assert.That(target, Is.EqualTo(original));
            Assert.That(
                context.MemoryPoolStatistics.AllocatedBuffers,
                Is.Zero);
        });
    }

    [Test]
    public void ZipInPlace_GpuSupportsAliasedRightArray()
    {
        using ComputeContext context = CreateCudaContext();
        float[] target = CreateSource(2_503);
        float[] original = (float[])target.Clone();
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context,
            GpuChunkElementCount = 1_000
        };

        float[] result = Compute.ZipInPlace(
            target,
            target,
            (left, right) => left + right,
            options);

        Assert.Multiple(() =>
        {
            Assert.That(result, Is.SameAs(target));
            Assert.That(
                target,
                Is.EqualTo(original.Select(value => value * 2.0f))
                    .Within(1e-6f));
        });
    }

    [Test]
    public void ComputeContext_ZipInPlaceUsesSelectedAccelerator()
    {
        using ComputeContext context = CreateCudaContext();
        float[] target = [1.0f, 2.0f, 3.0f];
        float[] right = [4.0f, 5.0f, 6.0f];

        float[] result =
            context.ZipInPlace(
                target,
                right,
                (left, value) => left * 2.0f + value);

        Assert.Multiple(() =>
        {
            Assert.That(result, Is.SameAs(target));
            Assert.That(target, Is.EqualTo(new[] { 6.0f, 9.0f, 12.0f }));
            Assert.That(context.DeviceName, Does.Contain("NVIDIA"));
        });
    }

    [Test]
    public void ZipInPlace_GpuHandlesEmptyArrays()
    {
        using ComputeContext context = CreateCudaContext();
        float[] target = [];
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context
        };

        float[] result = Compute.ZipInPlace(
            target,
            [],
            (left, right) => left + right,
            options);

        Assert.That(result, Is.SameAs(target));
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

    private static float[] CreateRight(int count)
    {
        var right = new float[count];
        for (int index = 0; index < count; index++)
        {
            right[index] = (count - index) / 20_000.0f;
        }

        return right;
    }
}
