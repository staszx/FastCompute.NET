namespace FastCompute.Tests;

[TestFixture]
[Category("GPU")]
public sealed class GpuInPlaceTests
{
    private const int NvidiaAcceleratorIndex = 2;

    [Test]
    public void RunInPlace_GpuMatchesScalarAndUsesOnePooledBuffer()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(1_003);
        float[] original = (float[])source.Clone();
        float[] expected = Compute.Run(
            original,
            value => ComputeMath.Clamp(
                ComputeMath.Sin(value) * ComputeMath.Exp(-value * value) +
                ComputeMath.Sqrt(ComputeMath.Abs(value)),
                -2.0f,
                2.0f),
            new ComputeOptions { Backend = ComputeBackendKind.Scalar });
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context
        };

        var result = Compute.RunInPlaceWithDiagnostics(
            source,
            value => ComputeMath.Clamp(
                ComputeMath.Sin(value) * ComputeMath.Exp(-value * value) +
                ComputeMath.Sqrt(ComputeMath.Abs(value)),
                -2.0f,
                2.0f),
            options);
        ComputeMemoryPoolStatistics pool = context.MemoryPoolStatistics;

        TestContext.Out.WriteLine(
            $"GPU in-place accelerator: {result.Diagnostics.DeviceName}");
        Assert.Multiple(() =>
        {
            Assert.That(result.Value, Is.SameAs(source));
            Assert.That(source, Is.EqualTo(expected).Within(2e-4f));
            Assert.That(
                result.Diagnostics.Backend,
                Is.EqualTo(ComputeBackendKind.Gpu));
            Assert.That(result.Diagnostics.IsInPlace, Is.True);
            Assert.That(
                result.Diagnostics.DeviceName,
                Does.Contain("NVIDIA"));
            Assert.That(
                result.Diagnostics.EstimatedGpuMemoryBytes,
                Is.EqualTo(Compute.EstimateGpuInPlaceWorkingSetBytes(source.Length)));
            Assert.That(
                result.Diagnostics.GpuMemoryBudgetBytes,
                Is.GreaterThan(result.Diagnostics.EstimatedGpuMemoryBytes));
            Assert.That(pool.AllocatedBuffers, Is.EqualTo(1));
            Assert.That(pool.AvailableBuffers, Is.EqualTo(1));
        });
    }

    [Test]
    public void RunInPlace_GpuHandlesEmptyAndSingleElementArrays()
    {
        using ComputeContext context = CreateCudaContext();
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context
        };
        float[] empty = [];
        float[] single = [0.25f];

        float[] emptyResult =
            Compute.RunInPlace(empty, value => ComputeMath.Sin(value), options);
        float[] singleResult =
            Compute.RunInPlace(single, value => ComputeMath.Sin(value), options);

        Assert.Multiple(() =>
        {
            Assert.That(emptyResult, Is.SameAs(empty));
            Assert.That(empty, Is.Empty);
            Assert.That(singleResult, Is.SameAs(single));
            Assert.That(single[0], Is.EqualTo(MathF.Sin(0.25f)).Within(2e-4f));
        });
    }

    [Test]
    public void ComputeContext_RunInPlaceUsesSelectedAccelerator()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = [1.0f, 2.0f, 3.0f];

        float[] result =
            context.RunInPlace(source, value => value * 3.0f - 1.0f);

        Assert.Multiple(() =>
        {
            Assert.That(result, Is.SameAs(source));
            Assert.That(source, Is.EqualTo(new[] { 2.0f, 5.0f, 8.0f }));
            Assert.That(context.DeviceName, Does.Contain("NVIDIA"));
        });
    }

    [Test]
    public void RunInPlace_GpuReusesKernelAndDeviceBuffer()
    {
        using ComputeContext context = CreateCudaContext();
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context
        };
        float[] firstSource = CreateSource(4_096);
        float[] secondSource = CreateSource(4_096);

        var first = Compute.RunInPlaceWithDiagnostics(
            firstSource,
            value => ComputeMath.Sin(value),
            options);
        var second = Compute.RunInPlaceWithDiagnostics(
            secondSource,
            value => ComputeMath.Sin(value),
            options);
        ComputeMemoryPoolStatistics pool = context.MemoryPoolStatistics;

        Assert.Multiple(() =>
        {
            Assert.That(first.Diagnostics.KernelCacheHit, Is.False);
            Assert.That(second.Diagnostics.KernelCacheHit, Is.True);
            Assert.That(pool.AllocatedBuffers, Is.EqualTo(1));
            Assert.That(pool.Rentals, Is.EqualTo(2));
            Assert.That(pool.Reuses, Is.EqualTo(1));
        });
    }

    [Test]
    public void RunInPlace_GpuRejectsWorkingSetAboveBudgetBeforeMutation()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = [1.0f, 2.0f, 3.0f];
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context,
            GpuMemoryBudgetBytes = 1_024,
            EnableGpuChunking = false
        };

        ComputeGpuMemoryBudgetExceededException exception =
            Assert.Throws<ComputeGpuMemoryBudgetExceededException>(
                () => Compute.RunInPlace(
                    source,
                    value => value + 1.0f,
                    options))!;

        Assert.Multiple(() =>
        {
            Assert.That(exception.EstimatedBytes, Is.GreaterThan(1_024));
            Assert.That(exception.BudgetBytes, Is.EqualTo(1_024));
            Assert.That(source, Is.EqualTo(new[] { 1.0f, 2.0f, 3.0f }));
            Assert.That(
                context.MemoryPoolStatistics.AllocatedBuffers,
                Is.Zero);
        });
    }

    [Test]
    public void RunInPlace_AutoSelectsGpuForHeavyExpression()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(300_000);
        var options = new ComputeOptions { GpuContext = context };

        var result = Compute.RunInPlaceWithDiagnostics(
            source,
            value => ComputeMath.Sin(value) * ComputeMath.Exp(-value * value),
            options);

        Assert.Multiple(() =>
        {
            Assert.That(
                result.Diagnostics.Backend,
                Is.EqualTo(ComputeBackendKind.Gpu));
            Assert.That(result.Diagnostics.IsInPlace, Is.True);
            Assert.That(
                result.Diagnostics.BackendSelectionReason,
                Does.Contain("GPU selected"));
        });
    }

    [Test]
    public void RunInPlace_AutoRejectsGpuAboveMemoryBudget()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(1_024);
        var options = new ComputeOptions
        {
            GpuContext = context,
            GpuMemoryBudgetBytes = 1_024,
            Thresholds = new ComputeThresholdOptions
            {
                GpuHeavyThreshold = 0
            }
        };

        var result = Compute.RunInPlaceWithDiagnostics(
            source,
            value => ComputeMath.Sin(value),
            options);

        Assert.Multiple(() =>
        {
            Assert.That(
                result.Diagnostics.Backend,
                Is.EqualTo(ComputeBackendKind.Scalar));
            Assert.That(
                result.Diagnostics.BackendSelectionReason,
                Does.Contain("exceeds"));
            Assert.That(
                result.Diagnostics.EstimatedGpuMemoryBytes,
                Is.GreaterThan(result.Diagnostics.GpuMemoryBudgetBytes));
        });
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
