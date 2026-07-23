namespace FastCompute.Tests;

[TestFixture]
[Category("GPU")]
public sealed class GpuReductionTests
{
    private const int NvidiaAcceleratorIndex = 2;

    [Test]
    public void Reductions_MatchScalarAcrossMultipleGpuPasses()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(70_003);
        var scalarOptions = new ComputeOptions
        {
            Backend = ComputeBackendKind.Scalar
        };
        var gpuOptions = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context
        };

        Assert.Multiple(() =>
        {
            Assert.That(
                Compute.Sum(source, gpuOptions),
                Is.EqualTo(Compute.Sum(source, scalarOptions)).Within(2e-2f));
            Assert.That(
                Compute.Min(source, gpuOptions),
                Is.EqualTo(Compute.Min(source, scalarOptions)));
            Assert.That(
                Compute.Max(source, gpuOptions),
                Is.EqualTo(Compute.Max(source, scalarOptions)));
            Assert.That(
                Compute.Average(source, gpuOptions),
                Is.EqualTo(Compute.Average(source, scalarOptions)).Within(1e-5f));
        });
    }

    [Test]
    public void Reductions_HandleSingleElementAndNaN()
    {
        using ComputeContext context = CreateCudaContext();
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context
        };

        Assert.Multiple(() =>
        {
            Assert.That(Compute.Sum([4f], options), Is.EqualTo(4f));
            Assert.That(Compute.Min([4f], options), Is.EqualTo(4f));
            Assert.That(Compute.Max([4f], options), Is.EqualTo(4f));
            Assert.That(Compute.Average([4f], options), Is.EqualTo(4f));
            Assert.That(float.IsNaN(Compute.Min(
                [1f, float.NaN, 2f],
                options)), Is.True);
            Assert.That(float.IsNaN(Compute.Max(
                [1f, float.NaN, 2f],
                options)), Is.True);
        });
    }

    [Test]
    public void PrecompileReduction_CachesReductionTemplate()
    {
        using ComputeContext context = CreateCudaContext();

        ComputeCompilationResult first =
            context.PrecompileReduction<float>(ComputeReductionKind.Sum);
        ComputeCompilationResult second =
            context.PrecompileReduction<float>(ComputeReductionKind.Max);

        Assert.Multiple(() =>
        {
            Assert.That(first.CacheHit, Is.False);
            Assert.That(second.CacheHit, Is.True);
        });
    }

    [Test]
    public void Auto_SelectsGpuUsingExpressionSpecificThresholds()
    {
        using ComputeContext context = CreateCudaContext();
        float[] source = CreateSource(1_024);
        var thresholds = new ComputeThresholdOptions
        {
            SimdThreshold = 0,
            ParallelThreshold = 0,
            GpuSimpleThreshold = 16,
            GpuMediumThreshold = 16,
            GpuHeavyThreshold = 16
        };
        var options = new ComputeOptions
        {
            GpuContext = context,
            Thresholds = thresholds
        };

        var simple = Compute.RunWithDiagnostics(
            source,
            value => value * 2f,
            options);
        var heavy = Compute.RunWithDiagnostics(
            source,
            value => GpuMath.Sin(value),
            options);
        float reduction = Compute.Max(source, options);

        Assert.Multiple(() =>
        {
            Assert.That(simple.Diagnostics.Backend, Is.EqualTo(ComputeBackendKind.Gpu));
            Assert.That(heavy.Diagnostics.Backend, Is.EqualTo(ComputeBackendKind.Gpu));
            Assert.That(simple.Diagnostics.DeviceName, Does.Contain("NVIDIA"));
            Assert.That(reduction, Is.EqualTo(Compute.Max(
                source,
                new ComputeOptions { Backend = ComputeBackendKind.Scalar })));
        });
    }

    private static ComputeContext CreateCudaContext()
    {
        ComputeDeviceInfo device = ComputeContext.GetAccelerators()
            .Single(item => item.Index == NvidiaAcceleratorIndex);
        Assert.That(device.AcceleratorType, Does.Contain("Cuda").IgnoreCase);
        TestContext.Out.WriteLine(
            $"GPU reduction accelerator: {device.Name} ({device.AcceleratorType})");

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
            source[index] = (index % 1_001 - 500) / 10f;
        }

        return source;
    }
}
