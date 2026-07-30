namespace FastCompute.Tests;

[TestFixture]
[Category("GPU")]
public sealed class GpuComputePipelineTests
{
    private const int NvidiaAcceleratorIndex = 2;

    [Test]
    public void ToArray_ExecutesFusedPipelineOnExplicitGpu()
    {
        ComputeDeviceInfo device = ComputeContext.GetAccelerators()
            .Single(item => item.Index == NvidiaAcceleratorIndex);
        Assert.That(
            device.AcceleratorType,
            Does.Contain("Cuda").IgnoreCase);
        TestContext.Out.WriteLine(
            $"Pipeline accelerator: {device.Name} " +
            $"({device.AcceleratorType}, index {device.Index})");

        using ComputeContext context = ComputeContext.Create(
            new ComputeContextOptions
            {
                AcceleratorIndex = device.Index
            });
        float[] source = Enumerable.Range(0, 4_096)
            .Select(index => index / 1_000.0f)
            .ToArray();

        float[] result = source
            .AsCompute(
                new ComputeOptions
                {
                    Backend = ComputeBackendKind.Gpu,
                    GpuContext = context
                })
            .Select(value => value * 0.75f)
            .SelectInPlace(value => GpuMath.Sin(value))
            .Select(value => GpuMath.Clamp(value, -0.5f, 0.5f))
            .ToArray();

        Assert.That(
            result,
            Is.EqualTo(
                    source.Select(
                        value => Math.Clamp(
                            MathF.Sin(value * 0.75f),
                            -0.5f,
                            0.5f)))
                .Within(1e-5f));
    }
}
