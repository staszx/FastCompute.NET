namespace FastCompute.Tests;

[TestFixture]
[Category("GPU")]
public sealed class NumericGpuTests
{
    private const int NvidiaAcceleratorIndex = 2;

    [Test]
    public void DoubleAndInt_ExecuteOnSelectedNvidiaAccelerator()
    {
        using ComputeContext context = CreateCudaContext();
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context
        };

        double[] doubles = Compute.Run(
            [0d, 0.25d, 1d],
            value => Math.Sin(value) * Math.Exp(-value),
            options);
        int[] integers = Compute.Zip(
            [1, 2, 3],
            [4, 5, 6],
            (left, right) => left * right + 2,
            options);

        TestContext.Out.WriteLine(
            $"Numeric accelerator: {context.DeviceName}");
        Assert.Multiple(() =>
        {
            Assert.That(
                doubles,
                Is.EqualTo(
                        new[]
                        {
                            0d,
                            Math.Sin(0.25d) * Math.Exp(-0.25d),
                            Math.Sin(1d) * Math.Exp(-1d)
                        })
                    .Within(1e-12));
            Assert.That(integers, Is.EqualTo(new[] { 6, 12, 20 }));
            Assert.That(
                Compute.Sum(new[] { 1d, 2d, 3d }, options),
                Is.EqualTo(6d));
            Assert.That(
                Compute.Min(new[] { 5, -2, 8 }, options),
                Is.EqualTo(-2));
            Assert.That(context.DeviceName, Does.Contain("NVIDIA"));
        });
    }

    private static ComputeContext CreateCudaContext()
    {
        ComputeDeviceInfo device = ComputeContext.GetAccelerators()
            .Single(item => item.Index == NvidiaAcceleratorIndex);
        Assert.That(
            device.AcceleratorType,
            Does.Contain("Cuda").IgnoreCase);
        return ComputeContext.Create(
            new ComputeContextOptions
            {
                AcceleratorIndex = NvidiaAcceleratorIndex
            });
    }
}
