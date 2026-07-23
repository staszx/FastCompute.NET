using System.Linq.Expressions;

namespace FastCompute.Tests;

public sealed class GpuMathTests
{
    private const int NvidiaAcceleratorIndex = 2;

    [Test]
    [Category("GPU")]
    public void Run_SupportsAllGpuMathFunctions_OnGpuBackend()
    {
        ComputeDeviceInfo gpuDevice = ComputeContext.GetAccelerators()
            .SingleOrDefault(device => device.Index == NvidiaAcceleratorIndex)
            ?? throw new AssertionException(
                $"GpuMathTests requires accelerator index {NvidiaAcceleratorIndex}.");

        Assert.That(
            gpuDevice.AcceleratorType,
            Does.Contain("Cuda").IgnoreCase,
            $"Accelerator {NvidiaAcceleratorIndex} must be NVIDIA CUDA.");

        using ComputeContext context = ComputeContext.Create(
            new ComputeContextOptions { AcceleratorIndex = gpuDevice.Index });
        float[] source = [0.25f];
        string acceleratorMessage =
            $"Selected accelerator: {gpuDevice.Name} ({gpuDevice.AcceleratorType})";
        TestContext.Out.WriteLine(acceleratorMessage);

        Assert.Multiple(() =>
        {
            AssertGpuResult(context, source, value => GpuMath.Abs(-value), MathF.Abs(-0.25f));
            AssertGpuResult(context, source, value => GpuMath.Min(value, 0.1f), MathF.Min(0.25f, 0.1f));
            AssertGpuResult(context, source, value => GpuMath.Max(value, 0.5f), MathF.Max(0.25f, 0.5f));
            AssertGpuResult(context, source, value => GpuMath.Clamp(value, 0.3f, 0.8f), 0.3f);
            AssertGpuResult(context, source, value => GpuMath.Sqrt(value), MathF.Sqrt(0.25f));
            AssertGpuResult(context, source, value => GpuMath.Sin(value), MathF.Sin(0.25f));
            AssertGpuResult(context, source, value => GpuMath.Cos(value), MathF.Cos(0.25f));
            AssertGpuResult(context, source, value => GpuMath.Tan(value), MathF.Tan(0.25f));
            AssertGpuResult(context, source, value => GpuMath.Exp(value), MathF.Exp(0.25f));
            AssertGpuResult(context, source, value => GpuMath.Log(value), MathF.Log(0.25f));
            AssertGpuResult(context, source, value => GpuMath.Log10(value), MathF.Log10(0.25f));
            AssertGpuResult(context, source, value => GpuMath.Pow(value, 2.0f), MathF.Pow(0.25f, 2.0f));
            AssertGpuResult(context, source, value => GpuMath.Floor(value), MathF.Floor(0.25f));
            AssertGpuResult(context, source, value => GpuMath.Ceiling(value), MathF.Ceiling(0.25f));
            AssertGpuResult(context, source, value => GpuMath.Round(value), MathF.Round(0.25f));
        });
    }

    private static void AssertGpuResult(
        ComputeContext context,
        float[] source,
        Expression<Func<float, float>> expression,
        float expected)
    {
        float[] result = context.Run(source, expression);
        Assert.That(
            result[0],
            Is.EqualTo(expected).Within(2e-4f),
            $"{expression} on {context.DeviceName}");
    }
}
