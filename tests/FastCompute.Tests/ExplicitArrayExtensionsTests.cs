using System.Runtime.Intrinsics.X86;
using FastCompute.Diagnostics;

namespace FastCompute.Tests;

public sealed class ExplicitArrayExtensionsTests
{
    private const int NvidiaAcceleratorIndex = 2;

    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    public void RunExplicit_UsesRequestedCpuBackend(
        ComputeBackendKind backend)
    {
        float[] source = [1.0f, 2.0f, 3.0f];
        var options = new ComputeOptions
        {
            Backend = backend,
            MaxDegreeOfParallelism = 2
        };

        ComputeResult<float[]> result =
            source.RunExplicitWithDiagnostics(
                value => value * 2.0f + 1.0f,
                options);

        Assert.Multiple(() =>
        {
            Assert.That(result.Value, Is.EqualTo(new[] { 3.0f, 5.0f, 7.0f }));
            Assert.That(result.Diagnostics.Backend, Is.EqualTo(backend));
            Assert.That(
                result.Diagnostics.BackendSelectionReason,
                Does.Contain("explicitly requested"));
        });
    }

    [Test]
    public void RunExplicit_UsesRequestedSimdBackend()
    {
        if (!Avx.IsSupported)
        {
            Assert.Ignore("Explicit SIMD extension test requires AVX.");
        }

        float[] source = Enumerable.Range(0, 32)
            .Select(value => (float)value)
            .ToArray();

        float[] result = source.RunExplicit(
            value => value * 2.0f + 1.0f,
            ComputeBackendKind.Simd);

        Assert.That(
            result,
            Is.EqualTo(source.Select(value => value * 2.0f + 1.0f)));
    }

    [Test]
    [Category("GPU")]
    public void RunExplicit_UsesRequestedGpuAndContext()
    {
        ComputeDeviceInfo gpuDevice = ComputeContext.GetAccelerators()
            .SingleOrDefault(device => device.Index == NvidiaAcceleratorIndex)
            ?? throw new AssertionException(
                $"Explicit GPU extension test requires accelerator index " +
                $"{NvidiaAcceleratorIndex}.");
        using ComputeContext context = ComputeContext.Create(
            new ComputeContextOptions
            {
                AcceleratorIndex = gpuDevice.Index
            });
        float[] source = [0.25f, 0.5f];
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context
        };

        ComputeResult<float[]> result =
            source.RunExplicitWithDiagnostics(
                value => GpuMath.Sin(value),
                options);

        TestContext.Out.WriteLine(
            $"Explicit accelerator: {result.Diagnostics.DeviceName}");
        Assert.Multiple(() =>
        {
            Assert.That(
                result.Diagnostics.Backend,
                Is.EqualTo(ComputeBackendKind.Gpu));
            Assert.That(
                result.Diagnostics.DeviceName,
                Does.Contain("NVIDIA"));
            Assert.That(
                result.Value[0],
                Is.EqualTo(MathF.Sin(source[0])).Within(2e-4f));
            Assert.That(
                result.Value[1],
                Is.EqualTo(MathF.Sin(source[1])).Within(2e-4f));
        });
    }

    [Test]
    public void RunExplicit_RejectsAuto()
    {
        float[] source = [1.0f];

        Assert.Multiple(() =>
        {
            Assert.Throws<ArgumentException>(
                () => source.RunExplicit(
                    value => value,
                    ComputeBackendKind.Auto));
            Assert.Throws<ArgumentException>(
                () => source.RunExplicit(
                    value => value,
                    new ComputeOptions()));
        });
    }

    [Test]
    public void RunExplicit_DoesNotFallbackFromUnsupportedBackend()
    {
        float[] source = [1.0f];

        ComputeBackendUnavailableException exception =
            Assert.Throws<ComputeBackendUnavailableException>(
                () => source.RunExplicit(
                    value => GpuMath.Sin(value),
                    new ComputeOptions
                    {
                        Backend = ComputeBackendKind.Simd,
                        AllowFallback = true
                    }))!;

        Assert.That(
            exception.Backend,
            Is.EqualTo(ComputeBackendKind.Simd));
    }

    [Test]
    public void RunExplicit_RejectsNullArguments()
    {
        float[] source = [1.0f];
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Scalar
        };

        Assert.Multiple(() =>
        {
            Assert.Throws<ArgumentNullException>(
                () => FloatArrayComputeExtensions.RunExplicit(
                    null!,
                    value => value,
                    options));
            Assert.Throws<ArgumentNullException>(
                () => source.RunExplicit(null!, options));
            Assert.Throws<ArgumentNullException>(
                () => source.RunExplicit(
                    value => value,
                    (ComputeOptions)null!));
        });
    }
}
