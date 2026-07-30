namespace FastCompute.Tests;

[TestFixture]
[Category("GPU")]
[NonParallelizable]
public sealed class PreferredGpuAcceleratorTests
{
    private const int NvidiaAcceleratorIndex = 2;

    [TearDown]
    public void ResetDefaults()
    {
        ComputeDefaults.PreferredGpuAcceleratorIndex = null;
    }

    [Test]
    public void Auto_UsesDefaultPreferredGpuWhenGpuIsBeneficial()
    {
        ComputeDeviceInfo device = GetNvidiaDevice();
        ComputeDefaults.PreferredGpuAcceleratorIndex = device.Index;

        var result = Compute.RunWithDiagnostics(
            CreateSource(4_096),
            value =>
                ComputeMath.Sin(value) *
                ComputeMath.Exp(-value * value),
            new ComputeOptions
            {
                Thresholds = new ComputeThresholdOptions
                {
                    GpuHeavyThreshold = 0
                }
            });

        Assert.Multiple(() =>
        {
            Assert.That(
                result.Diagnostics.Backend,
                Is.EqualTo(ComputeBackendKind.Gpu));
            Assert.That(
                result.Diagnostics.DeviceName,
                Is.EqualTo(device.Name));
        });
    }

    [Test]
    public void OperationPreference_OverridesDefaultPreference()
    {
        ComputeDeviceInfo device = GetNvidiaDevice();
        ComputeDefaults.PreferredGpuAcceleratorIndex = int.MaxValue;

        var result = Compute.RunWithDiagnostics(
            CreateSource(128),
            value => value * 2.0f,
            new ComputeOptions
            {
                Backend = ComputeBackendKind.Gpu,
                PreferredGpuAcceleratorIndex = device.Index
            });

        Assert.That(result.Diagnostics.DeviceName, Is.EqualTo(device.Name));
    }

    [Test]
    public void Auto_UsesPreferredGpuWhenGpuIsBeneficial()
    {
        ComputeDeviceInfo device = GetNvidiaDevice();
        float[] source = CreateSource(4_096);
        var options = new ComputeOptions
        {
            PreferredGpuAcceleratorIndex = device.Index,
            Thresholds = new ComputeThresholdOptions
            {
                GpuHeavyThreshold = 0
            }
        };

        var result = Compute.RunWithDiagnostics(
            source,
            value =>
                ComputeMath.Sin(value) *
                ComputeMath.Exp(-value * value),
            options);

        Assert.Multiple(() =>
        {
            Assert.That(result.Diagnostics.Backend, Is.EqualTo(ComputeBackendKind.Gpu));
            Assert.That(result.Diagnostics.DeviceName, Is.EqualTo(device.Name));
            Assert.That(
                result.Value,
                Is.EqualTo(
                    source.Select(
                        value =>
                            MathF.Sin(value) *
                            MathF.Exp(-value * value)))
                    .Within(1e-5f));
        });
    }

    [Test]
    public void Auto_UsesCpuWhenPreferredGpuIsNotBeneficial()
    {
        ComputeDeviceInfo device = GetNvidiaDevice();
        float[] source = CreateSource(32);

        var result = Compute.RunWithDiagnostics(
            source,
            value => value + 2.0f,
            new ComputeOptions
            {
                PreferredGpuAcceleratorIndex = device.Index
            });

        Assert.Multiple(() =>
        {
            Assert.That(result.Diagnostics.Backend, Is.Not.EqualTo(ComputeBackendKind.Gpu));
            Assert.That(result.Diagnostics.DeviceName, Is.Null);
            Assert.That(
                result.Value,
                Is.EqualTo(source.Select(value => value + 2.0f)));
        });
    }

    [Test]
    public void Auto_FallsBackToCpuWhenPreferredGpuIsUnavailable()
    {
        float[] source = CreateSource(4_096);
        var options = new ComputeOptions
        {
            PreferredGpuAcceleratorIndex = int.MaxValue,
            Thresholds = new ComputeThresholdOptions
            {
                GpuHeavyThreshold = 0
            }
        };

        var result = Compute.RunWithDiagnostics(
            source,
            value =>
                ComputeMath.Sin(value) *
                ComputeMath.Exp(-value * value),
            options);

        Assert.Multiple(() =>
        {
            Assert.That(result.Diagnostics.Backend, Is.Not.EqualTo(ComputeBackendKind.Gpu));
            Assert.That(result.Diagnostics.DeviceName, Is.Null);
            Assert.That(
                result.Diagnostics.BackendSelectionReason,
                Does.Contain(
                    $"Preferred hardware GPU accelerator index " +
                    $"{int.MaxValue} is unavailable."));
        });
    }

    [Test]
    public void ExplicitGpu_UsesPreferredAccelerator()
    {
        ComputeDeviceInfo device = GetNvidiaDevice();

        var result = Compute.RunWithDiagnostics(
            CreateSource(128),
            value => value * 2.0f,
            new ComputeOptions
            {
                Backend = ComputeBackendKind.Gpu,
                PreferredGpuAcceleratorIndex = device.Index
            });

        Assert.Multiple(() =>
        {
            Assert.That(result.Diagnostics.Backend, Is.EqualTo(ComputeBackendKind.Gpu));
            Assert.That(result.Diagnostics.DeviceName, Is.EqualTo(device.Name));
        });
    }

    [Test]
    public void Options_RejectContextAndPreferredIndexTogether()
    {
        ComputeDeviceInfo device = GetNvidiaDevice();
        using ComputeContext context = ComputeContext.Create(
            new ComputeContextOptions
            {
                AcceleratorIndex = device.Index
            });

        Assert.Throws<ArgumentException>(
            () => Compute.Run(
                CreateSource(8),
                value => value + 1.0f,
                new ComputeOptions
                {
                    GpuContext = context,
                    PreferredGpuAcceleratorIndex = device.Index
                }));
    }

    private static ComputeDeviceInfo GetNvidiaDevice()
    {
        ComputeDeviceInfo device = ComputeContext.GetAccelerators()
            .Single(item => item.Index == NvidiaAcceleratorIndex);
        Assert.That(
            device.AcceleratorType,
            Does.Contain("Cuda").IgnoreCase);
        TestContext.Out.WriteLine(
            $"Preferred GPU accelerator: {device.Name} " +
            $"({device.AcceleratorType}, index {device.Index})");
        return device;
    }

    private static float[] CreateSource(int count)
    {
        var source = new float[count];
        for (int index = 0; index < source.Length; index++)
        {
            source[index] = (index % 1_000) / 1_000.0f;
        }

        return source;
    }
}
