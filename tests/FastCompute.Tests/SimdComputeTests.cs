using System.Runtime.Intrinsics.X86;
using FastCompute.Diagnostics;

namespace FastCompute.Tests;

[TestFixture]
[Category("SIMD")]
public sealed class SimdComputeTests
{
    private static readonly ComputeOptions SimdOptions = new()
    {
        Backend = ComputeBackendKind.Simd
    };

    [OneTimeSetUp]
    public void RequireAvx()
    {
        if (!Avx.IsSupported)
        {
            Assert.Ignore("SIMD tests require AVX support.");
        }

        TestContext.Out.WriteLine("SIMD accelerator: AVX Vector256<float>");
    }

    [Test]
    public void Run_SimdMatchesScalarAndProcessesTail()
    {
        float[] source = CreateSource(1_003);

        float[] scalar = Compute.Run(
            source,
            value => GpuMath.Clamp(
                GpuMath.Abs(-value) * 1.5f + 0.25f,
                0.1f,
                7f) / 2f,
            new ComputeOptions { Backend = ComputeBackendKind.Scalar });
        float[] simd = Compute.Run(
            source,
            value => GpuMath.Clamp(
                GpuMath.Abs(-value) * 1.5f + 0.25f,
                0.1f,
                7f) / 2f,
            SimdOptions);

        Assert.That(simd, Is.EqualTo(scalar).Within(1e-6f));
    }

    [Test]
    public void Zip_SimdMatchesScalarAndProcessesTail()
    {
        float[] left = CreateSource(1_005);
        float[] right = CreateSource(1_005).Reverse().ToArray();

        float[] scalar = Compute.Zip(
            left,
            right,
            (x, y) => GpuMath.Max(x * y, GpuMath.Min(x + y, 2f)),
            new ComputeOptions { Backend = ComputeBackendKind.Scalar });
        float[] simd = Compute.Zip(
            left,
            right,
            (x, y) => GpuMath.Max(x * y, GpuMath.Min(x + y, 2f)),
            SimdOptions);

        Assert.That(simd, Is.EqualTo(scalar).Within(1e-6f));
    }

    [Test]
    public void Sum_SimdUsesVectorReductionAndScalarTail()
    {
        float[] source = Enumerable.Repeat(1f, 1_003).ToArray();

        float result = Compute.Sum(source, SimdOptions);

        Assert.That(result, Is.EqualTo(1_003f));
    }

    [Test]
    public void Run_SimdHandlesEmptyAndSingleElementArrays()
    {
        Assert.Multiple(() =>
        {
            Assert.That(
                Compute.Run([], value => value * 2f, SimdOptions),
                Is.Empty);
            Assert.That(
                Compute.Run([3f], value => value * 2f, SimdOptions),
                Is.EqualTo(new[] { 6f }));
            Assert.That(Compute.Sum([], SimdOptions), Is.EqualTo(0f));
        });
    }

    [Test]
    public void Run_SimdPreservesSpecialFloatingPointValues()
    {
        float[] source =
        [
            float.NaN,
            float.PositiveInfinity,
            float.NegativeInfinity,
            0f,
            -0f,
            1f,
            -1f,
            2f
        ];

        float[] result = Compute.Run(
            source,
            value => GpuMath.Min(value / 0f, 1f),
            SimdOptions);

        Assert.Multiple(() =>
        {
            Assert.That(float.IsNaN(result[0]), Is.True);
            Assert.That(result[1], Is.EqualTo(1f));
            Assert.That(result[2], Is.EqualTo(float.NegativeInfinity));
            Assert.That(float.IsNaN(result[3]), Is.True);
            Assert.That(float.IsNaN(result[4]), Is.True);
            Assert.That(result[5], Is.EqualTo(1f));
            Assert.That(result[6], Is.EqualTo(float.NegativeInfinity));
            Assert.That(result[7], Is.EqualTo(1f));
        });
    }

    [Test]
    public void MinAndMax_PreserveSignedZeroSemantics()
    {
        float[] left = [-0f, 0f, -0f, 0f, 1f, 1f, 1f, 1f];
        float[] right = [0f, -0f, -0f, 0f, 1f, 1f, 1f, 1f];

        float[] minimum = Compute.Zip(
            left,
            right,
            (x, y) => GpuMath.Min(x, y),
            SimdOptions);
        float[] maximum = Compute.Zip(
            left,
            right,
            (x, y) => GpuMath.Max(x, y),
            SimdOptions);

        Assert.Multiple(() =>
        {
            Assert.That(BitConverter.SingleToInt32Bits(minimum[0]), Is.EqualTo(int.MinValue));
            Assert.That(BitConverter.SingleToInt32Bits(minimum[1]), Is.EqualTo(int.MinValue));
            Assert.That(BitConverter.SingleToInt32Bits(maximum[0]), Is.EqualTo(0));
            Assert.That(BitConverter.SingleToInt32Bits(maximum[1]), Is.EqualTo(0));
        });
    }

    [Test]
    public void ForcedSimd_RejectsUnsupportedExpression()
    {
        ComputeBackendUnavailableException exception =
            Assert.Throws<ComputeBackendUnavailableException>(
                () => Compute.Run(
                    CreateSource(32),
                    value => GpuMath.Sin(value),
                    SimdOptions))!;

        Assert.That(exception.Backend, Is.EqualTo(ComputeBackendKind.Simd));
    }

    [Test]
    public void Clamp_SimdRejectsInvalidBounds()
    {
        Assert.That(
            () => Compute.Run(
                CreateSource(16),
                value => GpuMath.Clamp(value, 2f, 1f),
                SimdOptions),
            Throws.TypeOf<ArgumentException>());
    }

    [Test]
    public void Auto_SelectsSimdForSupportedExpressionAtThreshold()
    {
        var options = new ComputeOptions
        {
            Thresholds = new ComputeThresholdOptions
            {
                SimdThreshold = 16,
                ParallelThreshold = 16
            }
        };

        ComputeResult<float[]> result = Compute.RunWithDiagnostics(
            CreateSource(16),
            value => value * 2f + 1f,
            options);

        Assert.That(result.Diagnostics.Backend, Is.EqualTo(ComputeBackendKind.Simd));
    }

    [Test]
    public void Auto_SkipsSimdForUnsupportedExpression()
    {
        var options = new ComputeOptions
        {
            MaxDegreeOfParallelism = 2,
            Thresholds = new ComputeThresholdOptions
            {
                SimdThreshold = 1,
                ParallelThreshold = 16
            }
        };

        ComputeResult<float[]> result = Compute.RunWithDiagnostics(
            CreateSource(16),
            value => GpuMath.Sin(value),
            options);

        Assert.That(result.Diagnostics.Backend, Is.EqualTo(ComputeBackendKind.ParallelCpu));
    }

    private static float[] CreateSource(int count)
    {
        var source = new float[count];
        for (int index = 0; index < source.Length; index++)
        {
            source[index] = (index - count / 2) / 100f;
        }

        return source;
    }
}
