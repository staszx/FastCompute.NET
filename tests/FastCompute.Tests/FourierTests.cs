namespace FastCompute.Tests;

public sealed class FourierTests
{
    private static readonly ComputeBackendKind[] Backends =
    [
        ComputeBackendKind.Scalar,
        ComputeBackendKind.ParallelCpu,
        ComputeBackendKind.Simd,
        ComputeBackendKind.Gpu
    ];

    [TestCaseSource(nameof(Backends))]
    public void Fft_MatchesDirectTransformAndInverseRoundTrip(ComputeBackendKind backend)
    {
        Complex32[] source = Enumerable.Range(0, 16)
            .Select(index => new Complex32(
                MathF.Sin(index * 0.37f),
                MathF.Cos(index * 0.19f) * 0.25f))
            .ToArray();
        Complex32[] expected = DirectTransform(source, FourierDirection.Forward);

        Complex32[] actual = Compute.Fft(source, options: new ComputeOptions { Backend = backend });
        Complex32[] restored = Compute.Fft(actual, FourierDirection.Inverse, new ComputeOptions { Backend = backend });

        AssertComplex(actual, expected, 2e-4f);
        AssertComplex(restored, source, 2e-4f);
        Assert.That(actual, Is.Not.SameAs(source));
    }

    [TestCaseSource(nameof(Backends))]
    public void Fft2D_ProducesBackendParityAndInverseRoundTrip(ComputeBackendKind backend)
    {
        const int width = 8;
        const int height = 4;
        Complex32[] source = Enumerable.Range(0, width * height)
            .Select(index => new Complex32((index * 13 % 29) / 29f, (index * 7 % 17) / 34f))
            .ToArray();
        Complex32[] expected = Compute.Fft2D(
            source,
            width,
            height,
            options: new ComputeOptions { Backend = ComputeBackendKind.Scalar });

        Complex32[] actual = (Complex32[])source.Clone();
        Complex32[] returned = Compute.Fft2DInPlace(
            actual,
            width,
            height,
            options: new ComputeOptions { Backend = backend });
        Assert.That(returned, Is.SameAs(actual));
        AssertComplex(actual, expected, 4e-4f);
        Compute.Fft2DInPlace(
            actual,
            width,
            height,
            FourierDirection.Inverse,
            new ComputeOptions { Backend = backend });

        AssertComplex(actual, source, 4e-4f);
    }

    [Test]
    public void Fft_RejectsNonPowerOfTwoDimensions()
    {
        Assert.Throws<ArgumentException>(() => Compute.Fft(new Complex32[3]));
        Assert.Throws<ArgumentException>(() => Compute.Fft2D(new Complex32[12], 3, 4));
    }

    [TestCaseSource(nameof(Backends))]
    public void SpectrumOperations_ProduceBackendParity(ComputeBackendKind backend)
    {
        Complex32[] spectrum =
        [
            new Complex32(3, 4),
            new Complex32(5, 12),
            new Complex32(8, 15)
        ];
        var options = new ComputeOptions { Backend = backend };

        float[] power = Compute.PowerSpectrum(spectrum, options);
        float[] magnitude = Compute.MagnitudeSpectrum(spectrum, options);

        Assert.Multiple(() =>
        {
            Assert.That(power, Is.EqualTo(new[] { 25f, 169f, 289f }).Within(1e-4f));
            Assert.That(magnitude, Is.EqualTo(new[] { 5f, 13f, 17f }).Within(1e-4f));
        });
    }

    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    [TestCase(ComputeBackendKind.Gpu)]
    public void PhaseSpectrum_ProducesSupportedBackendParity(ComputeBackendKind backend)
    {
        Complex32[] spectrum = [new Complex32(1, 0), new Complex32(0, 1), new Complex32(-1, 0), new Complex32(0, -1)];

        float[] actual = Compute.PhaseSpectrum(spectrum, new ComputeOptions { Backend = backend });

        Assert.That(actual, Is.EqualTo(new[] { 0f, MathF.PI / 2f, MathF.PI, -MathF.PI / 2f }).Within(2e-6f));
    }

    [Test]
    public void PhaseSpectrum_RejectsExplicitSimdInsteadOfUsingScalarFallback() =>
        Assert.Throws<ComputeBackendNotSupportedException>(() =>
            Compute.PhaseSpectrum([new Complex32(1, 1)], new ComputeOptions { Backend = ComputeBackendKind.Simd }));

    private static Complex32[] DirectTransform(Complex32[] source, FourierDirection direction)
    {
        var result = new Complex32[source.Length];
        float sign = direction == FourierDirection.Forward ? -1f : 1f;
        for (int k = 0; k < source.Length; k++)
        {
            Complex32 sum = default;
            for (int n = 0; n < source.Length; n++)
            {
                float angle = sign * 2f * MathF.PI * k * n / source.Length;
                sum += source[n] * new Complex32(MathF.Cos(angle), MathF.Sin(angle));
            }
            result[k] = direction == FourierDirection.Inverse ? sum * (1f / source.Length) : sum;
        }
        return result;
    }

    private static void AssertComplex(Complex32[] returned, Complex32[] expected, float tolerance)
    {
        Assert.That(returned, Has.Length.EqualTo(expected.Length));
        for (int index = 0; index < returned.Length; index++)
        {
            Assert.Multiple(() =>
            {
                Assert.That(returned[index].Real, Is.EqualTo(expected[index].Real).Within(tolerance), $"real[{index}]");
                Assert.That(returned[index].Imaginary, Is.EqualTo(expected[index].Imaginary).Within(tolerance), $"imaginary[{index}]");
            });
        }
    }
}
