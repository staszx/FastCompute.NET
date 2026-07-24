using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using FastCompute.Diagnostics;

namespace FastCompute.Tests;

public sealed class InPlaceComputeTests
{
    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    public void RunInPlace_CpuBackendMatchesOutOfPlaceAndReturnsSource(
        ComputeBackendKind backend)
    {
        float[] source = CreateSource(25_003);
        float[] original = (float[])source.Clone();
        var options = new ComputeOptions
        {
            Backend = backend,
            MaxDegreeOfParallelism = 2
        };
        float[] expected = Compute.Run(
            original,
            value => GpuMath.Sin(value) * 2.0f + 1.0f,
            new ComputeOptions { Backend = ComputeBackendKind.Scalar });

        float[] result = Compute.RunInPlace(
            source,
            value => GpuMath.Sin(value) * 2.0f + 1.0f,
            options);

        Assert.Multiple(() =>
        {
            Assert.That(result, Is.SameAs(source));
            Assert.That(source, Is.EqualTo(expected).Within(1e-6f));
        });
    }

    [Test]
    public void RunInPlace_SimdMatchesOutOfPlaceAndProcessesTail()
    {
        RequireAvx();
        int count = Vector256<float>.Count * 125 + 3;
        float[] source = CreateSource(count);
        float[] original = (float[])source.Clone();
        var options = new ComputeOptions { Backend = ComputeBackendKind.Simd };
        float[] expected = Compute.Run(
            original,
            value => GpuMath.Clamp(
                GpuMath.Abs(value) * 1.5f + 0.25f,
                0.1f,
                7.0f),
            new ComputeOptions { Backend = ComputeBackendKind.Scalar });

        float[] result = Compute.RunInPlace(
            source,
            value => GpuMath.Clamp(
                GpuMath.Abs(value) * 1.5f + 0.25f,
                0.1f,
                7.0f),
            options);

        Assert.Multiple(() =>
        {
            Assert.That(result, Is.SameAs(source));
            Assert.That(source, Is.EqualTo(expected).Within(1e-6f));
        });
    }

    [Test]
    public void RunInPlace_HandlesEmptyAndSingleElementArrays()
    {
        float[] empty = [];
        float[] single = [3.0f];

        Assert.Multiple(() =>
        {
            Assert.That(
                Compute.RunInPlace(
                    empty,
                    value => value * 2.0f,
                    new ComputeOptions
                    {
                        Backend = ComputeBackendKind.Scalar
                    }),
                Is.SameAs(empty));
            Assert.That(
                Compute.RunInPlace(
                    single,
                    value => value * 2.0f,
                    new ComputeOptions
                    {
                        Backend = ComputeBackendKind.ParallelCpu,
                        MaxDegreeOfParallelism = 2
                    }),
                Is.SameAs(single));
            Assert.That(single, Is.EqualTo(new[] { 6.0f }));
        });
    }

    [Test]
    public void RunInPlaceWithDiagnostics_AutoSelectsSimdAndReportsInPlace()
    {
        RequireAvx();
        float[] source = CreateSource(32);
        var options = new ComputeOptions
        {
            Thresholds = new ComputeThresholdOptions
            {
                SimdThreshold = 16,
                ParallelThreshold = 16
            }
        };

        ComputeResult<float[]> result = Compute.RunInPlaceWithDiagnostics(
            source,
            value => value * 2.0f + 1.0f,
            options);

        Assert.Multiple(() =>
        {
            Assert.That(result.Value, Is.SameAs(source));
            Assert.That(
                result.Diagnostics.Backend,
                Is.EqualTo(ComputeBackendKind.Simd));
            Assert.That(result.Diagnostics.IsInPlace, Is.True);
            Assert.That(
                result.Diagnostics.BackendSelectionReason,
                Does.Contain("SIMD selected"));
        });
    }

    [Test]
    public void RunInPlace_AutoSkipsUnsupportedSimdExpression()
    {
        float[] source = CreateSource(32);
        var options = new ComputeOptions
        {
            MaxDegreeOfParallelism = 2,
            Thresholds = new ComputeThresholdOptions
            {
                SimdThreshold = 0,
                ParallelThreshold = 0
            }
        };

        ComputeResult<float[]> result = Compute.RunInPlaceWithDiagnostics(
            source,
            value => GpuMath.Sin(value),
            options);

        Assert.That(
            result.Diagnostics.Backend,
            Is.EqualTo(ComputeBackendKind.ParallelCpu));
    }

    [Test]
    public void RunInPlace_RejectsGpuBackend()
    {
        var options = new ComputeOptions { Backend = ComputeBackendKind.Gpu };

        ComputeBackendNotSupportedException exception =
            Assert.Throws<ComputeBackendNotSupportedException>(
                () => Compute.RunInPlace(
                    [1.0f],
                    value => value + 1.0f,
                    options))!;

        Assert.Multiple(() =>
        {
            Assert.That(
                exception.Backend,
                Is.EqualTo(ComputeBackendKind.Gpu));
            Assert.That(exception.Operation, Does.Contain("in-place"));
        });
    }

    [Test]
    public void RunInPlace_ForcedSimdRejectsUnsupportedExpression()
    {
        RequireAvx();
        var options = new ComputeOptions { Backend = ComputeBackendKind.Simd };

        ComputeBackendUnavailableException exception =
            Assert.Throws<ComputeBackendUnavailableException>(
                () => Compute.RunInPlace(
                    [1.0f],
                    value => GpuMath.Sin(value),
                    options))!;

        Assert.That(
            exception.Backend,
            Is.EqualTo(ComputeBackendKind.Simd));
    }

    [Test]
    public void RunInPlace_ValidatesArgumentsAndCancellationBeforeMutation()
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        float[] source = [1.0f, 2.0f];

        Assert.Multiple(() =>
        {
            Assert.Throws<ArgumentNullException>(
                () => Compute.RunInPlace(null!, value => value));
            Assert.Throws<ArgumentNullException>(
                () => Compute.RunInPlace([1.0f], null!));
            Assert.Throws<OperationCanceledException>(
                () => Compute.RunInPlace(
                    source,
                    value => value + 1.0f,
                    new ComputeOptions
                    {
                        CancellationToken = cancellation.Token
                    }));
            Assert.That(source, Is.EqualTo(new[] { 1.0f, 2.0f }));
        });
    }

    [Test]
    public void Run_RemainsOutOfPlace()
    {
        float[] source = [1.0f, 2.0f];

        float[] result = Compute.Run(source, value => value + 1.0f);

        Assert.Multiple(() =>
        {
            Assert.That(source, Is.EqualTo(new[] { 1.0f, 2.0f }));
            Assert.That(result, Is.Not.SameAs(source));
        });
    }

    private static void RequireAvx()
    {
        if (!Avx.IsSupported)
        {
            Assert.Ignore("SIMD in-place tests require AVX support.");
        }
    }

    private static float[] CreateSource(int count)
    {
        var source = new float[count];
        for (int index = 0; index < source.Length; index++)
        {
            source[index] = (index - count / 2) / 100.0f;
        }

        return source;
    }
}
