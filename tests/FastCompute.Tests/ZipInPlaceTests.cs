using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace FastCompute.Tests;

public sealed class ZipInPlaceTests
{
    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    [TestCase(ComputeBackendKind.Simd)]
    public void ZipInPlace_MatchesOutOfPlaceAndReturnsTarget(
        ComputeBackendKind backend)
    {
        if (backend == ComputeBackendKind.Simd && !Avx.IsSupported)
        {
            Assert.Ignore("SIMD in-place Zip tests require AVX support.");
        }

        int count = Vector256<float>.Count * 125 + 3;
        float[] target = CreateSource(count);
        float[] original = (float[])target.Clone();
        float[] right = CreateRight(count);
        float[] expected = Compute.Zip(
            original,
            right,
            (left, value) => left * 2.0f - value + 0.5f,
            new ComputeOptions { Backend = ComputeBackendKind.Scalar });
        var options = new ComputeOptions
        {
            Backend = backend,
            MaxDegreeOfParallelism = 2
        };

        float[] result = Compute.ZipInPlace(
            target,
            right,
            (left, value) => left * 2.0f - value + 0.5f,
            options);

        Assert.Multiple(() =>
        {
            Assert.That(result, Is.SameAs(target));
            Assert.That(target, Is.EqualTo(expected).Within(1e-6f));
        });
    }

    [Test]
    public void ZipInPlace_SupportsAliasedRightArray()
    {
        float[] target = [1.0f, 2.0f, 3.0f];

        float[] result = Compute.ZipInPlace(
            target,
            target,
            (left, right) => left + right,
            new ComputeOptions { Backend = ComputeBackendKind.Scalar });

        Assert.Multiple(() =>
        {
            Assert.That(result, Is.SameAs(target));
            Assert.That(target, Is.EqualTo(new[] { 2.0f, 4.0f, 6.0f }));
        });
    }

    [Test]
    public void ZipInPlaceWithDiagnostics_ReportsInPlaceExecution()
    {
        float[] target = CreateSource(32);
        float[] right = CreateRight(32);

        var result = Compute.ZipInPlaceWithDiagnostics(
            target,
            right,
            (left, value) => left + value,
            new ComputeOptions { Backend = ComputeBackendKind.Scalar });

        Assert.Multiple(() =>
        {
            Assert.That(result.Value, Is.SameAs(target));
            Assert.That(result.Diagnostics.IsInPlace, Is.True);
            Assert.That(
                result.Diagnostics.Backend,
                Is.EqualTo(ComputeBackendKind.Scalar));
        });
    }

    [Test]
    public void ZipInPlace_ValidatesArgumentsLengthsAndCancellation()
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        float[] target = [1.0f, 2.0f];
        float[] right = [3.0f, 4.0f];

        Assert.Multiple(() =>
        {
            Assert.Throws<ArgumentNullException>(
                () => Compute.ZipInPlace(
                    null!,
                    right,
                    (left, value) => left + value));
            Assert.Throws<ArgumentNullException>(
                () => Compute.ZipInPlace(
                    target,
                    null!,
                    (left, value) => left + value));
            Assert.Throws<ArgumentNullException>(
                () => Compute.ZipInPlace(target, right, null!));
            Assert.Throws<ArgumentException>(
                () => Compute.ZipInPlace(
                    target,
                    [1.0f],
                    (left, value) => left + value));
            Assert.Throws<OperationCanceledException>(
                () => Compute.ZipInPlace(
                    target,
                    right,
                    (left, value) => left + value,
                    new ComputeOptions
                    {
                        CancellationToken = cancellation.Token
                    }));
            Assert.That(target, Is.EqualTo(new[] { 1.0f, 2.0f }));
        });
    }

    private static float[] CreateSource(int count)
    {
        var source = new float[count];
        for (int index = 0; index < count; index++)
        {
            source[index] = (index - count / 2) / 100.0f;
        }

        return source;
    }

    private static float[] CreateRight(int count)
    {
        var right = new float[count];
        for (int index = 0; index < count; index++)
        {
            right[index] = (count - index) / 250.0f;
        }

        return right;
    }
}
