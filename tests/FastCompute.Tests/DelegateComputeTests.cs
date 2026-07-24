using System.Collections.Concurrent;
using FastCompute.Diagnostics;

namespace FastCompute.Tests;

public sealed class DelegateComputeTests
{
    [Test]
    public void RunDelegate_ExecutesArbitraryMethodWithoutChangingSource()
    {
        float[] source = [1.0f, 2.0f, 3.0f];
        var calculation = new CustomCalculation(2.0f);

        float[] result = Compute.RunDelegate(
            source,
            value => calculation.Transform(value),
            new ComputeOptions { Backend = ComputeBackendKind.Scalar });

        Assert.Multiple(() =>
        {
            Assert.That(result, Is.EqualTo(new[] { 3.0f, 5.0f, 7.0f }));
            Assert.That(source, Is.EqualTo(new[] { 1.0f, 2.0f, 3.0f }));
            Assert.That(result, Is.Not.SameAs(source));
        });
    }

    [Test]
    public void RunDelegate_ParallelExecutesEveryElement()
    {
        float[] source = Enumerable.Range(0, 25_003)
            .Select(value => (float)value)
            .ToArray();
        var visited = new ConcurrentDictionary<int, byte>();
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.ParallelCpu,
            MaxDegreeOfParallelism = 2
        };

        float[] result = Compute.RunDelegate(
            source,
            value =>
            {
                visited.TryAdd((int)value, 0);
                return CustomMethod(value);
            },
            options);

        Assert.Multiple(() =>
        {
            Assert.That(visited.Count, Is.EqualTo(source.Length));
            Assert.That(result[0], Is.EqualTo(CustomMethod(source[0])));
            Assert.That(result[^1], Is.EqualTo(CustomMethod(source[^1])));
        });
    }

    [TestCase(9, ComputeBackendKind.Scalar)]
    [TestCase(10, ComputeBackendKind.ParallelCpu)]
    public void RunDelegateWithDiagnostics_AutoUsesOnlyCpuBackends(
        int elementCount,
        ComputeBackendKind expectedBackend)
    {
        var options = new ComputeOptions
        {
            MaxDegreeOfParallelism = 2,
            Thresholds = new ComputeThresholdOptions
            {
                ParallelThreshold = 10,
                SimdThreshold = 0,
                GpuSimpleThreshold = 0
            }
        };

        ComputeResult<float[]> result = Compute.RunDelegateWithDiagnostics(
            new float[elementCount],
            CustomMethod,
            options);

        Assert.Multiple(() =>
        {
            Assert.That(result.Diagnostics.Backend, Is.EqualTo(expectedBackend));
            Assert.That(
                result.Diagnostics.BackendSelectionReason,
                Does.Contain("delegates are CPU-only"));
            Assert.That(result.Diagnostics.CompilationTime, Is.EqualTo(TimeSpan.Zero));
            Assert.That(result.Diagnostics.DeviceName, Is.Null);
        });
    }

    [TestCase(ComputeBackendKind.Simd)]
    [TestCase(ComputeBackendKind.Gpu)]
    public void RunDelegate_RejectsNonCpuBackend(ComputeBackendKind backend)
    {
        var options = new ComputeOptions { Backend = backend };

        ComputeBackendNotSupportedException exception =
            Assert.Throws<ComputeBackendNotSupportedException>(
                () => Compute.RunDelegate([1.0f], CustomMethod, options))!;

        Assert.Multiple(() =>
        {
            Assert.That(exception.Backend, Is.EqualTo(backend));
            Assert.That(exception.Operation, Does.Contain("delegates"));
            Assert.That(exception.Message, Does.Contain("Scalar"));
            Assert.That(exception.Message, Does.Contain("ParallelCpu"));
        });
    }

    [Test]
    public void RunDelegate_DoesNotSuppressUserException()
    {
        var expected = new CustomDelegateException();

        CustomDelegateException actual =
            Assert.Throws<CustomDelegateException>(
                () => Compute.RunDelegate(
                    [1.0f],
                    _ => throw expected,
                    new ComputeOptions
                    {
                        Backend = ComputeBackendKind.Scalar
                    }))!;

        Assert.That(actual, Is.SameAs(expected));
    }

    [Test]
    public void RunDelegate_ParallelPropagatesUserException()
    {
        var expected = new CustomDelegateException();

        AggregateException actual =
            Assert.Throws<AggregateException>(
                () => Compute.RunDelegate(
                    new float[10_000],
                    _ => throw expected,
                    new ComputeOptions
                    {
                        Backend = ComputeBackendKind.ParallelCpu,
                        MaxDegreeOfParallelism = 2
                    }))!;

        Assert.That(
            actual.Flatten().InnerExceptions,
            Has.Some.SameAs(expected));
    }

    [Test]
    public void RunDelegate_ValidatesArgumentsAndCancellation()
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();

        Assert.Multiple(() =>
        {
            Assert.Throws<ArgumentNullException>(
                () => Compute.RunDelegate(null!, CustomMethod));
            Assert.Throws<ArgumentNullException>(
                () => Compute.RunDelegate([1.0f], null!));
            Assert.Throws<OperationCanceledException>(
                () => Compute.RunDelegate(
                    [1.0f],
                    CustomMethod,
                    new ComputeOptions
                    {
                        CancellationToken = cancellation.Token
                    }));
        });
    }

    private static float CustomMethod(float value) =>
        MathF.Sin(value) + value * 2.0f;

    private sealed class CustomCalculation(float offset)
    {
        public float Transform(float value) =>
            value * 2.0f + offset - 1.0f;
    }

    private sealed class CustomDelegateException : Exception;
}
