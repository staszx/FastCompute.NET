using System.Runtime.Intrinsics.X86;

namespace FastCompute.Tests;

[TestFixture]
public sealed class ZipReductionFusionTests
{
    private const int NvidiaAcceleratorIndex = 2;

    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    [TestCase(ComputeBackendKind.Simd)]
    public void FloatZipPipeline_FusesEveryReduction(
        ComputeBackendKind backend)
    {
        if (backend == ComputeBackendKind.Simd && !Avx.IsSupported)
        {
            Assert.Ignore("SIMD Zip-reduction fusion requires AVX.");
        }

        float[] left = Enumerable.Range(1, 1_003)
            .Select(value => (value - 500) / 10f)
            .ToArray();
        float[] right = Enumerable.Range(1, 1_003)
            .Select(value => value / 20f)
            .ToArray();
        var options = new ComputeOptions
        {
            Backend = backend,
            MaxDegreeOfParallelism = 2
        };
        ComputePipeline<float> pipeline = left
            .AsCompute(options)
            .Select(value => value * 1.5f)
            .Zip(right, (first, second) => first - second)
            .Select(value => value + 2f);
        float[] expected = left
            .Zip(
                right,
                (first, second) => first * 1.5f - second + 2f)
            .ToArray();

        Assert.Multiple(() =>
        {
            Assert.That(
                pipeline.Sum(),
                Is.EqualTo(expected.Sum()).Within(0.05f));
            Assert.That(pipeline.Min(), Is.EqualTo(expected.Min()));
            Assert.That(pipeline.Max(), Is.EqualTo(expected.Max()));
            Assert.That(
                pipeline.Average(),
                Is.EqualTo(expected.Average()).Within(1e-4f));
        });
    }

    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    [TestCase(ComputeBackendKind.Simd)]
    public void TypedZipPipelines_FuseReductions(
        ComputeBackendKind backend)
    {
        double[] doubles = Enumerable.Range(1, 257)
            .Select(value => (double)value)
            .ToArray();
        double[] doubleRight = Enumerable.Range(1, 257)
            .Select(value => value / 4d)
            .ToArray();
        int[] integers = Enumerable.Range(1, 257).ToArray();
        int[] integerRight = Enumerable.Range(2, 257).ToArray();
        var options = new ComputeOptions
        {
            Backend = backend,
            MaxDegreeOfParallelism = 2
        };

        double doubleSum = doubles
            .AsCompute(options)
            .Select(value => value * 0.5)
            .Zip(doubleRight, (left, right) => left + right)
            .Select(value => value - 2.0)
            .Sum();
        int integerAverage = integers
            .AsCompute(options)
            .Select(value => value * 2)
            .Zip(integerRight, (left, right) => left - right)
            .Select(value => value + 3)
            .Average();

        Assert.Multiple(() =>
        {
            Assert.That(
                doubleSum,
                Is.EqualTo(
                        doubles.Zip(
                            doubleRight,
                            (left, right) => left * 0.5 + right - 2.0)
                        .Sum())
                    .Within(1e-10));
            Assert.That(
                integerAverage,
                Is.EqualTo(
                    integers.Zip(
                            integerRight,
                            (left, right) => left * 2 - right + 3)
                        .Sum() / integers.Length));
        });
    }

    [Test]
    public void EmptyZipPipeline_PreservesValidationAndReductionContracts()
    {
        ComputePipeline<float> pipeline = Array.Empty<float>()
            .AsCompute(new ComputeOptions { Backend = ComputeBackendKind.Scalar })
            .Zip(Array.Empty<float>(), (left, right) => left + right);
        ComputePipeline<float> invalidExpression = Array.Empty<float>()
            .AsCompute(new ComputeOptions { Backend = ComputeBackendKind.Scalar })
            .Zip(
                Array.Empty<float>(),
                (left, right) => MathF.Sin(left) + right);

        Assert.Multiple(() =>
        {
            Assert.That(pipeline.Sum(), Is.EqualTo(0f));
            Assert.That(
                () => pipeline.Min(),
                Throws.TypeOf<InvalidOperationException>());
            Assert.That(
                () => pipeline.Max(),
                Throws.TypeOf<InvalidOperationException>());
            Assert.That(
                () => pipeline.Average(),
                Throws.TypeOf<InvalidOperationException>());
            Assert.That(
                () => invalidExpression.Sum(),
                Throws.TypeOf<GpuExpressionNotSupportedException>());
        });
    }

    [Test]
    public void ZipReduction_ValidatesLengthAndCancellation()
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        ComputePipeline<float> cancelled = new[] { 1f }
            .AsCompute(
                new ComputeOptions
                {
                    Backend = ComputeBackendKind.Scalar,
                    CancellationToken = cancellation.Token
                })
            .Zip(new[] { 2f }, (left, right) => left + right);

        Assert.Multiple(() =>
        {
            Assert.That(
                () => new[] { 1f, 2f }
                    .AsCompute()
                    .Zip(new[] { 3f }, (left, right) => left + right)
                    .Sum(),
                Throws.TypeOf<ArgumentException>());
            Assert.That(
                () => cancelled.Sum(),
                Throws.TypeOf<OperationCanceledException>());
        });
    }

    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    [TestCase(ComputeBackendKind.Simd)]
    public void ZipMinMax_PropagateNaN(ComputeBackendKind backend)
    {
        if (backend == ComputeBackendKind.Simd && !Avx.IsSupported)
        {
            Assert.Ignore("SIMD Zip-reduction fusion requires AVX.");
        }

        var options = new ComputeOptions { Backend = backend };
        float[] left = [1f, 2f, float.NaN, 3f, 4f, 5f, 6f, 7f, 8f];
        float[] right = Enumerable.Repeat(1f, left.Length).ToArray();
        ComputePipeline<float> pipeline = left
            .AsCompute(options)
            .Zip(right, (first, second) => first + second);

        Assert.Multiple(() =>
        {
            Assert.That(float.IsNaN(pipeline.Min()), Is.True);
            Assert.That(float.IsNaN(pipeline.Max()), Is.True);
        });
    }

    [TestCase(false)]
    [TestCase(true)]
    [Category("GPU")]
    public void GpuZipReductions_FuseForAllTypes(bool enableChunking)
    {
        ComputeDeviceInfo? device = ComputeContext.GetAccelerators()
            .SingleOrDefault(item => item.Index == NvidiaAcceleratorIndex);
        if (device is null)
        {
            Assert.Ignore(
                $"GPU Zip-reduction test requires accelerator index " +
                $"{NvidiaAcceleratorIndex}.");
        }

        using ComputeContext context = ComputeContext.Create(
            new ComputeContextOptions
            {
                AcceleratorIndex = NvidiaAcceleratorIndex
            });
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = context,
            EnableGpuChunking = enableChunking,
            GpuChunkElementCount = 257
        };
        float[] floats = Enumerable.Range(1, 1_003)
            .Select(value => (float)value)
            .ToArray();
        float[] floatRight = Enumerable.Repeat(2f, floats.Length).ToArray();
        double[] doubles = Enumerable.Range(1, 1_003)
            .Select(value => (double)value)
            .ToArray();
        double[] doubleRight = Enumerable.Repeat(0.5d, doubles.Length).ToArray();
        int[] integers = Enumerable.Range(1, 1_003).ToArray();
        int[] integerRight = Enumerable.Repeat(3, integers.Length).ToArray();

        float floatSum = floats
            .AsCompute(options)
            .Select(value => value * 2f)
            .Zip(floatRight, (left, right) => left + right)
            .Sum();
        float floatAverage = floats
            .AsCompute(options)
            .Zip(floatRight, (left, right) => left + right)
            .Average();
        double doubleMax = doubles
            .AsCompute(options)
            .Zip(doubleRight, (left, right) => left * right)
            .Max();
        int integerMin = integers
            .AsCompute(options)
            .Zip(integerRight, (left, right) => left - right)
            .Min();

        Assert.Multiple(() =>
        {
            Assert.That(
                floatSum,
                Is.EqualTo(
                        floats.Zip(
                                floatRight,
                                (left, right) => left * 2f + right)
                            .Sum())
                    .Within(0.1f));
            Assert.That(
                floatAverage,
                Is.EqualTo(
                        floats.Zip(
                                floatRight,
                                (left, right) => left + right)
                            .Average())
                    .Within(1e-4f));
            Assert.That(doubleMax, Is.EqualTo(doubles.Max() * 0.5d));
            Assert.That(integerMin, Is.EqualTo(integers.Min() - 3));
            Assert.That(context.DeviceName, Does.Contain("NVIDIA"));
        });
    }
}
