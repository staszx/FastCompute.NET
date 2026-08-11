using System.Runtime.Intrinsics.X86;

namespace FastCompute.Tests;

[TestFixture]
public sealed class MapReductionFusionTests
{
    private const int NvidiaAcceleratorIndex = 2;

    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    [TestCase(ComputeBackendKind.Simd)]
    public void FloatPipeline_FusesMapWithEveryReduction(
        ComputeBackendKind backend)
    {
        if (backend == ComputeBackendKind.Simd && !Avx.IsSupported)
        {
            Assert.Ignore("SIMD map-reduction fusion requires AVX.");
        }

        float[] source = Enumerable.Range(1, 1_003)
            .Select(index => (index - 500) / 10f)
            .ToArray();
        float[] original = source.ToArray();
        var options = new ComputeOptions
        {
            Backend = backend,
            MaxDegreeOfParallelism = 2
        };
        ComputePipeline<float> pipeline = source
            .AsCompute(options)
            .Select(value => value * 1.5f)
            .Select(value => value - 2f);
        float[] expected = original
            .Select(value => value * 1.5f - 2f)
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
            Assert.That(source, Is.EqualTo(original));
        });
    }

    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    [TestCase(ComputeBackendKind.Simd)]
    public void TypedPipelines_FuseMapWithReduction(
        ComputeBackendKind backend)
    {
        double[] doubles = Enumerable.Range(1, 257)
            .Select(value => (double)value)
            .ToArray();
        int[] integers = Enumerable.Range(1, 257).ToArray();
        var options = new ComputeOptions
        {
            Backend = backend,
            MaxDegreeOfParallelism = 2
        };

        double doubleSum = doubles
            .AsCompute(options)
            .Select(value => value * 0.5 + 3.0)
            .Sum();
        int integerAverage = integers
            .AsCompute(options)
            .Select(value => value * 3 - 1)
            .Average();

        Assert.Multiple(() =>
        {
            Assert.That(
                doubleSum,
                Is.EqualTo(doubles.Select(value => value * 0.5 + 3.0).Sum())
                    .Within(1e-10));
            Assert.That(
                integerAverage,
                Is.EqualTo(integers.Select(value => value * 3 - 1).Sum() /
                    integers.Length));
        });
    }

    [Test]
    public void EmptyMappedPipeline_PreservesReductionContracts()
    {
        ComputePipeline<float> pipeline = Array.Empty<float>()
            .AsCompute(new ComputeOptions { Backend = ComputeBackendKind.Scalar })
            .Select(value => value + 1f);

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
        });
    }

    [Test]
    public void EmptyMappedPipeline_StillValidatesRecordedExpression()
    {
        ComputePipeline<double> pipeline = Array.Empty<double>()
            .AsCompute(new ComputeOptions { Backend = ComputeBackendKind.Scalar })
            .Select(value => decimal.ToDouble((decimal)value));

        Assert.Throws<GpuExpressionNotSupportedException>(
            () => pipeline.Sum());
    }

    [Test]
    [Category("GPU")]
    public void GpuPipelines_FuseMappedReductionsAcrossChunks()
    {
        ComputeDeviceInfo? device = ComputeContext.GetAccelerators()
            .SingleOrDefault(item => item.Index == NvidiaAcceleratorIndex);
        if (device is null)
        {
            Assert.Ignore(
                $"GPU fusion test requires accelerator index " +
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
            EnableGpuChunking = true,
            GpuChunkElementCount = 257
        };
        float[] floats = Enumerable.Range(1, 1_003)
            .Select(value => (float)value)
            .ToArray();
        double[] doubles = Enumerable.Range(1, 1_003)
            .Select(value => (double)value)
            .ToArray();
        int[] integers = Enumerable.Range(1, 1_003).ToArray();

        float floatSum = floats
            .AsCompute(options)
            .Select(value => value * 2f)
            .Select(value => value + 1f)
            .Sum();
        double doubleMax = doubles
            .AsCompute(options)
            .Select(value => value * 0.25 - 10.0)
            .Max();
        int integerMin = integers
            .AsCompute(options)
            .Select(value => value * 2 - 5)
            .Min();

        Assert.Multiple(() =>
        {
            Assert.That(
                floatSum,
                Is.EqualTo(floats.Select(value => value * 2f + 1f).Sum())
                    .Within(0.1f));
            Assert.That(
                doubleMax,
                Is.EqualTo(doubles.Select(value => value * 0.25 - 10.0).Max()));
            Assert.That(
                integerMin,
                Is.EqualTo(integers.Select(value => value * 2 - 5).Min()));
            Assert.That(context.DeviceName, Does.Contain("NVIDIA"));
        });
    }
}
