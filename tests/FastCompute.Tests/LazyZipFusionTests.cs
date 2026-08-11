using System.Runtime.Intrinsics.X86;

namespace FastCompute.Tests;

[TestFixture]
public sealed class LazyZipFusionTests
{
    private const int NvidiaAcceleratorIndex = 2;

    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    [TestCase(ComputeBackendKind.Simd)]
    public void SelectZipSelect_FusesIntoOneBinaryPipeline(
        ComputeBackendKind backend)
    {
        if (backend == ComputeBackendKind.Simd && !Avx.IsSupported)
        {
            Assert.Ignore("SIMD lazy Zip fusion requires AVX.");
        }

        float[] left = Enumerable.Range(1, 1_003)
            .Select(value => (float)value)
            .ToArray();
        float[] right = Enumerable.Range(1, 1_003)
            .Select(value => value / 10f)
            .ToArray();
        float[] originalLeft = left.ToArray();
        var options = new ComputeOptions
        {
            Backend = backend,
            MaxDegreeOfParallelism = 2
        };
        ComputePipeline<float> pipeline = left
            .AsCompute(options)
            .Select(value => value * 2f)
            .Select(value => value + 1f)
            .Zip(right, (first, second) => first - second)
            .Select(value => value * 0.5f)
            .SelectInPlace(value => ComputeMath.Max(value, 0f));

        float[] result = pipeline.ToArray();
        float[] expected = originalLeft
            .Zip(
                right,
                (first, second) =>
                    MathF.Max(((first * 2f + 1f) - second) * 0.5f, 0f))
            .ToArray();

        Assert.Multiple(() =>
        {
            Assert.That(pipeline.OperationCount, Is.EqualTo(5));
            Assert.That(result, Is.EqualTo(expected).Within(1e-5f));
            Assert.That(left, Is.EqualTo(originalLeft));
        });
    }

    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    [TestCase(ComputeBackendKind.Simd)]
    public void TypedZipPipelines_UseNumericBackends(
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
        var options = new ComputeOptions { Backend = backend };

        double[] doubleResult = doubles
            .AsCompute(options)
            .Select(value => value * 0.5)
            .Zip(doubleRight, (left, right) => left + right)
            .Select(value => value - 2.0)
            .ToArray();
        int[] integerResult = integers
            .AsCompute(options)
            .Select(value => value * 2)
            .Zip(integerRight, (left, right) => left - right)
            .Select(value => value + 3)
            .ToArray();

        Assert.Multiple(() =>
        {
            Assert.That(
                doubleResult,
                Is.EqualTo(
                    doubles.Zip(
                        doubleRight,
                        (left, right) => left * 0.5 + right - 2.0))
                    .Within(1e-12));
            Assert.That(
                integerResult,
                Is.EqualTo(
                    integers.Zip(
                        integerRight,
                        (left, right) => left * 2 - right + 3)));
        });
    }

    [Test]
    public void ZipPipeline_RemainsLazyAndValidatesLengthAtTerminal()
    {
        float[] left = [1f, 2f];
        float[] right = [3f, 4f];
        ComputePipeline<float> pipeline = left
            .AsCompute(new ComputeOptions { Backend = ComputeBackendKind.Scalar })
            .Select(value => value * 2f)
            .Zip(right, (first, second) => first + second);

        left[0] = 5f;
        right[1] = 10f;

        Assert.Multiple(() =>
        {
            Assert.That(pipeline.ToArray(), Is.EqualTo(new[] { 13f, 14f }));
            Assert.That(
                () => new[] { 1f, 2f }
                    .AsCompute()
                    .Zip(new[] { 3f }, (first, second) => first + second)
                    .ToArray(),
                Throws.TypeOf<ArgumentException>());
        });
    }

    [Test]
    public void ZipPipeline_ToArrayInPlaceReusesLeftSource()
    {
        float[] left = [1f, 2f, 3f];
        float[] right = [4f, 5f, 6f];

        float[] result = left
            .AsCompute(new ComputeOptions { Backend = ComputeBackendKind.Scalar })
            .Select(value => value * 2f)
            .Zip(right, (first, second) => first + second)
            .Select(value => value - 1f)
            .ToArrayInPlace();

        Assert.Multiple(() =>
        {
            Assert.That(result, Is.SameAs(left));
            Assert.That(left, Is.EqualTo(new[] { 5f, 8f, 11f }));
            Assert.That(right, Is.EqualTo(new[] { 4f, 5f, 6f }));
        });
    }

    [Test]
    public void ZipPipeline_RejectsASecondBinarySource()
    {
        ComputePipeline<float> pipeline = new[] { 1f }
            .AsCompute()
            .Zip(new[] { 2f }, (left, right) => left + right);

        Assert.Throws<NotSupportedException>(
            () => pipeline.Zip(
                new[] { 3f },
                (left, right) => left + right));
    }

    [Test]
    public void ZipPipeline_BranchesRemainIndependent()
    {
        ComputePipeline<float> root = new[] { 1f, 2f }
            .AsCompute(new ComputeOptions { Backend = ComputeBackendKind.Scalar })
            .Select(value => value * 2f)
            .Zip(new[] { 3f, 4f }, (left, right) => left + right);
        ComputePipeline<float> doubled =
            root.Select(value => value * 2f);
        ComputePipeline<float> shifted =
            root.Select(value => value - 1f);

        Assert.Multiple(() =>
        {
            Assert.That(doubled.ToArray(), Is.EqualTo(new[] { 10f, 16f }));
            Assert.That(shifted.ToArray(), Is.EqualTo(new[] { 4f, 7f }));
        });
    }

    [Test]
    public void ZipExpression_IsValidatedOnlyAtTerminalOperation()
    {
        ComputePipeline<float> pipeline = new[] { 1f }
            .AsCompute()
            .Zip(
                new[] { 2f },
                (left, right) => MathF.Sin(left) + right);

        Assert.Throws<GpuExpressionNotSupportedException>(
            () => pipeline.ToArray());
    }

    [Test]
    [Category("GPU")]
    public void GpuZipPipeline_FusesAcrossChunks()
    {
        ComputeDeviceInfo? device = ComputeContext.GetAccelerators()
            .SingleOrDefault(item => item.Index == NvidiaAcceleratorIndex);
        if (device is null)
        {
            Assert.Ignore(
                $"GPU Zip fusion test requires accelerator index " +
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
        float[] left = Enumerable.Range(1, 1_003)
            .Select(value => (float)value)
            .ToArray();
        float[] right = Enumerable.Range(1, 1_003)
            .Select(value => value / 3f)
            .ToArray();

        float[] result = left
            .AsCompute(options)
            .Select(value => value * 0.25f)
            .Zip(right, (first, second) => first + second)
            .Select(value => ComputeMath.Clamp(value, 0f, 400f))
            .ToArray();

        Assert.Multiple(() =>
        {
            Assert.That(
                result,
                Is.EqualTo(
                        left.Zip(
                            right,
                            (first, second) =>
                                Math.Clamp(first * 0.25f + second, 0f, 400f)))
                    .Within(1e-5f));
            Assert.That(context.DeviceName, Does.Contain("NVIDIA"));
        });
    }
}
