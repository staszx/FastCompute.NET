namespace FastCompute.Tests;

[TestFixture]
public sealed class ComputePipelineTests
{
    [Test]
    public void ToArray_FusesRecordedOperations()
    {
        float[] source = [0.0f, 0.25f, 0.5f, 0.75f];
        ComputePipeline<float> pipeline = source
            .AsCompute(
                new ComputeOptions
                {
                    Backend = ComputeBackendKind.Scalar
                })
            .Select(value => value * 2.0f)
            .SelectInPlace(value => value + 1.0f)
            .Select(value => GpuMath.Clamp(value, 0.0f, 2.0f));

        float[] result = pipeline.ToArray();

        Assert.Multiple(() =>
        {
            Assert.That(pipeline.OperationCount, Is.EqualTo(3));
            Assert.That(result, Is.EqualTo(new[] { 1.0f, 1.5f, 2.0f, 2.0f }));
            Assert.That(source, Is.EqualTo(new[] { 0.0f, 0.25f, 0.5f, 0.75f }));
        });
    }

    [Test]
    public void Pipeline_IsLazyUntilTerminalOperation()
    {
        float[] source = [1.0f, 2.0f];
        ComputePipeline<float> pipeline =
            source.AsCompute().Select(value => value * 2.0f);

        source[0] = 5.0f;

        Assert.That(pipeline.ToArray(), Is.EqualTo(new[] { 10.0f, 4.0f }));
    }

    [Test]
    public void ToArrayWithoutOperations_ReturnsIndependentCopy()
    {
        float[] source = [1.0f, 2.0f];

        float[] result = source.AsCompute().ToArray();

        Assert.Multiple(() =>
        {
            Assert.That(result, Is.EqualTo(source));
            Assert.That(result, Is.Not.SameAs(source));
        });
    }

    [Test]
    public void ToArrayInPlace_ReplacesSourceAndReturnsSameArray()
    {
        float[] source = [1.0f, 2.0f, 3.0f];

        float[] result = source
            .AsCompute(
                new ComputeOptions
                {
                    Backend = ComputeBackendKind.Scalar
                })
            .Select(value => value * 2.0f)
            .SelectInPlace(value => value + 1.0f)
            .ToArrayInPlace();

        Assert.Multiple(() =>
        {
            Assert.That(result, Is.SameAs(source));
            Assert.That(source, Is.EqualTo(new[] { 3.0f, 5.0f, 7.0f }));
        });
    }

    [Test]
    public void BranchesDoNotMutateOneAnother()
    {
        float[] source = [1.0f, 2.0f];
        ComputePipeline<float> root = source.AsCompute();
        ComputePipeline<float> doubled =
            root.SelectInPlace(value => value * 2.0f);
        ComputePipeline<float> shifted =
            root.Select(value => value + 10.0f);

        Assert.Multiple(() =>
        {
            Assert.That(doubled.ToArray(), Is.EqualTo(new[] { 2.0f, 4.0f }));
            Assert.That(shifted.ToArray(), Is.EqualTo(new[] { 11.0f, 12.0f }));
            Assert.That(source, Is.EqualTo(new[] { 1.0f, 2.0f }));
        });
    }

    [Test]
    public void DoubleAndIntegerPipelinesUseTypedBackends()
    {
        double[] doubles = [1.0, 2.0, 3.0];
        int[] integers = [1, 2, 3];

        double[] doubleResult = doubles
            .AsCompute(
                new ComputeOptions
                {
                    Backend = ComputeBackendKind.Scalar
                })
            .Select(value => value * 2.0)
            .Select(value => value + 0.5)
            .ToArray();
        int[] integerResult = integers
            .AsCompute(
                new ComputeOptions
                {
                    Backend = ComputeBackendKind.Simd
                })
            .Select(value => value * 2)
            .SelectInPlace(value => value + 1)
            .ToArray();

        Assert.Multiple(() =>
        {
            Assert.That(doubleResult, Is.EqualTo(new[] { 2.5, 4.5, 6.5 }));
            Assert.That(integerResult, Is.EqualTo(new[] { 3, 5, 7 }));
        });
    }

    [Test]
    public void ReductionTerminalsUsePipelineResult()
    {
        ComputePipeline<float> pipeline = new[] { 1.0f, 2.0f, 3.0f }
            .AsCompute(
                new ComputeOptions
                {
                    Backend = ComputeBackendKind.Scalar
                })
            .Select(value => value * 2.0f);

        Assert.Multiple(() =>
        {
            Assert.That(pipeline.Sum(), Is.EqualTo(12.0f));
            Assert.That(pipeline.Min(), Is.EqualTo(2.0f));
            Assert.That(pipeline.Max(), Is.EqualTo(6.0f));
            Assert.That(pipeline.Average(), Is.EqualTo(4.0f));
        });
    }

    [Test]
    public void InvalidExpressionIsRejectedOnlyAtTerminalOperation()
    {
        ComputePipeline<float> pipeline = new[] { 1.0f }
            .AsCompute()
            .Select(value => MathF.Sin(value));

        Assert.Throws<GpuExpressionNotSupportedException>(
            () => pipeline.ToArray());
    }
}
