using FastCompute.Diagnostics;

namespace FastCompute.Tests;

public sealed class ParallelComputeTests
{
    private static readonly ComputeOptions ParallelOptions = new()
    {
        Backend = ComputeBackendKind.ParallelCpu,
        MaxDegreeOfParallelism = 2
    };

    [Test]
    public void Run_ParallelMatchesScalar()
    {
        float[] source = CreateSource(25_003);
        float[] scalar = Compute.Run(
            source,
            value => GpuMath.Sin(value) * 2.0f - GpuMath.Sqrt(GpuMath.Abs(value)),
            new ComputeOptions { Backend = ComputeBackendKind.Scalar });

        float[] parallel = Compute.Run(
            source,
            value => GpuMath.Sin(value) * 2.0f - GpuMath.Sqrt(GpuMath.Abs(value)),
            ParallelOptions);

        Assert.That(parallel, Is.EqualTo(scalar).Within(1e-6f));
    }

    [Test]
    public void Zip_ParallelMatchesScalar()
    {
        float[] left = CreateSource(25_003);
        float[] right = CreateSource(25_003);

        float[] scalar = Compute.Zip(
            left,
            right,
            (x, y) => x * y + GpuMath.Max(x, y),
            new ComputeOptions { Backend = ComputeBackendKind.Scalar });
        float[] parallel = Compute.Zip(
            left,
            right,
            (x, y) => x * y + GpuMath.Max(x, y),
            ParallelOptions);

        Assert.That(parallel, Is.EqualTo(scalar).Within(1e-6f));
    }

    [Test]
    public void Sum_ParallelUsesChunkedReduction()
    {
        float[] source = Enumerable.Repeat(1.0f, 25_003).ToArray();

        float result = Compute.Sum(source, ParallelOptions);

        Assert.That(result, Is.EqualTo(25_003.0f));
    }

    [Test]
    public void ForcedParallel_HandlesEmptyArrays()
    {
        Assert.Multiple(() =>
        {
            Assert.That(
                Compute.Run([], value => value, ParallelOptions),
                Is.Empty);
            Assert.That(
                Compute.Zip([], [], (left, right) => left + right, ParallelOptions),
                Is.Empty);
            Assert.That(
                Compute.Sum([], ParallelOptions),
                Is.EqualTo(0.0f));
        });
    }

    [Test]
    public void Auto_SelectsParallelAtConfiguredThreshold()
    {
        var options = new ComputeOptions
        {
            MaxDegreeOfParallelism = 2,
            Thresholds = new ComputeThresholdOptions { ParallelThreshold = 10 }
        };

        ComputeResult<float[]> result =
            Compute.RunWithDiagnostics(CreateSource(10), value => value * 2.0f, options);

        Assert.That(result.Diagnostics.Backend, Is.EqualTo(ComputeBackendKind.ParallelCpu));
    }

    [Test]
    public void Auto_SelectsScalarBelowConfiguredThreshold()
    {
        var options = new ComputeOptions
        {
            MaxDegreeOfParallelism = 2,
            Thresholds = new ComputeThresholdOptions { ParallelThreshold = 11 }
        };

        ComputeResult<float[]> result =
            Compute.RunWithDiagnostics(CreateSource(10), value => value * 2.0f, options);

        Assert.That(result.Diagnostics.Backend, Is.EqualTo(ComputeBackendKind.Scalar));
    }

    [Test]
    public void Auto_DoesNotSelectParallelWhenMaximumParallelismIsOne()
    {
        var options = new ComputeOptions
        {
            MaxDegreeOfParallelism = 1,
            Thresholds = new ComputeThresholdOptions { ParallelThreshold = 0 }
        };

        ComputeResult<float[]> result =
            Compute.RunWithDiagnostics(CreateSource(10), value => value, options);

        Assert.That(result.Diagnostics.Backend, Is.EqualTo(ComputeBackendKind.Scalar));
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
