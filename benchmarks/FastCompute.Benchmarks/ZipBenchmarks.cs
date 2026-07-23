using BenchmarkDotNet.Attributes;

namespace FastCompute.Benchmarks;

[MemoryDiagnoser]
public class MediumZipBenchmarks : BenchmarkData
{
    private static readonly ComputeOptions ScalarOptions =
        new() { Backend = ComputeBackendKind.Scalar };

    private static readonly ComputeOptions ParallelOptions =
        new() { Backend = ComputeBackendKind.ParallelCpu };

    private float[] _right = null!;

    public override void Setup()
    {
        base.Setup();
        _right = new float[Count];
        for (int index = 0; index < _right.Length; index++)
        {
            _right[index] = 1.0f - Source[index];
        }
    }

    [Benchmark(Baseline = true)]
    public float[] ForLoop()
    {
        var result = new float[Source.Length];
        for (int index = 0; index < Source.Length; index++)
        {
            result[index] = Source[index] * _right[index] +
                            MathF.Sqrt(MathF.Abs(Source[index] - _right[index]));
        }

        return result;
    }

    [Benchmark]
    public float[] ParallelFor()
    {
        var result = new float[Source.Length];
        Parallel.For(
            0,
            Source.Length,
            index =>
            {
                result[index] = Source[index] * _right[index] +
                                MathF.Sqrt(MathF.Abs(Source[index] - _right[index]));
            });
        return result;
    }

    [Benchmark]
    public float[] FastComputeScalar() =>
        Compute.Zip(
            Source,
            _right,
            (left, right) =>
                left * right + GpuMath.Sqrt(GpuMath.Abs(left - right)),
            ScalarOptions);

    [Benchmark]
    public float[] FastComputeParallel() =>
        Compute.Zip(
            Source,
            _right,
            (left, right) =>
                left * right + GpuMath.Sqrt(GpuMath.Abs(left - right)),
            ParallelOptions);

    [Benchmark]
    public float[] FastComputeAuto() =>
        Compute.Zip(
            Source,
            _right,
            (left, right) =>
                left * right + GpuMath.Sqrt(GpuMath.Abs(left - right)));
}
