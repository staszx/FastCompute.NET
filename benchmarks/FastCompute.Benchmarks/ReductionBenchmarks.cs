using BenchmarkDotNet.Attributes;

namespace FastCompute.Benchmarks;

[MemoryDiagnoser]
public class ReductionBenchmarks : BenchmarkData
{
    private static readonly ComputeOptions ScalarOptions =
        new() { Backend = ComputeBackendKind.Scalar };

    private static readonly ComputeOptions ParallelOptions =
        new() { Backend = ComputeBackendKind.ParallelCpu };

    private static readonly ComputeOptions SimdOptions =
        new() { Backend = ComputeBackendKind.Simd };

    private ComputeContext _gpuContext = null!;
    private ComputeOptions _gpuOptions = null!;

    public override void Setup()
    {
        base.Setup();
        ComputeDeviceInfo cuda = ComputeContext.GetAccelerators()
            .First(device => device.AcceleratorType.Contains(
                "Cuda",
                StringComparison.OrdinalIgnoreCase));
        _gpuContext = ComputeContext.Create(new ComputeContextOptions
        {
            AcceleratorIndex = cuda.Index
        });
        _gpuOptions = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = _gpuContext
        };
    }

    [GlobalCleanup]
    public void Cleanup() => _gpuContext.Dispose();

    [Benchmark(Baseline = true)]
    public float ForLoop()
    {
        float result = 0.0f;
        for (int index = 0; index < Source.Length; index++)
        {
            result += Source[index];
        }

        return result;
    }

    [Benchmark]
    public float FastComputeScalar() =>
        Compute.Sum(Source, ScalarOptions);

    [Benchmark]
    public float FastComputeParallel() =>
        Compute.Sum(Source, ParallelOptions);

    [Benchmark]
    public float FastComputeSimd() =>
        Compute.Sum(Source, SimdOptions);

    [Benchmark]
    public float FastComputeGpu() =>
        Compute.Sum(Source, _gpuOptions);

    [Benchmark]
    public float FastComputeAuto() =>
        Compute.Sum(Source);
}
