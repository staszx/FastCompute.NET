using BenchmarkDotNet.Attributes;

namespace FastCompute.Benchmarks;

public enum SpecificationOperation
{
    SimpleMap,
    SimpleZip,
    MediumZip,
    Sum,
    Min,
    Max
}

/// <summary>
/// Implements the complete comparison matrix requested by the specification.
/// </summary>
[MemoryDiagnoser]
public sealed class SpecificationMatrixBenchmarks : BenchmarkData
{
    private static readonly ComputeOptions SimdOptions =
        new() { Backend = ComputeBackendKind.Simd };

    private float[] _right = null!;
    private ComputeContext _ilgpuCpuContext = null!;
    private ComputeContext _cudaContext = null!;
    private ComputeOptions _ilgpuCpuOptions = null!;
    private ComputeOptions _cudaOptions = null!;

    [ParamsAllValues]
    public SpecificationOperation Operation { get; set; }

    public override void Setup()
    {
        base.Setup();
        _right = new float[Count];
        for (int index = 0; index < _right.Length; index++)
        {
            _right[index] = 1.0f - Source[index];
        }

        IReadOnlyList<ComputeDeviceInfo> devices =
            ComputeContext.GetAccelerators();
        ComputeDeviceInfo cpu = devices.First(
            device => device.AcceleratorType.Contains(
                "CPU",
                StringComparison.OrdinalIgnoreCase));
        ComputeDeviceInfo cuda = devices.First(
            device => device.AcceleratorType.Contains(
                "Cuda",
                StringComparison.OrdinalIgnoreCase));
        _ilgpuCpuContext = ComputeContext.Create(
            new ComputeContextOptions { AcceleratorIndex = cpu.Index });
        _cudaContext = ComputeContext.Create(
            new ComputeContextOptions { AcceleratorIndex = cuda.Index });
        _ilgpuCpuOptions = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = _ilgpuCpuContext
        };
        _cudaOptions = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = _cudaContext
        };
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _cudaContext.Dispose();
        _ilgpuCpuContext.Dispose();
    }

    [Benchmark(Baseline = true)]
    public float ForLoop() => ExecuteLoop(parallel: false);

    [Benchmark]
    public float ParallelFor() => ExecuteLoop(parallel: true);

    [Benchmark]
    public float FastComputeSimd() => ExecuteFastCompute(SimdOptions);

    [Benchmark]
    public float IlgpuCpuAccelerator() =>
        ExecuteFastCompute(_ilgpuCpuOptions);

    [Benchmark]
    public float Cuda() => ExecuteFastCompute(_cudaOptions);

    [Benchmark]
    public float Auto() => ExecuteFastCompute(options: null);

    private float ExecuteLoop(bool parallel)
    {
        if (Operation is SpecificationOperation.Sum or
            SpecificationOperation.Min or
            SpecificationOperation.Max)
        {
            return parallel
                ? ExecuteParallelReduction()
                : ExecuteScalarReduction();
        }

        var result = new float[Source.Length];
        if (parallel)
        {
            Parallel.For(
                0,
                Source.Length,
                index => result[index] = EvaluateElement(index));
        }
        else
        {
            for (int index = 0; index < Source.Length; index++)
            {
                result[index] = EvaluateElement(index);
            }
        }

        return result.Length == 0 ? 0f : result[^1];
    }

    private float ExecuteFastCompute(ComputeOptions? options)
    {
        return Operation switch
        {
            SpecificationOperation.SimpleMap =>
                Compute.Run(
                    Source,
                    value => value * 2.0f + 1.0f,
                    options)[^1],
            SpecificationOperation.SimpleZip =>
                Compute.Zip(
                    Source,
                    _right,
                    (left, right) => left + right,
                    options)[^1],
            SpecificationOperation.MediumZip =>
                Compute.Zip(
                    Source,
                    _right,
                    (left, right) =>
                        left * right +
                        ComputeMath.Abs(left - right),
                    options)[^1],
            SpecificationOperation.Sum => Compute.Sum(Source, options),
            SpecificationOperation.Min => Compute.Min(Source, options),
            SpecificationOperation.Max => Compute.Max(Source, options),
            _ => throw new ArgumentOutOfRangeException()
        };
    }

    private float EvaluateElement(int index)
    {
        float value = Source[index];
        float right = _right[index];
        return Operation switch
        {
            SpecificationOperation.SimpleMap => value * 2.0f + 1.0f,
            SpecificationOperation.SimpleZip => value + right,
            SpecificationOperation.MediumZip =>
                value * right + MathF.Abs(value - right),
            _ => throw new InvalidOperationException()
        };
    }

    private float ExecuteScalarReduction()
    {
        float result = Operation == SpecificationOperation.Sum
            ? 0f
            : Source[0];
        int start = Operation == SpecificationOperation.Sum ? 0 : 1;
        for (int index = start; index < Source.Length; index++)
        {
            result = Operation switch
            {
                SpecificationOperation.Sum => result + Source[index],
                SpecificationOperation.Min =>
                    MathF.Min(result, Source[index]),
                SpecificationOperation.Max =>
                    MathF.Max(result, Source[index]),
                _ => throw new InvalidOperationException()
            };
        }

        return result;
    }

    private float ExecuteParallelReduction()
    {
        int processorCount = Environment.ProcessorCount;
        int chunkSize =
            (Source.Length + processorCount - 1) / processorCount;
        var partials = new float[processorCount];
        Parallel.For(
            0,
            processorCount,
            chunk =>
            {
                int start = chunk * chunkSize;
                int end = Math.Min(start + chunkSize, Source.Length);
                if (start >= end)
                {
                    partials[chunk] =
                        Operation == SpecificationOperation.Min
                            ? float.PositiveInfinity
                            : Operation == SpecificationOperation.Max
                                ? float.NegativeInfinity
                                : 0f;
                    return;
                }

                float partial = Operation == SpecificationOperation.Sum
                    ? 0f
                    : Source[start++];
                for (int index = start; index < end; index++)
                {
                    partial = Operation switch
                    {
                        SpecificationOperation.Sum =>
                            partial + Source[index],
                        SpecificationOperation.Min =>
                            MathF.Min(partial, Source[index]),
                        SpecificationOperation.Max =>
                            MathF.Max(partial, Source[index]),
                        _ => throw new InvalidOperationException()
                    };
                }

                partials[chunk] = partial;
            });

        float result = Operation == SpecificationOperation.Sum
            ? 0f
            : partials[0];
        int first = Operation == SpecificationOperation.Sum ? 0 : 1;
        for (int index = first; index < partials.Length; index++)
        {
            result = Operation switch
            {
                SpecificationOperation.Sum => result + partials[index],
                SpecificationOperation.Min =>
                    MathF.Min(result, partials[index]),
                SpecificationOperation.Max =>
                    MathF.Max(result, partials[index]),
                _ => throw new InvalidOperationException()
            };
        }

        return result;
    }
}

/// <summary>
/// Compares heavy transcendental Map on the backends that support its
/// expression. Forced SIMD is intentionally absent because that backend
/// rejects transcendental functions instead of silently falling back.
/// </summary>
[MemoryDiagnoser]
public sealed class SpecificationHeavyMapBenchmarks : BenchmarkData
{
    private ComputeContext _ilgpuCpuContext = null!;
    private ComputeContext _cudaContext = null!;
    private ComputeOptions _ilgpuCpuOptions = null!;
    private ComputeOptions _cudaOptions = null!;

    public override void Setup()
    {
        base.Setup();
        IReadOnlyList<ComputeDeviceInfo> devices =
            ComputeContext.GetAccelerators();
        ComputeDeviceInfo cpu = devices.First(
            device => device.AcceleratorType.Contains(
                "CPU",
                StringComparison.OrdinalIgnoreCase));
        ComputeDeviceInfo cuda = devices.First(
            device => device.AcceleratorType.Contains(
                "Cuda",
                StringComparison.OrdinalIgnoreCase));
        _ilgpuCpuContext = ComputeContext.Create(
            new ComputeContextOptions { AcceleratorIndex = cpu.Index });
        _cudaContext = ComputeContext.Create(
            new ComputeContextOptions { AcceleratorIndex = cuda.Index });
        _ilgpuCpuOptions = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = _ilgpuCpuContext
        };
        _cudaOptions = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = _cudaContext
        };
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _cudaContext.Dispose();
        _ilgpuCpuContext.Dispose();
    }

    [Benchmark(Baseline = true)]
    public float[] ForLoop()
    {
        var result = new float[Source.Length];
        for (int index = 0; index < Source.Length; index++)
        {
            float value = Source[index];
            result[index] =
                MathF.Sin(value) * MathF.Exp(-value * value);
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
                float value = Source[index];
                result[index] =
                    MathF.Sin(value) * MathF.Exp(-value * value);
            });
        return result;
    }

    [Benchmark]
    public float[] IlgpuCpuAccelerator() =>
        Execute(_ilgpuCpuOptions);

    [Benchmark]
    public float[] Cuda() => Execute(_cudaOptions);

    [Benchmark]
    public float[] Auto() => Execute(options: null);

    private float[] Execute(ComputeOptions? options) =>
        Compute.Run(
            Source,
            value =>
                ComputeMath.Sin(value) *
                ComputeMath.Exp(-value * value),
            options);
}

[MemoryDiagnoser]
public sealed class KernelLifecycleBenchmarks
{
    private const int NvidiaAcceleratorIndex = 2;
    private float[] _source = null!;
    private ComputeContext _warmContext = null!;
    private ComputeOptions _warmOptions = null!;

    [Params(1_000_000)]
    public int Count { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _source = Enumerable.Range(0, Count)
            .Select(index => index / (float)Count)
            .ToArray();
        _warmContext = ComputeContext.Create(
            new ComputeContextOptions
            {
                AcceleratorIndex = NvidiaAcceleratorIndex
            });
        _warmContext.Precompile<float>(
            value => ComputeMath.Sin(value) * 2.0f);
        _warmOptions = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = _warmContext
        };
    }

    [GlobalCleanup]
    public void Cleanup() => _warmContext.Dispose();

    [Benchmark]
    public float[] FirstRunWithContextAndKernelCompilation()
    {
        using ComputeContext context = ComputeContext.Create(
            new ComputeContextOptions
            {
                AcceleratorIndex = NvidiaAcceleratorIndex
            });
        return Compute.Run(
            _source,
            value => ComputeMath.Sin(value) * 2.0f,
            new ComputeOptions
            {
                Backend = ComputeBackendKind.Gpu,
                GpuContext = context
            });
    }

    [Benchmark]
    public float[] RepeatedRunWithWarmKernel() =>
        Compute.Run(
            _source,
            value => ComputeMath.Sin(value) * 2.0f,
            _warmOptions);
}

[MemoryDiagnoser]
public sealed class MemoryPoolLimitBenchmarks
{
    private const int NvidiaAcceleratorIndex = 2;
    private float[] _source = null!;
    private ComputeContext _pooledContext = null!;
    private ComputeContext _unpooledContext = null!;
    private ComputeOptions _pooledOptions = null!;
    private ComputeOptions _unpooledOptions = null!;

    [Params(1_000_000, 10_000_000)]
    public int Count { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _source = new float[Count];
        _pooledContext = ComputeContext.Create(
            new ComputeContextOptions
            {
                AcceleratorIndex = NvidiaAcceleratorIndex
            });
        _unpooledContext = ComputeContext.Create(
            new ComputeContextOptions
            {
                AcceleratorIndex = NvidiaAcceleratorIndex,
                MemoryPoolLimitBytes = 0
            });
        _pooledContext.Precompile<float>(value => value * 2.0f);
        _unpooledContext.Precompile<float>(value => value * 2.0f);
        _pooledOptions = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = _pooledContext
        };
        _unpooledOptions = new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            GpuContext = _unpooledContext
        };
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _pooledContext.Dispose();
        _unpooledContext.Dispose();
    }

    [Benchmark(Baseline = true)]
    public float[] WithMemoryPool() =>
        Compute.Run(
            _source,
            value => value * 2.0f,
            _pooledOptions);

    [Benchmark]
    public float[] WithoutRetainedMemoryPool() =>
        Compute.Run(
            _source,
            value => value * 2.0f,
            _unpooledOptions);
}
