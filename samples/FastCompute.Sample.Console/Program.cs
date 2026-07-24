using FastCompute;
using FastCompute.Diagnostics;

float[] source = Enumerable.Range(0, 1_000_000)
    .Select(index => index / 100_000.0f)
    .ToArray();

float multiplier = 0.75f;
ComputeResult<float[]> automatic = Compute.RunWithDiagnostics(
    source,
    value => GpuMath.Sin(value * multiplier) * GpuMath.Exp(-value),
    new ComputeOptions
    {
        Backend = ComputeBackendKind.Auto
    });

Console.WriteLine($"Auto backend: {automatic.Diagnostics.Backend}");
Console.WriteLine($"Auto device:  {automatic.Diagnostics.DeviceName ?? "CPU"}");
Console.WriteLine($"First result: {automatic.Value[0]}");

using ComputeContext context = ComputeContext.Create();
Console.WriteLine($"Resident-buffer accelerator: {context.DeviceName}");

IReadOnlyList<ComputeCompilationResult> compilation = context.PrecompileAll();
Console.WriteLine(
    $"Kernel templates prepared: {compilation.Count}; " +
    $"cache hits: {compilation.Count(result => result.CacheHit)}");

using ComputeBuffer<float> input = context.Upload(source);
using ComputeBuffer<float> scaled =
    input.Select(value => value * multiplier);
using ComputeBuffer<float> result = scaled
    .Select(value => GpuMath.Sin(value))
    .Select(value => GpuMath.Clamp(value, 0.0f, 1.0f));

float[] output = new float[result.Length];
result.Download(output);
float residentSum = result.Sum();
Console.WriteLine($"Resident pipeline result count: {output.Length}");
Console.WriteLine($"Resident pipeline sum: {residentSum}");
