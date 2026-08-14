using FastCompute;
using FastCompute.ImageProcessing;

byte[] publicKeyToken =
    typeof(Compute).Assembly.GetName().GetPublicKeyToken() ?? [];
Ensure(
    Convert.ToHexString(publicKeyToken)
        .Equals("C76A60C96D65300C", StringComparison.Ordinal),
    "The package assembly does not have the expected strong-name token.");

byte[] imageProcessingToken =
    typeof(Image<>).Assembly.GetName().GetPublicKeyToken() ?? [];
Ensure(
    Convert.ToHexString(imageProcessingToken)
        .Equals("C76A60C96D65300C", StringComparison.Ordinal),
    "The image processing assembly does not have the expected strong-name token.");

float[] source = [0.0f, 0.25f, 0.5f, 0.75f, 1.0f];
float[] scalar = Compute.Run(
    source,
    value => value * 2.0f + 1.0f,
    new ComputeOptions { Backend = ComputeBackendKind.Scalar });
Ensure(
    scalar.SequenceEqual([1.0f, 1.5f, 2.0f, 2.5f, 3.0f]),
    "Scalar package execution failed.");

double[] doubles = await Compute.RunAsync(
    new[] { 1.0, 2.0, 3.0 },
    value => value * 2.0 + 0.5,
    new ComputeOptions { Backend = ComputeBackendKind.Scalar });
Ensure(
    doubles.SequenceEqual([2.5, 4.5, 6.5]),
    "Double package execution failed.");

int[] integers = Compute.Zip(
    new[] { 1, 2, 3 },
    new[] { 3, 2, 1 },
    (left, right) => left * right + 1,
    new ComputeOptions { Backend = ComputeBackendKind.Simd });
Ensure(
    integers.SequenceEqual([4, 5, 4]),
    "Integer package execution failed.");

int[] histogram = Compute.Histogram(
    [-1.0f, 0.25f, 0.75f, 2.0f],
    2,
    0.0f,
    1.0f);
Ensure(
    histogram.SequenceEqual([2, 2]),
    "Histogram Clamp behavior failed.");

float[] pipelineResult = source
    .AsCompute(
        new ComputeOptions
        {
            Backend = ComputeBackendKind.Simd
        })
    .Select(value => value * 2.0f)
    .SelectInPlace(value => value + 1.0f)
    .ToArray();
Ensure(
    pipelineResult.SequenceEqual([1.0f, 1.5f, 2.0f, 2.5f, 3.0f]),
    "Lazy compute pipeline execution failed.");

Image<Rgb24> image = Image<Rgb24>.Load(
    Enumerable.Range(0, 16)
        .Select(index => new Rgb24(100, 100, 100))
        .ToArray(),
    width: 4,
    height: 4);
Image<GrayF32> gray = image.ToGrayscaleF32(
    ColorEncoding.Srgb,
    new ComputeOptions { Backend = ComputeBackendKind.Simd });
Ensure(
    gray.Pixels.Span[0].Value > 0.39f && gray.Pixels.Span[0].Value < 0.394f,
    "Image processing package conversion failed.");
Image<GrayF32> lowPass = gray.BoxBlur(
    radius: 1,
    options: new ComputeOptions { Backend = ComputeBackendKind.Simd });
Ensure(
    lowPass.Pixels.Span[0].Value > 0.39f && lowPass.Pixels.Span[0].Value < 0.394f,
    "Image processing package blur failed.");

ComputeDeviceInfo? hardwareGpu = ComputeContext.GetAccelerators()
    .FirstOrDefault(
        device => !string.Equals(
            device.AcceleratorType,
            "CPU",
            StringComparison.OrdinalIgnoreCase));

if (hardwareGpu is not null)
{
    try
    {
        ComputeDefaults.PreferredGpuAcceleratorIndex = hardwareGpu.Index;
        var gpuResult = Compute.RunWithDiagnostics(
            source,
            value => ComputeMath.Sin(value),
            new ComputeOptions
            {
                Backend = ComputeBackendKind.Gpu
            });
        Ensure(
            gpuResult.Diagnostics.Backend == ComputeBackendKind.Gpu,
            "The package did not execute on the selected GPU.");
        Ensure(
            gpuResult.Diagnostics.DeviceName == hardwareGpu.Name,
            "The package selected an unexpected default GPU.");
        Console.WriteLine(
            $"GPU package smoke test: {hardwareGpu.Name} " +
            $"({hardwareGpu.AcceleratorType}, index {hardwareGpu.Index})");
    }
    finally
    {
        ComputeDefaults.PreferredGpuAcceleratorIndex = null;
    }
}
else
{
    Console.WriteLine("GPU package smoke test skipped: no hardware GPU found.");
}

Console.WriteLine("FastCompute package smoke test passed.");

static void Ensure(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}
