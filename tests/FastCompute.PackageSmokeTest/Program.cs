using FastCompute;

float[] source = [0.0f, 0.25f, 0.5f, 0.75f, 1.0f];
float[] scalar = Compute.Run(
    source,
    value => value * 2.0f + 1.0f,
    new ComputeOptions { Backend = ComputeBackendKind.Scalar });
Ensure(
    scalar.SequenceEqual([1.0f, 1.5f, 2.0f, 2.5f, 3.0f]),
    "Scalar package execution failed.");

ComputeDeviceInfo? hardwareGpu = ComputeContext.GetAccelerators()
    .FirstOrDefault(
        device => !string.Equals(
            device.AcceleratorType,
            "CPU",
            StringComparison.OrdinalIgnoreCase));

if (hardwareGpu is not null)
{
    var gpuResult = Compute.RunWithDiagnostics(
        source,
        value => GpuMath.Sin(value),
        new ComputeOptions
        {
            Backend = ComputeBackendKind.Gpu,
            PreferredGpuAcceleratorIndex = hardwareGpu.Index
        });
    Ensure(
        gpuResult.Diagnostics.Backend == ComputeBackendKind.Gpu,
        "The package did not execute on the selected GPU.");
    Ensure(
        gpuResult.Diagnostics.DeviceName == hardwareGpu.Name,
        "The package selected an unexpected GPU.");
    Console.WriteLine(
        $"GPU package smoke test: {hardwareGpu.Name} " +
        $"({hardwareGpu.AcceleratorType}, index {hardwareGpu.Index})");
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
