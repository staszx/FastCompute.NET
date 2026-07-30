using System.Runtime.Intrinsics.X86;

namespace FastCompute.Tests;

using ILGPU.Algorithms.MatrixOperations;

[TestFixture]
public sealed class ReductionTests
{
    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    [TestCase(ComputeBackendKind.Simd)]
    public void Reductions_ReturnExpectedValues(ComputeBackendKind backend)
    {
        if (backend == ComputeBackendKind.Simd && !Avx.IsSupported)
        {
            Assert.Ignore("SIMD reduction requires AVX.");
        }

        float[] source = [-5f, 2f, 7f, -1f, 4f, 3f, 8f, -2f, 6f, 1f, 9f];
        var options = new ComputeOptions
        {
            Backend = backend,
            MaxDegreeOfParallelism = 2
        };

        Assert.Multiple(() =>
        {
            Assert.That(Compute.Sum(source, options), Is.EqualTo(32f));
            Assert.That(Compute.Min(source, options), Is.EqualTo(-5f));
            Assert.That(Compute.Max(source, options), Is.EqualTo(9f));
            Assert.That(
                Compute.Average(source, options),
                Is.EqualTo(32f / source.Length).Within(1e-6f));
        });
    }

    [Test]
    public void EmptyReduction_HasDefinedBehavior()
    {
        Assert.Multiple(() =>
        {
            Assert.That(Compute.Sum([]), Is.EqualTo(0f));
            Assert.That(() => Compute.Min([]), Throws.TypeOf<InvalidOperationException>());
            Assert.That(() => Compute.Max([]), Throws.TypeOf<InvalidOperationException>());
            Assert.That(() => Compute.Average([]), Throws.TypeOf<InvalidOperationException>());
        });
    }

    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    [TestCase(ComputeBackendKind.Simd)]
    public void MinMax_PropagateNaN(ComputeBackendKind backend)
    {
        if (backend == ComputeBackendKind.Simd && !Avx.IsSupported)
        {
            Assert.Ignore("SIMD reduction requires AVX.");
        }

        var options = new ComputeOptions { Backend = backend };
        float[] source = [1f, 2f, float.NaN, 3f, 4f, 5f, 6f, 7f, 8f];

        Assert.Multiple(() =>
        {
            Assert.That(float.IsNaN(Compute.Min(source, options)), Is.True);
            Assert.That(float.IsNaN(Compute.Max(source, options)), Is.True);
        });
    }


    private static float mmm(float x)
    {
        return x + 10;
    }

    [Test]
    [Explicit("Manual 4 GiB allocation experiment.")]
    public void Test()
    {
        var source = new float[1024 * 1024 * 1024];

        var s =  Compute.Run(source, w => w+2, new ComputeOptions() { Backend = ComputeBackendKind.ParallelCpu });
       //var diagnostics = s.Diagnostics;
       //Console.WriteLine($"Backend: {diagnostics.Backend}");
       //Console.WriteLine($"Device: {diagnostics.DeviceName}");
       //Console.WriteLine($"Planning: {diagnostics.PlanningTime}");
       //Console.WriteLine($"Compilation: {diagnostics.CompilationTime}");
       //Console.WriteLine($"Upload: {diagnostics.UploadTime}");
       //Console.WriteLine($"Execution: {diagnostics.ExecutionTime}");
       //Console.WriteLine($"Download: {diagnostics.DownloadTime}");
       //Console.WriteLine($"Cache hit: {diagnostics.KernelCacheHit}");
    }
}
