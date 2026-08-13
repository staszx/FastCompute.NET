namespace FastCompute.Tests;

public sealed class ConvolutionTests
{
    private static readonly ComputeBackendKind[] Backends =
    [
        ComputeBackendKind.Scalar,
        ComputeBackendKind.ParallelCpu,
        ComputeBackendKind.Simd,
        ComputeBackendKind.Gpu
    ];

    [TestCaseSource(nameof(Backends))]
    public void Convolve1D_ProducesBackendParity(ComputeBackendKind backend)
    {
        float[] source = Enumerable.Range(0, 37).Select(index => (index * 17 % 31) / 30f).ToArray();
        float[] kernel = [0.25f, 0.5f, 0.25f];
        float[] expected = Compute.Convolve1D(source, kernel, options: new ComputeOptions { Backend = ComputeBackendKind.Scalar });

        float[] actual = Compute.Convolve1D(source, kernel, options: new ComputeOptions { Backend = backend });

        Assert.That(actual, Is.EqualTo(expected).Within(2e-6f));
    }

    [TestCaseSource(nameof(Backends))]
    public void Convolve2D_ProducesBackendParity(ComputeBackendKind backend)
    {
        const int width = 11;
        const int height = 7;
        float[] source = Enumerable.Range(0, width * height).Select(index => (index * 29 % 53) / 52f).ToArray();
        float[] kernel =
        [
            0f, -1f, 0f,
            -1f, 4f, -1f,
            0f, -1f, 0f
        ];
        float[] expected = Compute.Convolve2D(source, width, height, kernel, 3, 3, options: new ComputeOptions { Backend = ComputeBackendKind.Scalar });

        float[] actual = Compute.Convolve2D(source, width, height, kernel, 3, 3, options: new ComputeOptions { Backend = backend });

        Assert.That(actual, Is.EqualTo(expected).Within(3e-6f));
    }
}
