namespace FastCompute.Tests;

[TestFixture]
[NonParallelizable]
public sealed class ComputeDefaultsTests
{
    [TearDown]
    public void ResetDefaults()
    {
        ComputeDefaults.PreferredGpuAcceleratorIndex = null;
    }

    [Test]
    public void PreferredGpuAcceleratorIndex_CanBeSetAndCleared()
    {
        ComputeDefaults.PreferredGpuAcceleratorIndex = 2;

        Assert.That(
            ComputeDefaults.PreferredGpuAcceleratorIndex,
            Is.EqualTo(2));

        ComputeDefaults.PreferredGpuAcceleratorIndex = null;

        Assert.That(
            ComputeDefaults.PreferredGpuAcceleratorIndex,
            Is.Null);
    }

    [Test]
    public void PreferredGpuAcceleratorIndex_RejectsNegativeValue()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => ComputeDefaults.PreferredGpuAcceleratorIndex = -1);
    }

    [Test]
    public void CpuOperation_DoesNotRequireDefaultGpuToBeAvailable()
    {
        ComputeDefaults.PreferredGpuAcceleratorIndex = int.MaxValue;

        float[] result = Compute.Run(
            [1.0f, 2.0f],
            value => value + 1.0f,
            new ComputeOptions
            {
                Backend = ComputeBackendKind.Scalar
            });

        Assert.That(result, Is.EqualTo(new[] { 2.0f, 3.0f }));
    }
}
