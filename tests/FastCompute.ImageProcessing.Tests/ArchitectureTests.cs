using System.Reflection;
using FastCompute.ImageProcessing;

namespace FastCompute.ImageProcessing.Tests;

public sealed class ArchitectureTests
{
    [Test]
    public void Core_DoesNotReferenceImageProcessingOrForensics()
    {
        string[] references = typeof(Compute).Assembly
            .GetReferencedAssemblies()
            .Select(reference => reference.Name ?? string.Empty)
            .ToArray();

        Assert.Multiple(() =>
        {
            Assert.That(references, Does.Not.Contain("FastCompute.ImageProcessing"));
            Assert.That(references, Does.Not.Contain("AiImageForensics"));
        });
    }

    [Test]
    public void ImageProcessing_ReferencesCoreButNotForensics()
    {
        Assembly assembly = typeof(Image<>).Assembly;
        string[] references = assembly
            .GetReferencedAssemblies()
            .Select(reference => reference.Name ?? string.Empty)
            .ToArray();

        Assert.Multiple(() =>
        {
            Assert.That(references, Does.Contain("FastCompute"));
            Assert.That(references, Does.Not.Contain("AiImageForensics"));
        });
    }
}
