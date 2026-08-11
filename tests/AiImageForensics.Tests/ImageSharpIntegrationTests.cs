using AiImageForensics.ImageSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace AiImageForensics.Tests;

public sealed class ImageSharpIntegrationTests
{
    [TestCase(typeof(Rgb24))]
    [TestCase(typeof(Rgba32))]
    [TestCase(typeof(Bgr24))]
    [TestCase(typeof(L8))]
    public void DetectAi_SupportsPixelFormatsWithoutModifyingSource(Type pixelType)
    {
        if (pixelType == typeof(Rgb24)) RunReadOnlyCheck(new Image<Rgb24>(32, 32, new Rgb24(50, 100, 150)));
        else if (pixelType == typeof(Rgba32)) RunReadOnlyCheck(new Image<Rgba32>(32, 32, new Rgba32(50, 100, 150, 200)));
        else if (pixelType == typeof(Bgr24)) RunReadOnlyCheck(new Image<Bgr24>(32, 32, new Bgr24(50, 100, 150)));
        else RunReadOnlyCheck(new Image<L8>(32, 32, new L8(90)));
    }

    [Test]
    public void SimulateCamera_ModifiesImageDeterministically()
    {
        using var image = new Image<Rgb24>(16, 16);
        image.ProcessPixelRows(a => { for (int y = 0; y < a.Height; y++) { Span<Rgb24> row = a.GetRowSpan(y); for (int x = 0; x < row.Length; x++) row[x] = new Rgb24((byte)(x * 15), (byte)(y * 15), (byte)((x + y) * 7)); } });
        Rgb24 before = image[8, 8];
        image.Mutate(x => x.SimulateCamera(new CameraSimulationOptions { ShotNoise = 0.002f, ReadNoise = 0.001f, OpticalBlur = 1, Sharpening = 0.2f, RandomSeed = 42 }));
        Assert.That(image[8, 8], Is.Not.EqualTo(before));
    }

    [Test]
    public void RobustnessSuite_ReportsFixedTransformations()
    {
        using var image = new Image<Rgb24>(16, 16, new Rgb24(60, 100, 140));
        AiRobustnessResult result = image.TestAiRobustness(new AiDetectionOptions { Mode = DetectionMode.Fast, AnalyzeMetadata = false });
        Assert.Multiple(() =>
        {
            Assert.That(result.Cases.Select(c => c.Name), Is.EqualTo(new[] { "jpeg-quality-85", "resize-50-percent", "gaussian-blur-1", "camera-simulation" }));
            Assert.That(result.MinimumScore, Is.InRange(0, 1));
            Assert.That(result.MaximumScore, Is.InRange(0, 1));
            Assert.That(result.Stability, Is.InRange(0, 1));
        });
    }

    private static void RunReadOnlyCheck<TPixel>(Image<TPixel> image) where TPixel : unmanaged, IPixel<TPixel>
    {
        using (image)
        using (Image<TPixel> clone = image.Clone())
        {
            AiDetectionResult result = image.DetectAi(new AiDetectionOptions { Mode = DetectionMode.Fast });
            Assert.That(result.Score, Is.InRange(0, 1));
            Assert.That(image[0, 0], Is.EqualTo(clone[0, 0]));
        }
    }
}
