using AiImageForensics.ImageSharp;
using FastCompute.ImageProcessing;
using SixLabors.ImageSharp.Formats.Png;
using ImageSharpRgb24 = SixLabors.ImageSharp.PixelFormats.Rgb24;
using NativeRgb24 = FastCompute.ImageProcessing.Rgb24;

namespace FastCompute.Tests;

public sealed class AiImageLoadTests
{
    [Test]
    public void Load_StreamDecodesDirectlyIntoNativeRgb24()
    {
        using MemoryStream encoded = CreateEncodedImage();

        Image<NativeRgb24> image = AiImage.Load(encoded);

        Assert.Multiple(() =>
        {
            Assert.That(image.Width, Is.EqualTo(2));
            Assert.That(image.Height, Is.EqualTo(1));
            Assert.That(image.OwnsPixelMemory, Is.True);
            Assert.That(image.Pixels.Span[0], Is.EqualTo(new NativeRgb24(10, 20, 30)));
            Assert.That(image.Pixels.Span[1], Is.EqualTo(new NativeRgb24(200, 150, 100)));
        });
    }

    [Test]
    public void Load_GenericConvertsToRequestedLinearFormat()
    {
        using MemoryStream encoded = CreateEncodedImage();

        Image<Rgb> image = AiImage.Load<Rgb>(
            encoded,
            ColorEncoding.Linear);

        Assert.Multiple(() =>
        {
            Assert.That(image.Encoding, Is.EqualTo(ColorEncoding.Linear));
            Assert.That(
                image.Pixels.Span[0].Red,
                Is.EqualTo(PixelConverter.SrgbToLinear(10f / 255f))
                    .Within(1e-6f));
        });
    }

    [Test]
    public async Task LoadAsync_StreamSupportsCancellationAndNativeResult()
    {
        using MemoryStream encoded = CreateEncodedImage();

        Image<NativeRgb24> image = await AiImage.LoadAsync(encoded);

        Assert.That(image.Pixels.ToArray()[1].Blue, Is.EqualTo(100));
    }

    private static MemoryStream CreateEncodedImage()
    {
        var encoded = new MemoryStream();
        using (var image = new SixLabors.ImageSharp.Image<ImageSharpRgb24>(2, 1))
        {
            image[0, 0] = new ImageSharpRgb24(10, 20, 30);
            image[1, 0] = new ImageSharpRgb24(200, 150, 100);
            image.Save(encoded, new PngEncoder());
        }

        encoded.Position = 0;
        return encoded;
    }
}
