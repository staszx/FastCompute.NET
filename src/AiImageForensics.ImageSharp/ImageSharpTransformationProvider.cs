using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace AiImageForensics.ImageSharp;

/// <summary>Provides a fixed ImageSharp transformation suite for detector stability testing.</summary>
public sealed class ImageSharpTransformationProvider<TPixel> : IAiImageTransformationProvider
    where TPixel : unmanaged, IPixel<TPixel>
{
    private readonly SixLabors.ImageSharp.Image<TPixel> image;

    /// <summary>Initializes a provider for the supplied source image.</summary>
    public ImageSharpTransformationProvider(SixLabors.ImageSharp.Image<TPixel> image) => this.image = image ?? throw new ArgumentNullException(nameof(image));

    /// <inheritdoc />
    public void VisitTransformations(IImagePixelSource source, Action<string, IImagePixelSource> visitor, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(visitor);
        cancellationToken.ThrowIfCancellationRequested();

        using (var stream = new MemoryStream())
        {
            image.Save(stream, new JpegEncoder { Quality = 85 });
            stream.Position = 0;
            using SixLabors.ImageSharp.Image<TPixel> jpeg = SixLabors.ImageSharp.Image.Load<TPixel>(stream);
            visitor("jpeg-quality-85", new ImageSharpPixelSource<TPixel>(jpeg));
        }

        using (SixLabors.ImageSharp.Image<TPixel> resized = image.Clone(x =>
            x.Resize(Math.Max(1, image.Width / 2), Math.Max(1, image.Height / 2))))
            visitor("resize-50-percent", new ImageSharpPixelSource<TPixel>(resized));

        using (SixLabors.ImageSharp.Image<TPixel> blurred = image.Clone(x => x.GaussianBlur(1f)))
            visitor("gaussian-blur-1", new ImageSharpPixelSource<TPixel>(blurred));

        using (SixLabors.ImageSharp.Image<TPixel> camera = image.Clone(x => x.SimulateCamera(new CameraSimulationOptions
        {
            OpticalBlur = 0.5f, ShotNoise = 0.001f, ReadNoise = 0.0005f, Sharpening = 0.15f, RandomSeed = 1
        })))
            visitor("camera-simulation", new ImageSharpPixelSource<TPixel>(camera));
    }
}
