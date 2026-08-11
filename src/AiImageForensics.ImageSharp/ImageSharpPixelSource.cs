using System.Numerics;
using SixLabors.ImageSharp.Metadata.Profiles.Exif;
using SixLabors.ImageSharp.PixelFormats;

namespace AiImageForensics.ImageSharp;

internal sealed class ImageSharpPixelSource<TPixel> : IImagePixelSource, IImageMetadataSource
    where TPixel : unmanaged, IPixel<TPixel>
{
    private readonly SixLabors.ImageSharp.Image<TPixel> image;

    public ImageSharpPixelSource(SixLabors.ImageSharp.Image<TPixel> image) => this.image = image ?? throw new ArgumentNullException(nameof(image));

    public int Width => image.Width;
    public int Height => image.Height;

    public void CopyRow(int y, Span<RgbFloat> destination)
    {
        if ((uint)y >= (uint)Height) throw new ArgumentOutOfRangeException(nameof(y));
        if (destination.Length < Width) throw new ArgumentException("Destination is too short.", nameof(destination));

        if (image.DangerousTryGetSinglePixelMemory(out Memory<TPixel> memory))
        {
            Convert(memory.Span.Slice(y * Width, Width), destination);
            return;
        }

        for (int x = 0; x < Width; x++)
        {
            Vector4 pixel = image[x, y].ToScaledVector4();
            destination[x] = new RgbFloat(pixel.X, pixel.Y, pixel.Z);
        }
    }

    public ImageMetadataInfo GetMetadata()
    {
        ExifProfile? exif = image.Metadata.ExifProfile;
        return new ImageMetadataInfo
        {
            Software = ReadString(exif, ExifTag.Software),
            CameraMake = ReadString(exif, ExifTag.Make),
            CameraModel = ReadString(exif, ExifTag.Model)
        };
    }

    private static string? ReadString(ExifProfile? profile, ExifTag<string> tag) =>
        profile is not null && profile.TryGetValue(tag, out IExifValue<string>? value)
            ? value.Value
            : null;

    private static void Convert(ReadOnlySpan<TPixel> source, Span<RgbFloat> destination)
    {
        for (int x = 0; x < source.Length; x++)
        {
            Vector4 pixel = source[x].ToScaledVector4();
            destination[x] = new RgbFloat(pixel.X, pixel.Y, pixel.Z);
        }
    }
}
