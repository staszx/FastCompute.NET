using System.Runtime.InteropServices;
using FastCompute.ImageProcessing;
using ImageSharpRgb24 = SixLabors.ImageSharp.PixelFormats.Rgb24;
using NativeRgb24 = FastCompute.ImageProcessing.Rgb24;

namespace AiImageForensics.ImageSharp;

/// <summary>
/// Decodes encoded raster images into native FastCompute image formats.
/// </summary>
public static class AiImage
{
    /// <summary>Decodes a file into a tightly packed native RGB24 image.</summary>
    public static Image<NativeRgb24> Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using SixLabors.ImageSharp.Image<ImageSharpRgb24> decoded =
            SixLabors.ImageSharp.Image.Load<ImageSharpRgb24>(path);
        return CopyToNative(decoded);
    }

    /// <summary>Decodes a stream into a tightly packed native RGB24 image.</summary>
    public static Image<NativeRgb24> Load(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        using SixLabors.ImageSharp.Image<ImageSharpRgb24> decoded =
            SixLabors.ImageSharp.Image.Load<ImageSharpRgb24>(stream);
        return CopyToNative(decoded);
    }

    /// <summary>Decodes encoded bytes into a tightly packed native RGB24 image.</summary>
    public static Image<NativeRgb24> Load(byte[] encodedData)
    {
        ArgumentNullException.ThrowIfNull(encodedData);
        using SixLabors.ImageSharp.Image<ImageSharpRgb24> decoded =
            SixLabors.ImageSharp.Image.Load<ImageSharpRgb24>(encodedData);
        return CopyToNative(decoded);
    }

    /// <summary>Decodes a file and converts it to a requested native format.</summary>
    public static Image<TPixel> Load<TPixel>(
        string path,
        ColorEncoding encoding = ColorEncoding.Srgb)
        where TPixel : unmanaged =>
        Convert<TPixel>(Load(path), encoding);

    /// <summary>Decodes a stream and converts it to a requested native format.</summary>
    public static Image<TPixel> Load<TPixel>(
        Stream stream,
        ColorEncoding encoding = ColorEncoding.Srgb)
        where TPixel : unmanaged =>
        Convert<TPixel>(Load(stream), encoding);

    /// <summary>Decodes bytes and converts them to a requested native format.</summary>
    public static Image<TPixel> Load<TPixel>(
        byte[] encodedData,
        ColorEncoding encoding = ColorEncoding.Srgb)
        where TPixel : unmanaged =>
        Convert<TPixel>(Load(encodedData), encoding);

    /// <summary>Asynchronously decodes a file into native RGB24.</summary>
    public static async Task<Image<NativeRgb24>> LoadAsync(
        string path,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        await using FileStream stream = File.OpenRead(path);
        return await LoadAsync(stream, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Asynchronously decodes a stream into native RGB24.</summary>
    public static async Task<Image<NativeRgb24>> LoadAsync(
        Stream stream,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(stream);
        using SixLabors.ImageSharp.Image<ImageSharpRgb24> decoded =
            await SixLabors.ImageSharp.Image
                .LoadAsync<ImageSharpRgb24>(stream, cancellationToken)
                .ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();
        return CopyToNative(decoded, cancellationToken);
    }

    /// <summary>Asynchronously decodes and converts a file.</summary>
    public static async Task<Image<TPixel>> LoadAsync<TPixel>(
        string path,
        ColorEncoding encoding = ColorEncoding.Srgb,
        CancellationToken cancellationToken = default)
        where TPixel : unmanaged
    {
        Image<NativeRgb24> decoded = await LoadAsync(path, cancellationToken)
            .ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();
        return Convert<TPixel>(decoded, encoding);
    }

    /// <summary>Asynchronously decodes and converts a stream.</summary>
    public static async Task<Image<TPixel>> LoadAsync<TPixel>(
        Stream stream,
        ColorEncoding encoding = ColorEncoding.Srgb,
        CancellationToken cancellationToken = default)
        where TPixel : unmanaged
    {
        Image<NativeRgb24> decoded = await LoadAsync(stream, cancellationToken)
            .ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();
        return Convert<TPixel>(decoded, encoding);
    }

    private static Image<TPixel> Convert<TPixel>(
        Image<NativeRgb24> decoded,
        ColorEncoding encoding)
        where TPixel : unmanaged
    {
        if (typeof(TPixel) == typeof(NativeRgb24) &&
            encoding == ColorEncoding.Srgb)
        {
            return (Image<TPixel>)(object)decoded;
        }

        return decoded.ConvertTo<NativeRgb24, TPixel>(encoding);
    }

    private static Image<NativeRgb24> CopyToNative(
        SixLabors.ImageSharp.Image<ImageSharpRgb24> decoded,
        CancellationToken cancellationToken = default)
    {
        var pixels = GC.AllocateUninitializedArray<NativeRgb24>(
            checked(decoded.Width * decoded.Height));
        decoded.ProcessPixelRows(
            accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    ReadOnlySpan<NativeRgb24> source =
                        MemoryMarshal.Cast<ImageSharpRgb24, NativeRgb24>(
                            accessor.GetRowSpan(y));
                    source.CopyTo(pixels.AsSpan(y * decoded.Width, decoded.Width));
                }
            });
        return Image<NativeRgb24>.Load(
            pixels,
            decoded.Width,
            decoded.Height,
            ColorEncoding.Srgb);
    }
}
