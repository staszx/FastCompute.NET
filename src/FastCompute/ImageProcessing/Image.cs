namespace FastCompute.ImageProcessing;

/// <summary>
/// Represents an RGB image whose channel values are normalized to the
/// <c>0..1</c> range when loaded from bytes.
/// </summary>
public sealed class Image
{
    /// <summary>
    /// Initializes an image without allocating its pixel buffer.
    /// </summary>
    /// <param name="width">The width in pixels.</param>
    /// <param name="height">The height in pixels.</param>
    public Image(int width, int height)
    {
        ValidateDimensions(width, height);
        Width = width;
        Height = height;
    }

    /// <summary>
    /// Gets the image pixels.
    /// </summary>
    public Rgb[] Pixels { get; private set; } = Array.Empty<Rgb>();

    /// <summary>
    /// Gets the image width in pixels.
    /// </summary>
    public int Width { get; }

    /// <summary>
    /// Gets the image height in pixels.
    /// </summary>
    public int Height { get; }

    /// <summary>
    /// Gets the number of loaded pixels.
    /// </summary>
    public long Length => Pixels.LongLength;

    /// <summary>
    /// Loads tightly packed RGB24 pixel bytes in row-major order.
    /// </summary>
    /// <param name="pixelBytes">
    /// RGB bytes ordered as red, green, and blue for each pixel. This is the
    /// format produced by ImageSharp's <c>Image&lt;Rgb24&gt;.CopyPixelDataTo(byte[])</c>.
    /// </param>
    /// <param name="width">The image width in pixels.</param>
    /// <param name="height">The image height in pixels.</param>
    /// <param name="options">Optional backend-selection settings.</param>
    /// <returns>An image with channels normalized to the <c>0..1</c> range.</returns>
    /// <exception cref="ArgumentException">
    /// The byte array length does not equal <c>width * height * 3</c>.
    /// </exception>
    public static Image Load(
        byte[] pixelBytes,
        int width,
        int height,
        ComputeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(pixelBytes);
        ValidateDimensions(width, height);

        int pixelCount = checked(width * height);
        int expectedByteCount = checked(pixelCount * 3);
        if (pixelBytes.Length != expectedByteCount)
        {
            throw new ArgumentException(
                $"RGB24 data for a {width}x{height} image must contain " +
                $"{expectedByteCount} bytes, but contains {pixelBytes.Length}.",
                nameof(pixelBytes));
        }

        var pixels = GC.AllocateUninitializedArray<Rgb>(pixelCount);
        PixelConverter.Convert<Rgb24, Rgb>(
            System.Runtime.InteropServices.MemoryMarshal.Cast<byte, Rgb24>(
                pixelBytes),
            pixels,
            options: options);

        return new Image(width, height) { Pixels = pixels };
    }

    /// <summary>
    /// Converts the image to normalized grayscale luminance values.
    /// </summary>
    /// <returns>One grayscale value per source pixel in row-major order.</returns>
    /// <exception cref="InvalidOperationException">
    /// The image has no pixel buffer matching its dimensions.
    /// </exception>
    public float[] Grayscale(ComputeOptions? options = null)
    {
        int expectedPixelCount = checked(Width * Height);
        if (Pixels.Length != expectedPixelCount)
        {
            throw new InvalidOperationException(
                "The image pixel buffer has not been loaded.");
        }

        return Pixels
            .AsCompute(options ?? ComputeOptions.Default)
            .Select(Rgb.Luminance)
            .ToArray();
    }

    /// <summary>Converts the image to a compact one-byte grayscale image.</summary>
    public Image<Gray8> Grayscale8(
        ColorEncoding encoding = ColorEncoding.Srgb,
        ComputeOptions? options = null)
    {
        ValidatePixelBuffer();
        return Image<Rgb>
            .Wrap(Pixels, Width, Height)
            .ConvertTo<Rgb, Gray8>(encoding, options);
    }

    /// <summary>Returns a zero-copy typed view over the floating-point RGB pixels.</summary>
    public Image<Rgb> AsRgbImage(
        ColorEncoding encoding = ColorEncoding.Srgb)
    {
        ValidatePixelBuffer();
        return Image<Rgb>.Wrap(Pixels, Width, Height, encoding);
    }

    private void ValidatePixelBuffer()
    {
        int expectedPixelCount = checked(Width * Height);
        if (Pixels.Length != expectedPixelCount)
        {
            throw new InvalidOperationException(
                "The image pixel buffer has not been loaded.");
        }
    }

    private static void ValidateDimensions(int width, int height)
    {
        if (width <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(width),
                width,
                "Image width must be positive.");
        }

        if (height <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(height),
                height,
                "Image height must be positive.");
        }

        _ = checked(width * height);
    }
}
