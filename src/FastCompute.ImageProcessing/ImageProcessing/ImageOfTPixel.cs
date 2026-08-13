namespace FastCompute.ImageProcessing;

/// <summary>
/// Represents an image backed by native pixels of a specific physical format.
/// </summary>
/// <typeparam name="TPixel">The unmanaged pixel format.</typeparam>
public sealed class Image<TPixel>
    where TPixel : unmanaged
{
    private Image(
        int width,
        int height,
        Memory<TPixel> pixels,
        ColorEncoding encoding,
        bool ownsPixelMemory)
    {
        Validate(width, height, pixels.Length);
        Width = width;
        Height = height;
        Pixels = pixels;
        Encoding = encoding;
        OwnsPixelMemory = ownsPixelMemory;
    }

    /// <summary>Gets the image width.</summary>
    public int Width { get; }

    /// <summary>Gets the image height.</summary>
    public int Height { get; }

    /// <summary>Gets the contiguous pixel memory.</summary>
    public Memory<TPixel> Pixels { get; }

    /// <summary>Gets the channel transfer function.</summary>
    public ColorEncoding Encoding { get; }

    /// <summary>
    /// Gets whether this image owns the array supplied to <see cref="Load"/>.
    /// </summary>
    public bool OwnsPixelMemory { get; }

    /// <summary>Gets the total number of pixels.</summary>
    public int Length => Pixels.Length;

    /// <summary>Gets a mutable span for one row without allocation.</summary>
    public Span<TPixel> GetRowSpan(int y)
    {
        ValidateRow(y);
        return Pixels.Span.Slice(checked(y * Width), Width);
    }

    /// <summary>Gets a read-only span for one row without allocation.</summary>
    public ReadOnlySpan<TPixel> GetReadOnlyRowSpan(int y)
    {
        ValidateRow(y);
        return Pixels.Span.Slice(checked(y * Width), Width);
    }

    /// <summary>Copies one row into caller-provided memory.</summary>
    public void CopyRow(int y, Span<TPixel> destination)
    {
        if (destination.Length < Width)
        {
            throw new ArgumentException(
                $"The destination must contain at least {Width} pixels.",
                nameof(destination));
        }

        GetReadOnlyRowSpan(y).CopyTo(destination);
    }

    /// <summary>Creates an independent image with copied pixels.</summary>
    public Image<TPixel> Clone() => Load(
        Pixels.ToArray(),
        Width,
        Height,
        Encoding);

    /// <summary>Copies a rectangular region into a new image.</summary>
    public Image<TPixel> Crop(int x, int y, int width, int height)
    {
        if (x < 0 || y < 0 || width <= 0 || height <= 0 ||
            x > Width - width || y > Height - height)
        {
            throw new ArgumentOutOfRangeException(
                nameof(x),
                "The crop rectangle must be positive and contained in the image.");
        }

        var result = GC.AllocateUninitializedArray<TPixel>(
            checked(width * height));
        for (int row = 0; row < height; row++)
        {
            GetReadOnlyRowSpan(y + row)
                .Slice(x, width)
                .CopyTo(result.AsSpan(row * width, width));
        }

        return Load(result, width, height, Encoding);
    }

    /// <summary>Creates an image that takes ownership of an existing array.</summary>
    public static Image<TPixel> Load(
        TPixel[] pixels,
        int width,
        int height,
        ColorEncoding encoding = ColorEncoding.Srgb)
    {
        ArgumentNullException.ThrowIfNull(pixels);
        return new Image<TPixel>(
            width,
            height,
            pixels,
            encoding,
            ownsPixelMemory: true);
    }

    /// <summary>
    /// Creates a zero-copy view over caller-owned contiguous pixel memory.
    /// The caller must keep that memory alive while the image is used.
    /// </summary>
    public static Image<TPixel> Wrap(
        Memory<TPixel> pixels,
        int width,
        int height,
        ColorEncoding encoding = ColorEncoding.Srgb) =>
        new(
            width,
            height,
            pixels,
            encoding,
            ownsPixelMemory: false);

    private static void Validate(int width, int height, int pixelCount)
    {
        if (width <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width));
        }

        if (height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(height));
        }

        int expected = checked(width * height);
        if (pixelCount != expected)
        {
            throw new ArgumentException(
                $"A {width}x{height} image requires {expected} pixels, " +
                $"but the memory contains {pixelCount}.",
                nameof(pixelCount));
        }
    }

    private void ValidateRow(int y)
    {
        if ((uint)y >= (uint)Height)
        {
            throw new ArgumentOutOfRangeException(nameof(y));
        }
    }
}
