namespace FastCompute.ImageProcessing;

/// <summary>Provides reusable row-major image-region operations.</summary>
public static class ImageRegionOperations
{
    /// <summary>Copies a rectangular region into a compact row-major buffer.</summary>
    public static float[] Crop(ReadOnlySpan<float> source, int sourceWidth, int sourceHeight, int x, int y, int width, int height)
    {
        if (sourceWidth <= 0) throw new ArgumentOutOfRangeException(nameof(sourceWidth));
        if (sourceHeight <= 0) throw new ArgumentOutOfRangeException(nameof(sourceHeight));
        if (source.Length < checked(sourceWidth * sourceHeight)) throw new ArgumentException("Source is shorter than its dimensions.", nameof(source));
        if (x < 0 || width <= 0 || x > sourceWidth - width) throw new ArgumentOutOfRangeException(nameof(x));
        if (y < 0 || height <= 0 || y > sourceHeight - height) throw new ArgumentOutOfRangeException(nameof(y));
        var result = GC.AllocateUninitializedArray<float>(checked(width * height));
        for (int row = 0; row < height; row++)
            source.Slice(((y + row) * sourceWidth) + x, width).CopyTo(result.AsSpan(row * width, width));
        return result;
    }
}
