namespace FastCompute.ImageProcessing;

/// <summary>Provides statistics whose traversal depends on image coordinates.</summary>
public static class ImageStatistics
{
    /// <summary>Calculates correlation between an image and an offset copy.</summary>
    public static double SpatialCorrelation(
        ReadOnlySpan<float> image,
        int width,
        int height,
        int offsetX,
        int offsetY,
        ComputeOptions? options = null)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if (image.Length < checked(width * height)) throw new ArgumentException("Image buffer is shorter than its dimensions.", nameof(image));
        int startX = Math.Max(0, -offsetX);
        int endX = Math.Min(width, width - offsetX);
        int startY = Math.Max(0, -offsetY);
        int endY = Math.Min(height, height - offsetY);
        int count = checked(Math.Max(0, endX - startX) * Math.Max(0, endY - startY));
        if (count <= 1) return 0d;

        var left = GC.AllocateUninitializedArray<float>(count);
        var right = GC.AllocateUninitializedArray<float>(count);
        int destination = 0;
        for (int y = startY; y < endY; y++)
        for (int x = startX; x < endX; x++)
        {
            left[destination] = image[(y * width) + x];
            right[destination] = image[((y + offsetY) * width) + x + offsetX];
            destination++;
        }
        return Compute.Correlation(left, right, options);
    }
}
