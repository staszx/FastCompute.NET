using System.Runtime.InteropServices;

namespace FastCompute.ImageProcessing;

/// <summary>Provides reusable filters for contiguous floating-point images.</summary>
public static class ImageFilters
{
    private static readonly float[] SobelXKernel =
    [
        -0.125f, 0f, 0.125f,
        -0.25f, 0f, 0.25f,
        -0.125f, 0f, 0.125f
    ];

    private static readonly float[] SobelYKernel =
    [
        -0.125f, -0.25f, -0.125f,
        0f, 0f, 0f,
        0.125f, 0.25f, 0.125f
    ];

    private static readonly float[] LaplacianKernel =
    [
        0f, 1f, 0f,
        1f, -4f, 1f,
        0f, 1f, 0f
    ];

    /// <summary>Applies a Gaussian blur generated from the requested radius and sigma.</summary>
    public static Image<GrayF32> GaussianBlur(this Image<GrayF32> source, int radius = 1, float sigma = 0f, ComputeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        float[] result = GaussianBlur(AsFloats(source), source.Width, source.Height, radius, sigma, options);
        return FromFloats(result, source);
    }

    /// <summary>Applies a reusable low-pass box filter.</summary>
    public static Image<GrayF32> LowPassFilter(this Image<GrayF32> source, int radius = 1, ComputeOptions? options = null) =>
        source.BoxBlur(radius, options: options);

    /// <summary>Returns the high-pass residual of a low-pass box filter.</summary>
    public static Image<GrayF32> HighPassFilter(this Image<GrayF32> source, int radius = 1, ComputeOptions? options = null) =>
        source.ExtractResidual(radius, options);

    /// <summary>Applies a Gaussian blur to a row-major numeric buffer.</summary>
    public static float[] GaussianBlur(ReadOnlySpan<float> source, int width, int height, int radius = 1, float sigma = 0f, ComputeOptions? options = null)
    {
        if (radius < 0) throw new ArgumentOutOfRangeException(nameof(radius));
        if (radius == 0) return source.ToArray();
        if (sigma == 0f) sigma = MathF.Max(0.5f, radius / 2f);
        if (!float.IsFinite(sigma) || sigma <= 0f) throw new ArgumentOutOfRangeException(nameof(sigma));
        int size = checked((radius * 2) + 1);
        var kernel = new float[checked(size * size)];
        double sum = 0d;
        double denominator = 2d * sigma * sigma;
        for (int y = -radius; y <= radius; y++)
        for (int x = -radius; x <= radius; x++)
        {
            float weight = (float)Math.Exp(-((x * x) + (y * y)) / denominator);
            kernel[((y + radius) * size) + x + radius] = weight;
            sum += weight;
        }
        float normalization = (float)(1d / sum);
        Compute.RunInPlace(kernel, value => value * normalization, options);
        return Compute.Convolve2D(source, width, height, kernel, size, size, options: options);
    }

    /// <summary>Returns source minus a low-pass box-filtered version.</summary>
    public static Image<GrayF32> ExtractResidual(this Image<GrayF32> source, int radius = 1, ComputeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        float[] result = ExtractResidual(AsFloats(source), source.Width, source.Height, radius, options);
        return FromFloats(result, source);
    }

    /// <summary>Returns source minus a low-pass box-filtered version.</summary>
    public static float[] ExtractResidual(ReadOnlySpan<float> source, int width, int height, int radius = 1, ComputeOptions? options = null)
    {
        float[] input = source.ToArray();
        var lowPass = new float[input.Length];
        GrayImageOperations.BoxBlur(input, lowPass, width, height, radius, options: options);
        return Compute.Zip(input, lowPass, (value, smooth) => value - smooth, options);
    }

    /// <summary>Calculates the horizontal Sobel derivative.</summary>
    public static float[] GradientX(ReadOnlySpan<float> source, int width, int height, ComputeOptions? options = null) =>
        Compute.Convolve2D(source, width, height, SobelXKernel, 3, 3, options: options);

    /// <summary>Calculates the vertical Sobel derivative.</summary>
    public static float[] GradientY(ReadOnlySpan<float> source, int width, int height, ComputeOptions? options = null) =>
        Compute.Convolve2D(source, width, height, SobelYKernel, 3, 3, options: options);

    /// <summary>Calculates the Sobel gradient magnitude.</summary>
    public static float[] GradientMagnitude(ReadOnlySpan<float> source, int width, int height, ComputeOptions? options = null)
    {
        float[] x = GradientX(source, width, height, options);
        float[] y = GradientY(source, width, height, options);
        return Compute.Zip(x, y, (horizontal, vertical) => ComputeMath.Sqrt((horizontal * horizontal) + (vertical * vertical)), options);
    }

    /// <summary>Calculates the Sobel gradient magnitude image.</summary>
    public static Image<GrayF32> Sobel(this Image<GrayF32> source, ComputeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        return FromFloats(GradientMagnitude(AsFloats(source), source.Width, source.Height, options), source);
    }

    /// <summary>Applies a four-neighbour discrete Laplacian.</summary>
    public static float[] Laplacian(ReadOnlySpan<float> source, int width, int height, ComputeOptions? options = null) =>
        Compute.Convolve2D(source, width, height, LaplacianKernel, 3, 3, options: options);

    /// <summary>Applies a four-neighbour discrete Laplacian.</summary>
    public static Image<GrayF32> Laplacian(this Image<GrayF32> source, ComputeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        return FromFloats(Laplacian(AsFloats(source), source.Width, source.Height, options), source);
    }

    /// <summary>Creates a binary edge map from Sobel magnitude.</summary>
    public static float[] EdgeMap(ReadOnlySpan<float> source, int width, int height, float threshold, ComputeOptions? options = null)
    {
        if (!float.IsFinite(threshold) || threshold < 0f) throw new ArgumentOutOfRangeException(nameof(threshold));
        float[] magnitude = GradientMagnitude(source, width, height, options);
        return Compute.Threshold(magnitude, threshold, options);
    }

    /// <summary>Sharpens a floating-point image using its Laplacian response.</summary>
    public static float[] Sharpen(ReadOnlySpan<float> source, int width, int height, float amount = 1f, ComputeOptions? options = null)
    {
        if (!float.IsFinite(amount) || amount < 0f) throw new ArgumentOutOfRangeException(nameof(amount));
        float[] input = source.ToArray();
        float[] laplacian = Laplacian(source, width, height, options);
        return Compute.Zip(input, laplacian, (value, detail) => value - (detail * amount), options);
    }

    private static ReadOnlySpan<float> AsFloats(Image<GrayF32> source) =>
        MemoryMarshal.Cast<GrayF32, float>(source.Pixels.Span);

    private static Image<GrayF32> FromFloats(float[] values, Image<GrayF32> source) =>
        Image<GrayF32>.Load(MemoryMarshal.Cast<float, GrayF32>(values).ToArray(), source.Width, source.Height, source.Encoding);
}
