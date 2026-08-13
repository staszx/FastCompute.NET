namespace FastCompute.ImageProcessing;

/// <summary>Provides named conversions between native image formats.</summary>
public static class ImageColorExtensions
{
    /// <summary>Converts an image to compact 8-bit grayscale.</summary>
    public static Image<Gray8> ToGrayscale8<TPixel>(
        this Image<TPixel> image,
        ColorEncoding? encoding = null,
        ComputeOptions? options = null)
        where TPixel : unmanaged =>
        image.ConvertTo<TPixel, Gray8>(encoding, options);

    /// <summary>Converts an image to floating-point grayscale.</summary>
    public static Image<GrayF32> ToGrayscaleF32<TPixel>(
        this Image<TPixel> image,
        ColorEncoding? encoding = null,
        ComputeOptions? options = null)
        where TPixel : unmanaged =>
        image.ConvertTo<TPixel, GrayF32>(encoding, options);

    /// <summary>Converts an image to tightly packed RGB24.</summary>
    public static Image<Rgb24> ToRgb24<TPixel>(
        this Image<TPixel> image,
        ColorEncoding? encoding = null,
        ComputeOptions? options = null)
        where TPixel : unmanaged =>
        image.ConvertTo<TPixel, Rgb24>(encoding, options);

    /// <summary>Converts an image to floating-point RGB.</summary>
    public static Image<Rgb> ToRgbF32<TPixel>(
        this Image<TPixel> image,
        ColorEncoding? encoding = null,
        ComputeOptions? options = null)
        where TPixel : unmanaged =>
        image.ConvertTo<TPixel, Rgb>(encoding, options);

    /// <summary>Converts channel values to linear light.</summary>
    public static Image<TPixel> ToLinear<TPixel>(this Image<TPixel> image, ComputeOptions? options = null)
        where TPixel : unmanaged =>
        image.ConvertTo<TPixel, TPixel>(ColorEncoding.Linear, options);

    /// <summary>Converts channel values to nonlinear sRGB.</summary>
    public static Image<TPixel> ToSrgb<TPixel>(this Image<TPixel> image, ComputeOptions? options = null)
        where TPixel : unmanaged =>
        image.ConvertTo<TPixel, TPixel>(ColorEncoding.Srgb, options);
}
