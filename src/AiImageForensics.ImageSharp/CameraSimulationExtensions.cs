using System.Numerics;
using FastCompute.ImageProcessing;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Processing.Processors;

namespace AiImageForensics.ImageSharp;

/// <summary>Camera-simulation operations for ImageSharp processing pipelines.</summary>
public static class CameraSimulationExtensions
{
    /// <summary>Applies deterministic camera-pipeline simulation.</summary>
    public static IImageProcessingContext SimulateCamera(this IImageProcessingContext context, CameraSimulationOptions options)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(options);
        return context.ApplyProcessor(new CameraSimulationProcessor(options));
    }
}

internal sealed class CameraSimulationProcessor(CameraSimulationOptions options) : IImageProcessor
{
    public IImageProcessor<TPixel> CreatePixelSpecificProcessor<TPixel>(Configuration configuration, SixLabors.ImageSharp.Image<TPixel> source, Rectangle sourceRectangle)
        where TPixel : unmanaged, IPixel<TPixel> =>
        new CameraSimulationProcessor<TPixel>(configuration, source, sourceRectangle, options);
}

internal sealed class CameraSimulationProcessor<TPixel>(
    Configuration configuration,
    SixLabors.ImageSharp.Image<TPixel> source,
    Rectangle sourceRectangle,
    CameraSimulationOptions options)
    : ImageProcessor<TPixel>(configuration, source, sourceRectangle)
    where TPixel : unmanaged, IPixel<TPixel>
{
    protected override void OnFrameApply(SixLabors.ImageSharp.ImageFrame<TPixel> source)
    {
        int width = source.Width, height = source.Height;
        var pixels = new Rgb[checked(width * height)];
        if (source.DangerousTryGetSinglePixelMemory(out Memory<TPixel> memory))
        {
            ReadOnlySpan<TPixel> sourcePixels = memory.Span;
            for (int i = 0; i < pixels.Length; i++) pixels[i] = ToRgb(sourcePixels[i]);
        }
        else
        {
            for (int y = 0; y < height; y++) for (int x = 0; x < width; x++) pixels[(y * width) + x] = ToRgb(source[x, y]);
        }

        FastCompute.ImageProcessing.Image<Rgb> native = FastCompute.ImageProcessing.Image<Rgb>.Load(pixels, width, height, ColorEncoding.Srgb);
        FastCompute.ImageProcessing.Image<Rgb> simulated = CameraSimulator.SimulateCamera(native, options);
        ReadOnlySpan<Rgb> result = simulated.Pixels.Span;

        if (source.DangerousTryGetSinglePixelMemory(out memory))
        {
            Span<TPixel> destination = memory.Span;
            for (int i = 0; i < result.Length; i++) destination[i] = FromRgb(result[i]);
        }
        else
        {
            for (int y = 0; y < height; y++) for (int x = 0; x < width; x++) source[x, y] = FromRgb(result[(y * width) + x]);
        }
    }

    private static Rgb ToRgb(TPixel pixel)
    {
        Vector4 vector = pixel.ToScaledVector4();
        return new Rgb(vector.X, vector.Y, vector.Z);
    }

    private static TPixel FromRgb(Rgb pixel)
    {
        TPixel result = default;
        result.FromScaledVector4(new Vector4(pixel.Red, pixel.Green, pixel.Blue, 1));
        return result;
    }
}
