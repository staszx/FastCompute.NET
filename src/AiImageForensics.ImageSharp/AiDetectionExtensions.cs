using SixLabors.ImageSharp.PixelFormats;

namespace AiImageForensics.ImageSharp;

/// <summary>AiImageForensics operations for ImageSharp images.</summary>
public static class AiDetectionExtensions
{
    /// <summary>Detects AI evidence without modifying or copying the source image.</summary>
    public static AiDetectionResult DetectAi<TPixel>(this SixLabors.ImageSharp.Image<TPixel> image, AiDetectionOptions? options = null, CancellationToken cancellationToken = default)
        where TPixel : unmanaged, IPixel<TPixel> =>
        AiDetector.Detect(new ImageSharpPixelSource<TPixel>(image), options, cancellationToken);

    /// <summary>Performs detailed analysis without modifying or copying the source image.</summary>
    public static AiAnalysisResult AnalyzeAi<TPixel>(this SixLabors.ImageSharp.Image<TPixel> image, AiAnalysisOptions? options = null, CancellationToken cancellationToken = default)
        where TPixel : unmanaged, IPixel<TPixel> =>
        AiAnalyzer.Analyze(new ImageSharpPixelSource<TPixel>(image), options, cancellationToken);

    /// <summary>Extracts the stable feature vector without modifying the source image.</summary>
    public static AiFeatureVector ExtractAiFeatures<TPixel>(this SixLabors.ImageSharp.Image<TPixel> image, AiDetectionOptions? options = null, CancellationToken cancellationToken = default)
        where TPixel : unmanaged, IPixel<TPixel> =>
        AiAnalyzer.ExtractFeatures(new ImageSharpPixelSource<TPixel>(image), options, cancellationToken);

    /// <summary>Measures detector stability over the package's fixed JPEG, resize, blur, and camera-simulation suite.</summary>
    public static AiRobustnessResult TestAiRobustness<TPixel>(this SixLabors.ImageSharp.Image<TPixel> image, AiDetectionOptions? options = null, CancellationToken cancellationToken = default)
        where TPixel : unmanaged, IPixel<TPixel>
    {
        ArgumentNullException.ThrowIfNull(image);
        var source = new ImageSharpPixelSource<TPixel>(image);
        return AiRobustnessTester.Test(source, new ImageSharpTransformationProvider<TPixel>(image), options, cancellationToken);
    }
}
