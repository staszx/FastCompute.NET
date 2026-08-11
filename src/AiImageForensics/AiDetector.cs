namespace AiImageForensics;

/// <summary>Provides normalized heuristic AI-image detection.</summary>
public static class AiDetector
{
    /// <summary>Analyzes an image and evaluates the configured detection threshold.</summary>
    public static AiDetectionResult Detect(IImagePixelSource image, AiDetectionOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        options ??= new AiDetectionOptions();
        AiAnalysisResult analysis = AiAnalyzer.Run(image, options, cancellationToken).Analysis;
        return new AiDetectionResult
        {
            Score = analysis.AiScore,
            Confidence = analysis.Confidence,
            IsLikelyAi = analysis.AiScore >= options.DetectionThreshold,
            Evidence = analysis.Evidence
        };
    }
}
