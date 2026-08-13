using FastCompute.ImageProcessing;

namespace AiImageForensics.Analysis;

internal interface IAiImageAnalyzer
{
    AiAnalyzerResult Analyze(IImagePixelSource image, AiAnalysisContext context, CancellationToken cancellationToken);
}

internal sealed class AiAnalyzerResult
{
    public float Score { get; init; }
    public float Confidence { get; init; }
    public IReadOnlyList<AiEvidence> Evidence { get; init; } = Array.Empty<AiEvidence>();
    public object? Details { get; init; }
}

internal sealed class CfaAnalysisResult
{
    public BayerPattern? EstimatedPattern { get; init; }
    public float Score { get; init; }
    public float Confidence { get; init; }
}

internal interface ILinearLuminanceSource
{
    bool TryCreateLinearLuminance(CancellationToken cancellationToken, out float[]? luminance);
}
