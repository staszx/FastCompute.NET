namespace AiImageForensics.Analysis;

internal sealed class MetadataAnalyzer : IAiImageAnalyzer
{
    private static readonly string[] AiMarkers = ["stable diffusion", "midjourney", "dall-e", "dalle", "comfyui", "automatic1111", "invokeai", "firefly", "generative fill"];

    public AiAnalyzerResult Analyze(IImagePixelSource image, AiAnalysisContext context, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (image is not IImageMetadataSource metadataSource)
            return new AiAnalyzerResult { Details = new MetadataAnalysisResult { IsAvailable = false } };

        ImageMetadataInfo metadata = metadataSource.GetMetadata();
        string combined = string.Join(' ', metadata.Software, metadata.CreatorTool, metadata.Generator).ToLowerInvariant();
        bool explicitAi = AiMarkers.Any(combined.Contains);
        var details = new MetadataAnalysisResult { IsAvailable = true, HasExplicitAiProvenance = explicitAi, Software = metadata.Software };
        if (!explicitAi) return new AiAnalyzerResult { Score = 0, Confidence = 0.25f, Details = details, Evidence = [new AiEvidence { Type = AiEvidenceType.Metadata, Score = 0, Confidence = 0.25f, Message = "Metadata was available but contained no explicit recognized AI provenance." }] };
        return new AiAnalyzerResult { Score = 1, Confidence = 0.98f, Details = details, Evidence = [new AiEvidence { Type = AiEvidenceType.Metadata, Score = 1, Confidence = 0.98f, Message = "Metadata contains an explicit recognized generative-software identifier." }] };
    }
}
