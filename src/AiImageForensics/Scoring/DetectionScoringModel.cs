using AiImageForensics.Analysis;

namespace AiImageForensics.Scoring;

internal static class DefaultDetectionWeights
{
    public const float Frequency = 0.25f;
    public const float Noise = 0.25f;
    public const float Camera = 0.20f;
    public const float Spatial = 0.20f;
    public const float Metadata = 0.10f;
}

internal sealed class DetectionScoringModel
{
    public (float Score, float Confidence) Combine(IReadOnlyList<WeightedAnalyzerResult> results)
    {
        double contribution = 0, effectiveWeight = 0, confidenceWeight = 0;
        for (int i = 0; i < results.Count; i++)
        {
            WeightedAnalyzerResult item = results[i];
            if (item.Result.Confidence <= 0 || item.Weight <= 0) continue;
            double weight = item.Weight * item.Result.Confidence;
            contribution += item.Result.Score * weight;
            effectiveWeight += weight;
            confidenceWeight += item.Weight;
        }
        if (effectiveWeight <= 0 || confidenceWeight <= 0) return (0, 0);
        float score = (float)Math.Clamp(contribution / effectiveWeight, 0, 1);
        float confidence = (float)Math.Clamp(effectiveWeight / confidenceWeight, 0, 1);
        return (score, confidence);
    }
}

internal readonly record struct WeightedAnalyzerResult(AiAnalyzerResult Result, float Weight);
