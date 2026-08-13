using FastCompute.ImageProcessing;

namespace AiImageForensics.Analysis;

internal sealed class CameraAnalyzer : IAiImageAnalyzer
{
    public AiAnalyzerResult Analyze(IImagePixelSource image, AiAnalysisContext context, CancellationToken cancellationToken)
    {
        int width = image.Width, height = image.Height;
        Image<Rgb> native = context.GetRgbImage(cancellationToken);
        RgbCorrelationMeasurements correlations = RgbImageMeasurements.CalculateCorrelations(native);
        float[] channels = correlations.ChannelCorrelations;
        float[] neighbors = correlations.NeighbourCorrelations;
        bool detailed = context.Options.Mode != DetectionMode.Fast;
        CfaAnalysisResult cfa = detailed
            ? AnalyzeCfa(RgbImageMeasurements.CalculateCfaParityResiduals(native).MeanResiduals)
            : new CfaAnalysisResult();
        float sameMean = MeanAbsolute(channels), neighborMean = MeanAbsolute(neighbors);
        float demosaicing = detailed ? Math.Clamp((neighborMean - sameMean + 0.25f) / 0.5f, 0, 1) : 0;
        float cameraSupport = detailed ? (0.65f * cfa.Score) + (0.35f * demosaicing) : Math.Clamp((sameMean - 0.8f) / 0.2f, 0, 1);
        float score = Math.Clamp(0.5f - (0.45f * cameraSupport), 0, 1);
        float confidence = detailed
            ? Math.Clamp((cfa.Confidence + MathF.Min(1, width * height / 65536f)) * 0.5f, 0.2f, 0.9f)
            : Math.Clamp(width * height / 65536f, 0.2f, 0.55f);
        var details = new CameraAnalysisResult
        {
            EstimatedPattern = cfa.EstimatedPattern, CfaScore = cfa.Score, CfaConfidence = cfa.Confidence,
            ChannelCorrelations = channels, NeighborCorrelations = neighbors, DemosaicingScore = demosaicing
        };
        var evidence = new List<AiEvidence>
        {
            new AiEvidence { Type = AiEvidenceType.CameraSensor, Score = score, Confidence = confidence, Message = detailed ? "Weak statistical consistency with a traditional camera pipeline was evaluated without claiming a camera model." : "Fast mode evaluated RGB channel correlations as weak camera-pipeline evidence." }
        };
        if (detailed)
        {
            evidence.Add(new AiEvidence { Type = AiEvidenceType.Cfa, Score = 1 - cfa.Score, Confidence = cfa.Confidence, Message = "Four parity classes were compared for CFA-like interpolation structure." });
            evidence.Add(new AiEvidence { Type = AiEvidenceType.Demosaicing, Score = 1 - demosaicing, Confidence = confidence, Message = "Cross-channel neighbor correlations were compared for demosaicing-like structure." });
        }
        return new AiAnalyzerResult { Score = score, Confidence = confidence, Details = details, Evidence = evidence };
    }

    private static CfaAnalysisResult AnalyzeCfa(double[,] means)
    {
        BayerPattern[] patterns = Enum.GetValues<BayerPattern>();
        float best = 0, second = 0; BayerPattern? bestPattern = null;
        foreach (BayerPattern pattern in patterns)
        {
            int[] expected = GetPatternChannels(pattern);
            double matched = 0, alternatives = 0;
            for (int p = 0; p < 4; p++)
            {
                matched += means[p, expected[p]];
                for (int c = 0; c < 3; c++) if (c != expected[p]) alternatives += means[p, c] / 2;
            }
            float score = (float)Math.Clamp((matched - alternatives) / Math.Max(matched + alternatives, 1e-12) * 2 + 0.5, 0, 1);
            if (score > best) { second = best; best = score; bestPattern = pattern; } else if (score > second) second = score;
        }
        float confidence = Math.Clamp((best - second) * 4f, 0, 1);
        return new CfaAnalysisResult { EstimatedPattern = confidence >= 0.08f ? bestPattern : null, Score = best, Confidence = confidence };
    }

    private static int[] GetPatternChannels(BayerPattern pattern) => pattern switch
    {
        BayerPattern.Rggb => [0, 1, 1, 2], BayerPattern.Bggr => [2, 1, 1, 0],
        BayerPattern.Grbg => [1, 0, 2, 1], BayerPattern.Gbrg => [1, 2, 0, 1], _ => throw new ArgumentOutOfRangeException(nameof(pattern))
    };

    private static float MeanAbsolute(float[] values)
    {
        float sum = 0; for (int i = 0; i < values.Length; i++) sum += MathF.Abs(values[i]); return sum / values.Length;
    }

}
