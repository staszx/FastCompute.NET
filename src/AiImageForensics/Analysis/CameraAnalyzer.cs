using System.Buffers;

namespace AiImageForensics.Analysis;

internal sealed class CameraAnalyzer : IAiImageAnalyzer
{
    public AiAnalyzerResult Analyze(IImagePixelSource image, AiAnalysisContext context, CancellationToken cancellationToken)
    {
        int width = image.Width, height = image.Height;
        RgbFloat[] previous = ArrayPool<RgbFloat>.Shared.Rent(width);
        RgbFloat[] current = ArrayPool<RgbFloat>.Shared.Rent(width);
        var rg = new PairAccumulator(); var rb = new PairAccumulator(); var gb = new PairAccumulator();
        var rgRight = new PairAccumulator(); var rgDown = new PairAccumulator(); var bgRight = new PairAccumulator(); var bgDown = new PairAccumulator();
        double[,] detail = new double[4, 3];
        int[,] detailCounts = new int[4, 3];
        try
        {
            image.CopyRow(0, previous.AsSpan(0, width));
            AccumulateSame(previous, width, ref rg, ref rb, ref gb);
            for (int y = 1; y < height; y++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                image.CopyRow(y, current.AsSpan(0, width));
                AccumulateSame(current, width, ref rg, ref rb, ref gb);
                for (int x = 0; x < width; x++)
                {
                    RgbFloat pixel = current[x];
                    RgbFloat above = previous[x];
                    rgDown.Add(pixel.R, above.G); bgDown.Add(pixel.B, above.G);
                    if (x + 1 < width) { RgbFloat right = current[x + 1]; rgRight.Add(pixel.R, right.G); bgRight.Add(pixel.B, right.G); }
                    if (y > 1 && x > 0 && x < width - 1)
                    {
                        int parity = ((y & 1) << 1) | (x & 1);
                        RgbFloat left = current[x - 1], right = current[x + 1], up = previous[x];
                        AddDetail(detail, detailCounts, parity, 0, pixel.R, left.R, right.R, up.R);
                        AddDetail(detail, detailCounts, parity, 1, pixel.G, left.G, right.G, up.G);
                        AddDetail(detail, detailCounts, parity, 2, pixel.B, left.B, right.B, up.B);
                    }
                }
                (previous, current) = (current, previous);
            }
        }
        finally
        {
            ArrayPool<RgbFloat>.Shared.Return(previous);
            ArrayPool<RgbFloat>.Shared.Return(current);
        }

        float[] channels = [(float)rg.Correlation, (float)rb.Correlation, (float)gb.Correlation];
        float[] neighbors = [(float)rgRight.Correlation, (float)rgDown.Correlation, (float)bgRight.Correlation, (float)bgDown.Correlation];
        bool detailed = context.Options.Mode != DetectionMode.Fast;
        CfaAnalysisResult cfa = detailed ? AnalyzeCfa(detail, detailCounts) : new CfaAnalysisResult();
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

    private static void AccumulateSame(RgbFloat[] row, int width, ref PairAccumulator rg, ref PairAccumulator rb, ref PairAccumulator gb)
    {
        for (int x = 0; x < width; x++) { RgbFloat p = row[x]; rg.Add(p.R, p.G); rb.Add(p.R, p.B); gb.Add(p.G, p.B); }
    }

    private static void AddDetail(double[,] sums, int[,] counts, int parity, int channel, float center, float left, float right, float up)
    {
        double value = Math.Abs(center - ((left + right + up) / 3f));
        sums[parity, channel] += value; counts[parity, channel]++;
    }

    private static CfaAnalysisResult AnalyzeCfa(double[,] sums, int[,] counts)
    {
        double[,] means = new double[4, 3];
        for (int p = 0; p < 4; p++) for (int c = 0; c < 3; c++) means[p, c] = counts[p, c] > 0 ? sums[p, c] / counts[p, c] : 0;
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

    private struct PairAccumulator
    {
        private long count; private double sumX, sumY, sumXX, sumYY, sumXY;
        public void Add(float x, float y) { count++; sumX += x; sumY += y; sumXX += x * x; sumYY += y * y; sumXY += x * y; }
        public readonly double Correlation
        {
            get
            {
                if (count < 2) return 0;
                double covariance = (count * sumXY) - (sumX * sumY);
                double varianceX = (count * sumXX) - (sumX * sumX), varianceY = (count * sumYY) - (sumY * sumY);
                double denominator = Math.Sqrt(Math.Max(0, varianceX * varianceY));
                return denominator > 1e-20 ? Math.Clamp(covariance / denominator, -1, 1) : 0;
            }
        }
    }
}
