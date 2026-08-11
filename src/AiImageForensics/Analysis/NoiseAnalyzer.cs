using AiImageForensics.Statistics;

namespace AiImageForensics.Analysis;

internal sealed class NoiseAnalyzer : IAiImageAnalyzer
{
    private static readonly (int X, int Y)[] Offsets = [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2)];

    public AiAnalyzerResult Analyze(IImagePixelSource image, AiAnalysisContext context, CancellationToken cancellationToken)
    {
        ReadOnlySpan<float> residual = context.GetResidual(cancellationToken);
        ReadOnlySpan<float> luminance = context.GetLinearLuminance(cancellationToken);
        DistributionStatistics stats = StatisticsMath.Calculate(residual);
        var correlations = new float[Offsets.Length];
        for (int i = 0; i < Offsets.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            correlations[i] = (float)StatisticsMath.CalculateCorrelation(residual, image.Width, image.Height, Offsets[i].X, Offsets[i].Y);
        }
        NoiseSignalModel signalModel = FitSignalModel(luminance, residual);

        float correlationStrength = correlations.Select(MathF.Abs).Average();
        float smoothNoiseEvidence = Math.Clamp((0.12f - correlationStrength) / 0.12f, 0f, 1f);
        float weakSignalEvidence = (float)Math.Clamp((0.35 - signalModel.RSquared) / 0.35, 0, 1);
        float score = (0.7f * smoothNoiseEvidence) + (0.3f * weakSignalEvidence);
        float confidence = Math.Clamp((float)Math.Log10(Math.Max(10, residual.Length)) / 7f, 0.25f, 1f);

        var details = new NoiseAnalysisResult { Statistics = stats, Autocorrelations = correlations, SignalModel = signalModel };
        return new AiAnalyzerResult
        {
            Score = score,
            Confidence = confidence,
            Details = details,
            Evidence =
            [
                new AiEvidence { Type = AiEvidenceType.Noise, Score = score, Confidence = confidence, Message = "Residual noise statistics and signal dependence were evaluated as weak forensic evidence." },
                new AiEvidence { Type = AiEvidenceType.NoiseCorrelation, Score = smoothNoiseEvidence, Confidence = confidence, Message = "Spatial residual correlations were evaluated at five fixed offsets." }
            ]
        };
    }

    internal static NoiseSignalModel FitSignalModel(ReadOnlySpan<float> signal, ReadOnlySpan<float> residual)
    {
        const int binCount = 16;
        Span<double> signalSums = stackalloc double[binCount];
        Span<double> residualSums = stackalloc double[binCount];
        Span<double> residualSquares = stackalloc double[binCount];
        Span<int> counts = stackalloc int[binCount];
        for (int i = 0; i < signal.Length; i++)
        {
            int bin = Math.Clamp((int)(signal[i] * binCount), 0, binCount - 1);
            signalSums[bin] += signal[i]; residualSums[bin] += residual[i]; residualSquares[bin] += residual[i] * residual[i]; counts[bin]++;
        }

        Span<double> x = stackalloc double[binCount];
        Span<double> y = stackalloc double[binCount];
        int valid = 0;
        for (int bin = 0; bin < binCount; bin++)
        {
            if (counts[bin] < 2) continue;
            double meanResidual = residualSums[bin] / counts[bin];
            x[valid] = signalSums[bin] / counts[bin];
            y[valid] = Math.Max(0, (residualSquares[bin] / counts[bin]) - (meanResidual * meanResidual));
            valid++;
        }
        if (valid < 2) return default;

        double meanX = 0, meanY = 0;
        for (int i = 0; i < valid; i++) { meanX += x[i]; meanY += y[i]; }
        meanX /= valid; meanY /= valid;
        double covariance = 0, varianceX = 0;
        for (int i = 0; i < valid; i++) { double dx = x[i] - meanX; covariance += dx * (y[i] - meanY); varianceX += dx * dx; }
        double a = varianceX > 1e-20 ? covariance / varianceX : 0;
        double b = meanY - (a * meanX);
        double ssResidual = 0, ssTotal = 0;
        for (int i = 0; i < valid; i++) { double d = y[i] - ((a * x[i]) + b); ssResidual += d * d; double total = y[i] - meanY; ssTotal += total * total; }
        double rSquared = ssTotal > 1e-20 ? 1 - (ssResidual / ssTotal) : 0;
        return new NoiseSignalModel { A = a, B = b, RSquared = Math.Clamp(rSquared, 0, 1) };
    }
}
