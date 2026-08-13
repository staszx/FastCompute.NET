using System.Numerics;
using FastCompute;
using FastCompute.ImageProcessing;

namespace AiImageForensics.Analysis;

internal sealed class FrequencyAnalyzer : IAiImageAnalyzer
{
    public AiAnalyzerResult Analyze(IImagePixelSource image, AiAnalysisContext context, CancellationToken cancellationToken)
    {
        ReadOnlySpan<float> luminance = context.GetLinearLuminance(cancellationToken);
        FrequencyAnalysisResult details = context.Options.Mode == DetectionMode.Fast
            ? AnalyzeBasic(luminance, image.Width, image.Height, cancellationToken)
            : AnalyzeFft(luminance, image.Width, image.Height, context.Options.Mode, cancellationToken);

        float spectralUniformity = Math.Clamp((details.HighToTotal - 0.08f) / 0.42f, 0, 1);
        float periodicEvidence = Math.Clamp(details.PeriodicityScore, 0, 1);
        float roughnessEvidence = Math.Clamp(details.SpectrumRoughness * 3f, 0, 1);
        float score = (0.45f * spectralUniformity) + (0.35f * periodicEvidence) + (0.2f * roughnessEvidence);
        float confidence = context.Options.Mode == DetectionMode.Fast ? 0.45f : Math.Clamp((float)Math.Log2(Math.Max(4, details.RadialSpectrum.Count)) / 6f, 0.55f, 1f);
        return new AiAnalyzerResult
        {
            Score = score,
            Confidence = confidence,
            Details = details,
            Evidence = [new AiEvidence { Type = AiEvidenceType.Frequency, Score = score, Confidence = confidence, Message = context.Options.Mode == DetectionMode.Fast ? "Basic multi-band spatial frequency energy was evaluated." : "Windowed FFT, radial energy, spectral roughness, and non-specific periodic peaks were evaluated." }]
        };
    }

    private static FrequencyAnalysisResult AnalyzeBasic(ReadOnlySpan<float> data, int width, int height, CancellationToken cancellationToken)
    {
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Auto,
            CancellationToken = cancellationToken
        };
        float[] gradient = ImageFilters.GradientMagnitude(data, width, height, options);
        float[] laplacian = ImageFilters.Laplacian(data, width, height, options);
        double low = Compute.SumOfSquares(data, options);
        double mid = Compute.SumOfSquares(gradient, options);
        double high = Compute.SumOfSquares(laplacian, options);
        return CreateEnergyResult(low, mid, high, [], 0, 0, 0);
    }

    private static FrequencyAnalysisResult AnalyzeFft(ReadOnlySpan<float> source, int sourceWidth, int sourceHeight, DetectionMode mode, CancellationToken cancellationToken)
    {
        int cap = mode == DetectionMode.Accurate ? 1024 : 512;
        int width = LargestPowerOfTwo(Math.Min(sourceWidth, cap));
        int height = LargestPowerOfTwo(Math.Min(sourceHeight, cap));
        if (width < 2 || height < 2) return new FrequencyAnalysisResult();
        var options = new ComputeOptions
        {
            Backend = ComputeBackendKind.Auto,
            CancellationToken = cancellationToken
        };
        Complex32[] transformed = ImageSpectrumOperations.PrepareSpectrumInput(source, sourceWidth, sourceHeight, width, height, options);
        Compute.Fft2DInPlace(transformed, width, height, options: options);
        float[] power = Compute.PowerSpectrum(transformed, options);
        float[] radial = ImageSpectrumOperations.CalculateRadialSpectrum(
            power,
            width,
            height,
            mode == DetectionMode.Accurate ? 64 : 32,
            out FrequencyBandEnergy energy,
            options: options);
        float roughness = Compute.MeanAbsoluteDifference(radial, options);
        SpectrumPeakMetrics peaks = ImageSpectrumOperations.CalculatePeakMetrics(power, width, height, options: options);
        float peakRatio = peaks.MaximumRatio;
        int peakCount = peaks.StrongPeakCount;
        float periodicity = Math.Clamp(((peakRatio - 3f) / 12f) + (peakCount / 100f), 0, 1);
        return CreateEnergyResult(energy.Low, energy.Mid, energy.High, radial, peakRatio, peakCount, roughness, periodicity);
    }

    private static FrequencyAnalysisResult CreateEnergyResult(double low, double mid, double high, IReadOnlyList<float> radial, float peakRatio, int peakCount, float roughness, float periodicity = 0)
    {
        double total = low + mid + high;
        if (total <= 1e-30) return new FrequencyAnalysisResult { RadialSpectrum = radial };
        return new FrequencyAnalysisResult
        {
            LowFrequencyEnergy = (float)(low / total), MidFrequencyEnergy = (float)(mid / total), HighFrequencyEnergy = (float)(high / total),
            HighToTotal = (float)(high / total), MidToTotal = (float)(mid / total), HighToLow = (float)(high / Math.Max(low, 1e-30)),
            RadialSpectrum = radial, PeakRatio = peakRatio, StrongPeakCount = peakCount, SpectrumRoughness = roughness, PeriodicityScore = periodicity
        };
    }

    private static int LargestPowerOfTwo(int value) => value <= 0 ? 0 : 1 << BitOperations.Log2((uint)value);
}
