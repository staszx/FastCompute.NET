using System.Numerics;
using AiImageForensics.Frequency;

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
        double low = 0, mid = 0, high = 0;
        for (int y = 1; y < height - 1; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (int x = 1; x < width - 1; x++)
            {
                int i = (y * width) + x;
                double gx = data[i + 1] - data[i - 1];
                double gy = data[i + width] - data[i - width];
                double lap = data[i - 1] + data[i + 1] + data[i - width] + data[i + width] - (4 * data[i]);
                low += data[i] * data[i]; mid += (gx * gx) + (gy * gy); high += lap * lap;
            }
        }
        return CreateEnergyResult(low, mid, high, [], 0, 0, 0);
    }

    private static FrequencyAnalysisResult AnalyzeFft(ReadOnlySpan<float> source, int sourceWidth, int sourceHeight, DetectionMode mode, CancellationToken cancellationToken)
    {
        int cap = mode == DetectionMode.Accurate ? 1024 : 512;
        int width = LargestPowerOfTwo(Math.Min(sourceWidth, cap));
        int height = LargestPowerOfTwo(Math.Min(sourceHeight, cap));
        if (width < 2 || height < 2) return new FrequencyAnalysisResult();
        int startX = (sourceWidth - width) / 2, startY = (sourceHeight - height) / 2;
        var transformed = new Complex[checked(width * height)];
        double mean = 0;
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) mean += source[((startY + y) * sourceWidth) + startX + x];
        mean /= transformed.Length;
        for (int y = 0; y < height; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            double wy = height > 1 ? 0.5 * (1 - Math.Cos((2 * Math.PI * y) / (height - 1))) : 1;
            for (int x = 0; x < width; x++)
            {
                double wx = width > 1 ? 0.5 * (1 - Math.Cos((2 * Math.PI * x) / (width - 1))) : 1;
                transformed[(y * width) + x] = (source[((startY + y) * sourceWidth) + startX + x] - mean) * wx * wy;
            }
        }
        Fft2D.Transform(transformed, width, height, cancellationToken);

        var power = new double[transformed.Length];
        double low = 0, mid = 0, high = 0;
        double maxRadius = Math.Sqrt((width * width / 4d) + (height * height / 4d));
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int shiftedX = (x + (width / 2)) % width;
            int shiftedY = (y + (height / 2)) % height;
            double value = transformed[(y * width) + x].Magnitude;
            value *= value;
            power[(shiftedY * width) + shiftedX] = value;
            double dx = shiftedX - (width / 2d), dy = shiftedY - (height / 2d);
            double radius = Math.Sqrt((dx * dx) + (dy * dy)) / maxRadius;
            if (radius < 0.15) low += value;
            else if (radius < 0.5) mid += value;
            else high += value;
        }

        float[] radial = CalculateRadialSpectrum(power, width, height, mode == DetectionMode.Accurate ? 64 : 32);
        float roughness = 0;
        for (int i = 1; i < radial.Length; i++) roughness += MathF.Abs(radial[i] - radial[i - 1]);
        roughness /= Math.Max(1, radial.Length - 1);
        (float peakRatio, int peakCount) = DetectPeaks(power, width, height);
        float periodicity = Math.Clamp(((peakRatio - 3f) / 12f) + (peakCount / 100f), 0, 1);
        return CreateEnergyResult(low, mid, high, radial, peakRatio, peakCount, roughness, periodicity);
    }

    internal static float[] CalculateRadialSpectrum(ReadOnlySpan<double> power, int width, int height, int binCount)
    {
        var sums = new double[binCount];
        var counts = new int[binCount];
        double cx = width / 2d, cy = height / 2d;
        double maximum = Math.Sqrt((cx * cx) + (cy * cy));
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            double dx = x - cx, dy = y - cy;
            int bin = Math.Min(binCount - 1, (int)(Math.Sqrt((dx * dx) + (dy * dy)) / maximum * binCount));
            sums[bin] += power[(y * width) + x]; counts[bin]++;
        }
        var result = new float[binCount];
        double total = 0;
        for (int i = 0; i < binCount; i++) { if (counts[i] > 0) result[i] = (float)(sums[i] / counts[i]); total += result[i]; }
        if (total > 1e-30) for (int i = 0; i < result.Length; i++) result[i] = (float)(result[i] / total);
        return result;
    }

    private static (float Ratio, int Count) DetectPeaks(ReadOnlySpan<double> power, int width, int height)
    {
        double maxRatio = 0;
        int strong = 0;
        for (int y = 2; y < height - 2; y += 2)
        for (int x = 2; x < width - 2; x += 2)
        {
            int index = (y * width) + x;
            double neighborhood = 0;
            int count = 0;
            for (int yy = -2; yy <= 2; yy++)
            for (int xx = -2; xx <= 2; xx++) if (xx != 0 || yy != 0) { neighborhood += power[index + (yy * width) + xx]; count++; }
            double ratio = power[index] / Math.Max(1e-30, neighborhood / count);
            maxRatio = Math.Max(maxRatio, ratio);
            if (ratio > 8) strong++;
        }
        return ((float)Math.Min(maxRatio, 1000), strong);
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
