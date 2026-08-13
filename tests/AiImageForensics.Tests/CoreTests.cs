using FastCompute;
using FastCompute.ImageProcessing;

namespace AiImageForensics.Tests;

public sealed class CoreTests
{
    [TestCase(0f, 0f)]
    [TestCase(1f, 1f)]
    [TestCase(0.5f, 0.21404114f)]
    public void SrgbToLinear_ReturnsExpectedValues(float input, float expected) =>
        Assert.That(ColorMath.SrgbToLinear(input), Is.EqualTo(expected).Within(1e-6));

    [Test]
    public void Luminance_UsesRec709Weights()
    {
        Assert.Multiple(() =>
        {
            Assert.That(ColorMath.GetLuminance(new RgbFloat(1, 0, 0)), Is.EqualTo(0.2126f).Within(1e-6));
            Assert.That(ColorMath.GetLuminance(new RgbFloat(0, 1, 0)), Is.EqualTo(0.7152f).Within(1e-6));
            Assert.That(ColorMath.GetLuminance(new RgbFloat(0, 0, 1)), Is.EqualTo(0.0722f).Within(1e-6));
            Assert.That(ColorMath.GetLuminance(new RgbFloat(0.4f, 0.4f, 0.4f)), Is.EqualTo(0.4f).Within(1e-6));
        });
    }

    [Test]
    public void Statistics_CalculateMomentsAndCorrelationSafely()
    {
        float[] values = [1, 2, 3, 4];
        StatisticsResult result = Compute.CalculateStatistics(values);
        Assert.Multiple(() =>
        {
            Assert.That(result.Mean, Is.EqualTo(2.5).Within(1e-12));
            Assert.That(result.Variance, Is.EqualTo(1.25).Within(1e-12));
            Assert.That(result.StandardDeviation, Is.EqualTo(Math.Sqrt(1.25)).Within(1e-12));
            Assert.That(result.Skewness, Is.EqualTo(0).Within(1e-12));
            Assert.That(Compute.Correlation(values, values), Is.EqualTo(1).Within(1e-12));
            Assert.That(ImageStatistics.SpatialCorrelation(new float[16], 4, 4, 1, 0), Is.Zero);
        });
    }

    [Test]
    public void ConstantImage_ProducesFiniteNoiseAndSpatialFeatures()
    {
        AiAnalysisResult result = AiAnalyzer.Analyze(SyntheticPixelSource.Solid(32, 32, 0.5f));
        Assert.Multiple(() =>
        {
            Assert.That(result.Noise.Statistics.Variance, Is.Zero.Within(1e-12));
            Assert.That(result.Spatial.LaplacianVariance, Is.Zero.Within(1e-12));
            Assert.That(result.AiScore, Is.InRange(0, 1));
            Assert.That(result.Confidence, Is.InRange(0, 1));
        });
    }

    [Test]
    public void NativeLinearRgb_UsesNativeAnalysisPath()
    {
        var pixels = Enumerable.Repeat(new Rgb(0.2f, 0.4f, 0.6f), 32 * 32).ToArray();
        Image<Rgb> image = Image<Rgb>.Load(pixels, 32, 32, ColorEncoding.Linear);
        AiAnalysisResult result = image.AnalyzeAi(new AiAnalysisOptions { Mode = DetectionMode.Fast });
        Assert.That(result.AiScore, Is.InRange(0, 1));
    }

    [Test]
    public void CameraSimulation_StrengthensCfaEvidenceOnSyntheticRgb()
    {
        const int width = 64, height = 64;
        var random = new Random(7);
        var pixels = new Rgb[width * height];
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            float baseValue = (x + y) / 126f;
            pixels[(y * width) + x] = new Rgb(
                Math.Clamp(baseValue + ((float)random.NextDouble() - 0.5f) * 0.08f, 0, 1),
                Math.Clamp(baseValue + ((float)random.NextDouble() - 0.5f) * 0.08f, 0, 1),
                Math.Clamp(baseValue + ((float)random.NextDouble() - 0.5f) * 0.08f, 0, 1));
        }
        Image<Rgb> original = Image<Rgb>.Load(pixels, width, height, ColorEncoding.Linear);
        Image<Rgb> simulated = CameraSimulator.SimulateCamera(original, new CameraSimulationOptions { BayerPattern = BayerPattern.Rggb });
        CameraAnalysisResult originalResult = original.AnalyzeAi(new AiAnalysisOptions { Mode = DetectionMode.Balanced }).Camera;
        CameraAnalysisResult simulatedResult = simulated.AnalyzeAi(new AiAnalysisOptions { Mode = DetectionMode.Balanced }).Camera;
        Assert.That(simulatedResult.CfaScore, Is.GreaterThan(originalResult.CfaScore));
    }

    [Test]
    public void AccurateMode_UsesBlockAndMultiScaleAnalysisDeterministically()
    {
        SyntheticPixelSource source = SyntheticPixelSource.Sine(300, 260);
        var options = new AiAnalysisOptions { Mode = DetectionMode.Accurate, MaxDegreeOfParallelism = 2 };
        AiAnalysisResult first = AiAnalyzer.Analyze(source, options);
        AiAnalysisResult second = AiAnalyzer.Analyze(source, options);
        Assert.Multiple(() =>
        {
            Assert.That(first.AiScore, Is.EqualTo(second.AiScore));
            Assert.That(first.Evidence.Any(e => e.Message.StartsWith("Accurate mode", StringComparison.Ordinal)), Is.True);
            Assert.That(first.Frequency.RadialSpectrum, Has.Count.EqualTo(64));
        });
    }

    [TestCaseSource(nameof(FrequencySources))]
    public void BalancedFrequencyAnalysis_ReturnsNormalizedSpectrum(SyntheticPixelSource source)
    {
        AiAnalysisResult result = AiAnalyzer.Analyze(source, new AiAnalysisOptions { Mode = DetectionMode.Balanced });
        Assert.Multiple(() =>
        {
            Assert.That(result.Frequency.RadialSpectrum, Has.Count.EqualTo(32));
            Assert.That(result.Frequency.LowFrequencyEnergy + result.Frequency.MidFrequencyEnergy + result.Frequency.HighFrequencyEnergy, Is.EqualTo(1).Within(1e-5).Or.EqualTo(0));
        });
    }

    private static IEnumerable<SyntheticPixelSource> FrequencySources()
    {
        yield return SyntheticPixelSource.Solid(64, 64, 0.5f);
        yield return SyntheticPixelSource.Sine(64, 64);
        yield return SyntheticPixelSource.Checkerboard(64, 64);
    }
}
