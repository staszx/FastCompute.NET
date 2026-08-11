namespace AiImageForensics.Tests;

public sealed class DetectorTests
{
    [Test]
    public void Detect_IsDeterministicBoundedAndHonorsThreshold()
    {
        SyntheticPixelSource image = SyntheticPixelSource.Checkerboard(64, 64);
        AiDetectionResult first = AiDetector.Detect(image, new AiDetectionOptions { DetectionThreshold = 0 });
        AiDetectionResult second = AiDetector.Detect(image, new AiDetectionOptions { DetectionThreshold = 1 });
        AiDetectionResult repeated = AiDetector.Detect(image, new AiDetectionOptions { DetectionThreshold = 0 });
        Assert.Multiple(() =>
        {
            Assert.That(first.Score, Is.InRange(0, 1));
            Assert.That(first.Confidence, Is.InRange(0, 1));
            Assert.That(first.IsLikelyAi, Is.True);
            Assert.That(second.IsLikelyAi, Is.EqualTo(second.Score >= 1));
            Assert.That(repeated.Score, Is.EqualTo(first.Score));
            Assert.That(repeated.Confidence, Is.EqualTo(first.Confidence));
        });
    }

    [Test]
    public void DisabledAnalyzers_AreNotIncluded()
    {
        AiDetectionResult result = AiDetector.Detect(SyntheticPixelSource.Solid(16, 16, 0.2f), new AiDetectionOptions
        {
            Mode = DetectionMode.Fast, AnalyzeMetadata = false, AnalyzeFrequency = false,
            AnalyzeCameraPipeline = false, AnalyzeSpatialStatistics = false, AnalyzeNoise = true
        });
        Assert.That(result.Evidence, Has.All.Matches<AiEvidence>(e => e.Type is AiEvidenceType.Noise or AiEvidenceType.NoiseCorrelation));
    }

    [Test]
    public void Cancellation_IsObserved() =>
        Assert.Throws<OperationCanceledException>(() => AiDetector.Detect(SyntheticPixelSource.Solid(32, 32, 0.5f), cancellationToken: new CancellationToken(true)));

    [Test]
    public void FeatureOrder_IsStable()
    {
        AiFeatureVector first = AiAnalyzer.ExtractFeatures(SyntheticPixelSource.Solid(16, 16, 0.5f));
        AiFeatureVector second = AiAnalyzer.ExtractFeatures(SyntheticPixelSource.Solid(16, 16, 0.5f));
        Assert.Multiple(() =>
        {
            Assert.That(first.Names, Is.EqualTo(second.Names));
            Assert.That(first.Values, Has.Length.EqualTo(first.Names.Length));
            Assert.That(first.Names[0], Is.EqualTo("noise.mean"));
            Assert.That(first.Names[^1], Is.EqualTo("metadata.explicit_ai"));
        });
    }

    [Test]
    public void InvalidOptions_AreRejected()
    {
        SyntheticPixelSource image = SyntheticPixelSource.Solid(8, 8, 0.5f);
        Assert.Throws<ArgumentOutOfRangeException>(() => AiDetector.Detect(image, new AiDetectionOptions { DetectionThreshold = 1.1f }));
        Assert.Throws<ArgumentOutOfRangeException>(() => AiDetector.Detect(image, new AiDetectionOptions { MaxDegreeOfParallelism = 0 }));
    }
}
