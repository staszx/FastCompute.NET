namespace AiImageForensics;

/// <summary>Supplies a fixed, caller-defined transformation suite for robustness measurement.</summary>
public interface IAiImageTransformationProvider
{
    /// <summary>Visits each transformation exactly once; the provider owns transformed-image lifetimes during the callback.</summary>
    void VisitTransformations(IImagePixelSource image, Action<string, IImagePixelSource> visitor, CancellationToken cancellationToken = default);
}

/// <summary>One detector result from a named robustness transformation.</summary>
public sealed class AiRobustnessCase
{
    /// <summary>Transformation name.</summary>
    public string Name { get; init; } = string.Empty;
    /// <summary>Detection score.</summary>
    public float Score { get; init; }
    /// <summary>Detection confidence.</summary>
    public float Confidence { get; init; }
}

/// <summary>Detector stability over a predefined transformation suite.</summary>
public sealed class AiRobustnessResult
{
    /// <summary>Untransformed score.</summary>
    public float OriginalScore { get; init; }
    /// <summary>Minimum observed score.</summary>
    public float MinimumScore { get; init; }
    /// <summary>Maximum observed score.</summary>
    public float MaximumScore { get; init; }
    /// <summary>Mean observed score including the original.</summary>
    public float MeanScore { get; init; }
    /// <summary>Normalized stability, where one means no score variation.</summary>
    public float Stability { get; init; }
    /// <summary>Named transformed cases.</summary>
    public IReadOnlyList<AiRobustnessCase> Cases { get; init; } = Array.Empty<AiRobustnessCase>();
}

/// <summary>Measures score stability without optimizing transformations against the detector.</summary>
public static class AiRobustnessTester
{
    /// <summary>Runs the original image and every transformation supplied by the fixed provider.</summary>
    public static AiRobustnessResult Test(IImagePixelSource image, IAiImageTransformationProvider transformations, AiDetectionOptions? detectionOptions = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(transformations);
        detectionOptions ??= new AiDetectionOptions();
        AiDetectionResult original = AiDetector.Detect(image, detectionOptions, cancellationToken);
        var cases = new List<AiRobustnessCase>();
        transformations.VisitTransformations(image, (name, transformed) =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (string.IsNullOrWhiteSpace(name)) throw new ArgumentException("Transformation names must be non-empty.", nameof(transformations));
            ArgumentNullException.ThrowIfNull(transformed);
            AiDetectionResult detected = AiDetector.Detect(transformed, detectionOptions, cancellationToken);
            cases.Add(new AiRobustnessCase { Name = name, Score = detected.Score, Confidence = detected.Confidence });
        }, cancellationToken);

        float minimum = original.Score, maximum = original.Score, sum = original.Score;
        for (int i = 0; i < cases.Count; i++) { minimum = Math.Min(minimum, cases[i].Score); maximum = Math.Max(maximum, cases[i].Score); sum += cases[i].Score; }
        return new AiRobustnessResult
        {
            OriginalScore = original.Score, MinimumScore = minimum, MaximumScore = maximum,
            MeanScore = sum / (cases.Count + 1), Stability = Math.Clamp(1 - (maximum - minimum), 0, 1), Cases = cases.ToArray()
        };
    }
}
