# AiImageForensics

`AiImageForensics` is a deterministic image-forensics library for .NET 8. It
combines several weak statistical evidence groups; it does not use a neural
network or an external AI service.

Install the core package when an application already provides
`IImagePixelSource`. Install `AiImageForensics.ImageSharp` for decoding and
ImageSharp extension methods. The core project has no ImageSharp dependency.

## Simple detection

```csharp
using AiImageForensics.ImageSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

using Image<Rgba32> image = Image.Load<Rgba32>("image.jpg");
AiDetectionResult result = image.DetectAi();

Console.WriteLine($"AI score: {result.Score:P1}");
Console.WriteLine($"Confidence: {result.Confidence:P1}");
```

## Detailed analysis

```csharp
AiAnalysisResult analysis = image.AnalyzeAi(
    new AiAnalysisOptions { Mode = DetectionMode.Accurate });

foreach (AiEvidence evidence in analysis.Evidence)
    Console.WriteLine($"{evidence.Type}: {evidence.Score:F2}");
```

## Native FastCompute images

```csharp
Image<Rgb> linear = AiImage.Load<Rgb>("image.jpg", ColorEncoding.Linear);
AiDetectionResult result = linear.DetectAi();
```

This path lets the analysis reuse FastCompute's native image representation.
The analysis context pools one full-resolution luminance plane and creates the
residual plane only when required. Balanced/Accurate FFT input is a bounded,
deterministic center crop.

## Camera simulation

```csharp
image.Mutate(x => x.SimulateCamera(new CameraSimulationOptions
{
    ShotNoise = 0.002f,
    ReadNoise = 0.0005f,
    OpticalBlur = 0.5f,
    Sharpening = 0.15f,
    RandomSeed = 1
}));
```

Camera simulation is intended for robustness testing. It implements optical
blur, signal-dependent/read noise, Bayer sampling, basic demosaicing, and
sharpening. Chromatic aberration and vignetting options are reserved for a
future implementation. It never creates fake camera metadata.

## Interpretation and limitations

The public score is a normalized heuristic detection score, not a scientifically
calibrated probability. Detection can produce false positives and false
negatives. Current weights are conservative placeholders and must be calibrated
against a representative, versioned validation dataset before production use.

Metadata only contributes when it contains explicit recognized provenance.
Missing EXIF, camera make/model, lens, or GPS is never treated as AI evidence.
