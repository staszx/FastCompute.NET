using BenchmarkDotNet.Attributes;
using FastCompute.ImageProcessing;

namespace AiImageForensics.Benchmarks;

[MemoryDiagnoser]
public class ForensicsBenchmarks
{
    private Image<Rgb24> image = null!;

    public IEnumerable<string> ImageSizes => ["1920x1080", "4000x3000", "6000x4000"];

    [ParamsSource(nameof(ImageSizes))]
    public string ImageSize { get; set; } = "1920x1080";

    [GlobalSetup]
    public void Setup()
    {
        string[] parts = ImageSize.Split('x');
        int width = int.Parse(parts[0]), height = int.Parse(parts[1]);
        var pixels = GC.AllocateUninitializedArray<Rgb24>(checked(width * height));
        var random = new Random(1);
        for (int i = 0; i < pixels.Length; i++) pixels[i] = new Rgb24((byte)random.Next(256), (byte)random.Next(256), (byte)random.Next(256));
        image = Image<Rgb24>.Load(pixels, width, height);
    }

    [Benchmark]
    public AiDetectionResult NoiseAnalyzer() => image.DetectAi(Only(noise: true));

    [Benchmark]
    public AiDetectionResult FrequencyAnalyzer() => image.DetectAi(Only(frequency: true));

    [Benchmark]
    public AiDetectionResult CameraAnalyzer() => image.DetectAi(Only(camera: true));

    [Benchmark]
    public AiDetectionResult FullDetectFast() => image.DetectAi(new AiDetectionOptions { Mode = DetectionMode.Fast, AnalyzeMetadata = false });

    [Benchmark]
    public AiDetectionResult FullDetectBalanced() => image.DetectAi(new AiDetectionOptions { Mode = DetectionMode.Balanced, AnalyzeMetadata = false });

    private static AiDetectionOptions Only(bool noise = false, bool frequency = false, bool camera = false) => new()
    {
        Mode = frequency ? DetectionMode.Balanced : DetectionMode.Fast,
        AnalyzeNoise = noise, AnalyzeFrequency = frequency, AnalyzeCameraPipeline = camera,
        AnalyzeSpatialStatistics = false, AnalyzeMetadata = false
    };
}
