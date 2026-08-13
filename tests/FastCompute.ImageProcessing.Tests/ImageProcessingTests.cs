using FastCompute.ImageProcessing;
using SixLabors.ImageSharp.Formats.Png;
using System.Runtime.InteropServices;
using SixLabors.ImageSharp.Processing;
using System.Numerics;
using ImageSharpImage = SixLabors.ImageSharp.Image;
using ImageSharpRgb24 = SixLabors.ImageSharp.PixelFormats.Rgb24;

namespace FastCompute.Tests;

public sealed class ImageProcessingTests
{
    [Test]
    public void NativePixelFormats_HaveExpectedPhysicalSizes()
    {
        Assert.Multiple(() =>
        {
            Assert.That(Marshal.SizeOf<Rgb24>(), Is.EqualTo(3));
            Assert.That(Marshal.SizeOf<Rgb>(), Is.EqualTo(12));
            Assert.That(Marshal.SizeOf<Gray8>(), Is.EqualTo(1));
            Assert.That(Marshal.SizeOf<GrayF32>(), Is.EqualTo(4));
        });
    }

    [Test]
    public void PixelFormats_ConvertBetweenByteAndFloatRepresentations()
    {
        Rgb24[] source =
        [
            new Rgb24(255, 0, 0),
            new Rgb24(10, 127, 240)
        ];
        var floating = new Rgb[source.Length];
        var roundTrip = new Rgb24[source.Length];

        PixelConverter.Convert<Rgb24, Rgb>(source, floating);
        PixelConverter.Convert<Rgb, Rgb24>(floating, roundTrip);

        Assert.That(roundTrip, Is.EqualTo(source));
    }

    [Test]
    public void PixelConverter_SimdKernelsMatchScalarReferenceAndTail()
    {
        Rgb[] rgb = Enumerable.Range(0, 19)
            .Select(i => new Rgb(
                ((i * 17) % 101) / 100f,
                ((i * 31) % 103) / 102f,
                ((i * 47) % 107) / 106f))
            .ToArray();
        var grayF32 = new GrayF32[rgb.Length];
        var gray8 = new Gray8[rgb.Length];
        var rgb24 = new Rgb24[rgb.Length];
        var roundTrip = new Rgb[rgb.Length];

        var simd = new ComputeOptions { Backend = ComputeBackendKind.Simd };
        PixelConverter.Convert<Rgb, GrayF32>(rgb, grayF32, options: simd);
        PixelConverter.Convert<Rgb, Gray8>(rgb, gray8, options: simd);
        PixelConverter.Convert<Rgb, Rgb24>(rgb, rgb24, options: simd);
        PixelConverter.Convert<Rgb24, Rgb>(rgb24, roundTrip, options: simd);

        for (int i = 0; i < rgb.Length; i++)
        {
            float expectedGray =
                (0.2126f * rgb[i].Red) +
                (0.7152f * rgb[i].Green) +
                (0.0722f * rgb[i].Blue);
            Assert.Multiple(() =>
            {
                Assert.That(grayF32[i].Value, Is.EqualTo(expectedGray).Within(2e-6f));
                Assert.That(gray8[i].Value, Is.EqualTo((byte)Math.Clamp((int)MathF.Round(expectedGray * 255f), 0, 255)));
                Assert.That(roundTrip[i].Red, Is.EqualTo(rgb24[i].Red / 255f).Within(1e-7f));
                Assert.That(roundTrip[i].Green, Is.EqualTo(rgb24[i].Green / 255f).Within(1e-7f));
                Assert.That(roundTrip[i].Blue, Is.EqualTo(rgb24[i].Blue / 255f).Within(1e-7f));
            });
        }
    }

    [Test]
    public void PixelConverter_GraySimdKernelsMatchScalarReferenceAndTail()
    {
        Gray8[] gray8 = Enumerable.Range(0, 35).Select(i => new Gray8((byte)((i * 23) % 256))).ToArray();
        var grayF32 = new GrayF32[gray8.Length];
        var rgb = new Rgb[gray8.Length];
        var rgb24 = new Rgb24[gray8.Length];
        var directRgb = new Rgb[gray8.Length];
        var directRgb24 = new Rgb24[gray8.Length];
        var roundTrip = new Gray8[gray8.Length];

        var simd = new ComputeOptions { Backend = ComputeBackendKind.Simd };
        PixelConverter.Convert<Gray8, GrayF32>(gray8, grayF32, options: simd);
        PixelConverter.Convert<GrayF32, Rgb>(grayF32, rgb, options: simd);
        PixelConverter.Convert<GrayF32, Rgb24>(grayF32, rgb24, options: simd);
        PixelConverter.Convert<Gray8, Rgb>(gray8, directRgb, options: simd);
        PixelConverter.Convert<Gray8, Rgb24>(gray8, directRgb24, options: simd);
        PixelConverter.Convert<GrayF32, Gray8>(grayF32, roundTrip, options: simd);

        for (int i = 0; i < gray8.Length; i++)
        {
            float expected = gray8[i].Value / 255f;
            Assert.Multiple(() =>
            {
                Assert.That(grayF32[i].Value, Is.EqualTo(expected).Within(1e-7f));
                Assert.That(rgb[i].Red, Is.EqualTo(expected).Within(1e-7f));
                Assert.That(rgb[i].Green, Is.EqualTo(expected).Within(1e-7f));
                Assert.That(rgb[i].Blue, Is.EqualTo(expected).Within(1e-7f));
                Assert.That(rgb24[i].Red, Is.EqualTo(gray8[i].Value));
                Assert.That(directRgb[i].Red, Is.EqualTo(expected).Within(1e-7f));
                Assert.That(directRgb[i].Green, Is.EqualTo(expected).Within(1e-7f));
                Assert.That(directRgb[i].Blue, Is.EqualTo(expected).Within(1e-7f));
                Assert.That(directRgb24[i].Red, Is.EqualTo(gray8[i].Value));
                Assert.That(directRgb24[i].Green, Is.EqualTo(gray8[i].Value));
                Assert.That(directRgb24[i].Blue, Is.EqualTo(gray8[i].Value));
                Assert.That(roundTrip[i].Value, Is.EqualTo(gray8[i].Value));
            });
        }
    }

    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    [TestCase(ComputeBackendKind.Simd)]
    [TestCase(ComputeBackendKind.Gpu)]
    public void PixelConverter_AllBackendsMatchScalar(ComputeBackendKind backend)
    {
        Rgb24[] source = Enumerable.Range(0, 4099)
            .Select(index => new Rgb24(
                (byte)(index * 17),
                (byte)(index * 29 + 3),
                (byte)(index * 43 + 7)))
            .ToArray();
        var expected = new GrayF32[source.Length];
        var actual = new GrayF32[source.Length];

        PixelConverter.Convert<Rgb24, GrayF32>(
            source,
            expected,
            options: new ComputeOptions { Backend = ComputeBackendKind.Scalar });
        PixelConverter.Convert<Rgb24, GrayF32>(
            source,
            actual,
            options: new ComputeOptions { Backend = backend });

        for (int index = 0; index < expected.Length; index++)
            Assert.That(actual[index].Value, Is.EqualTo(expected[index].Value).Within(2e-6f));
    }

    [Test]
    [Category("GPU")]
    public void ImageProcessing_GpuMatchesCpuForConversionsAndSpatialOperations()
    {
        var gpu = new ComputeOptions { Backend = ComputeBackendKind.Gpu };
        Rgb24[] pixels = Enumerable.Range(0, 77)
            .Select(index => new Rgb24(
                (byte)(index * 17),
                (byte)(index * 29 + 3),
                (byte)(index * 43 + 7)))
            .ToArray();
        Image<Rgb24> image = Image<Rgb24>.Load(pixels, 11, 7);

        Image<Gray8> cpuGray8 = image.ToGrayscale8(options: new ComputeOptions { Backend = ComputeBackendKind.Simd });
        Image<Gray8> gpuGray8 = image.ToGrayscale8(options: gpu);
        Image<GrayF32> cpuGray = image.ToGrayscaleF32(ColorEncoding.Linear, new ComputeOptions { Backend = ComputeBackendKind.ParallelCpu });
        Image<GrayF32> gpuGray = image.ToGrayscaleF32(ColorEncoding.Linear, gpu);
        Image<Rgb> cpuLinearRgb = image.ToRgbF32(ColorEncoding.Linear, new ComputeOptions { Backend = ComputeBackendKind.ParallelCpu });
        Image<Rgb> gpuLinearRgb = image.ToRgbF32(ColorEncoding.Linear, gpu);
        Image<Rgb> cpuSrgbRgb = cpuLinearRgb.ToSrgb(new ComputeOptions { Backend = ComputeBackendKind.ParallelCpu });
        Image<Rgb> gpuSrgbRgb = gpuLinearRgb.ToSrgb(gpu);
        Image<Rgb24> gpuRoundTrip = gpuGray.ToRgb24(ColorEncoding.Srgb, gpu);
        Image<GrayF32> cpuBlur = cpuGray.BoxBlur(3, options: new ComputeOptions { Backend = ComputeBackendKind.Simd });
        Image<GrayF32> gpuBlur = gpuGray.BoxBlur(3, options: gpu);
        Image<GrayF32> cpuDownsample = cpuGray.Downsample(5, 3, options: new ComputeOptions { Backend = ComputeBackendKind.Simd });
        Image<GrayF32> gpuDownsample = gpuGray.Downsample(5, 3, options: gpu);
        Image<GrayF32> cpuResidual = cpuGray.Subtract(cpuBlur, options: new ComputeOptions { Backend = ComputeBackendKind.Simd });
        Image<GrayF32> gpuResidual = gpuGray.Subtract(gpuBlur, options: gpu);
        Image<GrayF32> cpuWideBlur = cpuGray.BoxBlur(7, options: new ComputeOptions { Backend = ComputeBackendKind.Simd });
        Image<GrayF32> gpuWideBlur = gpuGray.BoxBlur(7, options: gpu);

        Assert.Multiple(() =>
        {
            Assert.That(gpuGray8.Pixels.ToArray(), Is.EqualTo(cpuGray8.Pixels.ToArray()));
            Assert.That(
                gpuGray.Pixels.ToArray().Select(pixel => pixel.Value),
                Is.EqualTo(cpuGray.Pixels.ToArray().Select(pixel => pixel.Value)).Within(2e-6f));
            Assert.That(gpuRoundTrip.Pixels.Length, Is.EqualTo(pixels.Length));
            Assert.That(
                MemoryMarshal.Cast<Rgb, float>(gpuSrgbRgb.Pixels.Span).ToArray(),
                Is.EqualTo(MemoryMarshal.Cast<Rgb, float>(cpuSrgbRgb.Pixels.Span).ToArray()).Within(3e-6f));
            Assert.That(
                gpuBlur.Pixels.ToArray().Select(pixel => pixel.Value),
                Is.EqualTo(cpuBlur.Pixels.ToArray().Select(pixel => pixel.Value)).Within(3e-6f));
            Assert.That(
                gpuDownsample.Pixels.ToArray().Select(pixel => pixel.Value),
                Is.EqualTo(cpuDownsample.Pixels.ToArray().Select(pixel => pixel.Value)).Within(3e-6f));
            Assert.That(
                gpuResidual.Pixels.ToArray().Select(pixel => pixel.Value),
                Is.EqualTo(cpuResidual.Pixels.ToArray().Select(pixel => pixel.Value)).Within(4e-6f));
            Assert.That(
                gpuWideBlur.Pixels.ToArray().Select(pixel => pixel.Value),
                Is.EqualTo(cpuWideBlur.Pixels.ToArray().Select(pixel => pixel.Value)).Within(6e-6f));
        });
    }

    [Test]
    public void ExplicitSimd_RejectsNonlinearColorTransferInsteadOfSilentlyUsingScalarLoop()
    {
        Image<Rgb24> source = Image<Rgb24>.Load(
            [new Rgb24(10, 20, 30)],
            1,
            1,
            ColorEncoding.Srgb);

        ComputeBackendNotSupportedException? exception = Assert.Throws<ComputeBackendNotSupportedException>(
            () => source.ToGrayscaleF32(
                ColorEncoding.Linear,
                new ComputeOptions { Backend = ComputeBackendKind.Simd }));

        Assert.That(exception!.Backend, Is.EqualTo(ComputeBackendKind.Simd));
    }

    [Test]
    [Category("GPU")]
    public void ImageBuffer_KeepsMultiStagePipelineOnGpuUntilDownload()
    {
        Rgb24[] pixels = Enumerable.Range(0, 165)
            .Select(index => new Rgb24(
                (byte)(index * 11 + 1),
                (byte)(index * 19 + 2),
                (byte)(index * 31 + 3)))
            .ToArray();
        Image<Rgb24> source = Image<Rgb24>.Load(pixels, 15, 11);
        using ComputeContext context = ComputeContext.Create();
        using ImageBuffer<Rgb24> resident = source.UploadToGpu(context);
        using ImageBuffer<GrayF32> luminance = resident.ToGrayscaleF32(ColorEncoding.Linear);
        using ImageBuffer<GrayF32> blur = luminance.BoxBlur(2);
        using ImageBuffer<GrayF32> residual = luminance.Subtract(blur);
        using ImageBuffer<GrayF32> reduced = residual.Downsample(5, 3);

        Image<GrayF32> actual = reduced.Download();
        Image<GrayF32> cpuLuminance = source.ToGrayscaleF32(
            ColorEncoding.Linear,
            new ComputeOptions { Backend = ComputeBackendKind.ParallelCpu });
        Image<GrayF32> cpuBlur = cpuLuminance.BoxBlur(
            2,
            options: new ComputeOptions { Backend = ComputeBackendKind.Simd });
        Image<GrayF32> expected = cpuLuminance
            .Subtract(cpuBlur, options: new ComputeOptions { Backend = ComputeBackendKind.Simd })
            .Downsample(5, 3, options: new ComputeOptions { Backend = ComputeBackendKind.Simd });

        Assert.Multiple(() =>
        {
            Assert.That(actual.Width, Is.EqualTo(5));
            Assert.That(actual.Height, Is.EqualTo(3));
            Assert.That(
                actual.Pixels.ToArray().Select(pixel => pixel.Value),
                Is.EqualTo(expected.Pixels.ToArray().Select(pixel => pixel.Value)).Within(5e-6f));
            Assert.That(resident.IsDisposed, Is.False);
        });
    }

    [Test]
    public void Gray8_IsOneByteAndReplicatesIntoRgbChannels()
    {
        Gray8[] source = [new Gray8(0), new Gray8(127), new Gray8(255)];
        var destination = new Rgb[source.Length];

        PixelConverter.Convert<Gray8, Rgb>(source, destination);

        Assert.Multiple(() =>
        {
            Assert.That(destination[1].Red, Is.EqualTo(127f / 255f).Within(1e-6f));
            Assert.That(destination[1].Green, Is.EqualTo(destination[1].Red));
            Assert.That(destination[1].Blue, Is.EqualTo(destination[1].Red));
        });
    }

    [Test]
    public void ColorEncoding_RoundTripPreservesNormalizedValue()
    {
        const float srgb = 0.42f;

        float linear = PixelConverter.SrgbToLinear(srgb);
        float roundTrip = PixelConverter.LinearToSrgb(linear);

        Assert.That(roundTrip, Is.EqualTo(srgb).Within(1e-6f));
    }

    [Test]
    public void GenericImage_WrapDoesNotCopyPixelMemory()
    {
        Rgb24[] pixels = [new Rgb24(1, 2, 3)];

        Image<Rgb24> image = Image<Rgb24>.Wrap(pixels, 1, 1);
        pixels[0] = new Rgb24(4, 5, 6);

        Assert.Multiple(() =>
        {
            Assert.That(image.OwnsPixelMemory, Is.False);
            Assert.That(image.Pixels.Span[0].Red, Is.EqualTo(4));
        });
    }

    [Test]
    public void GenericImage_ProvidesRowsCopyAndCrop()
    {
        Gray8[] pixels =
        [
            new Gray8(1), new Gray8(2), new Gray8(3),
            new Gray8(4), new Gray8(5), new Gray8(6)
        ];
        Image<Gray8> image = Image<Gray8>.Load(pixels, 3, 2);
        var row = new Gray8[3];

        image.CopyRow(1, row);
        Image<Gray8> crop = image.Crop(1, 0, 2, 2);

        Assert.Multiple(() =>
        {
            Assert.That(
                row.Select(pixel => pixel.Value),
                Is.EqualTo(new byte[] { 4, 5, 6 }));
            Assert.That(
                crop.Pixels.ToArray().Select(pixel => pixel.Value),
                Is.EqualTo(new byte[] { 2, 3, 5, 6 }));
            Assert.That(crop.OwnsPixelMemory, Is.True);
        });
    }

    [Test]
    public void GrayImageOperations_BlurResidualAndDownsampleAreDeterministic()
    {
        GrayF32[] pixels =
        [
            new GrayF32(0f), new GrayF32(0f),
            new GrayF32(0f), new GrayF32(1f)
        ];
        Image<GrayF32> image = Image<GrayF32>.Load(
            pixels,
            2,
            2,
            ColorEncoding.Linear);

        Image<GrayF32> blur = image.BoxBlur();
        Image<GrayF32> residual = image.Subtract(blur);
        Image<GrayF32> downsampled = image.Downsample(0.5f);

        Assert.Multiple(() =>
        {
            Assert.That(
                blur.Pixels.ToArray().Select(pixel => pixel.Value),
                Is.All.EqualTo(0.25f).Within(1e-6f));
            Assert.That(residual.Pixels.Span[3].Value, Is.EqualTo(0.75f).Within(1e-6f));
            Assert.That(downsampled.Width, Is.EqualTo(1));
            Assert.That(downsampled.Height, Is.EqualTo(1));
            Assert.That(
                downsampled.Pixels.Span[0].Value,
                Is.EqualTo(0.25f).Within(1e-6f));
        });
    }

    [TestCase(1)]
    [TestCase(3)]
    public void GrayImageOperations_BoxBlurMatchesScalarReferenceWithVectorTail(int radius)
    {
        int width = (Vector<float>.Count * 2) + 3;
        const int height = 7;
        float[] source = Enumerable.Range(0, width * height)
            .Select(index => ((index * 17) % 101) / 100f)
            .ToArray();
        var actual = new float[source.Length];

        GrayImageOperations.BoxBlur(source, actual, width, height, radius);
        float[] expected = ScalarBoxBlur(source, width, height, radius);

        Assert.That(actual, Is.EqualTo(expected).Within(2e-6f));
    }

    [Test]
    public void GrayImageOperations_SubtractSupportsInPlaceRightOperandAndVectorTail()
    {
        int length = (Vector<float>.Count * 3) + 5;
        float[] left = Enumerable.Range(0, length).Select(i => i * 0.25f).ToArray();
        float[] rightAndDestination = Enumerable.Range(0, length).Select(i => i * 0.1f).ToArray();

        GrayImageOperations.Subtract(left, rightAndDestination, rightAndDestination);

        Assert.That(rightAndDestination, Is.EqualTo(Enumerable.Range(0, length).Select(i => i * 0.15f)).Within(1e-5f));
    }

    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    [TestCase(ComputeBackendKind.Simd)]
    [TestCase(ComputeBackendKind.Gpu)]
    public void ImageFilters_GradientsLaplacianAndEdgeMapHaveBackendParity(ComputeBackendKind backend)
    {
        const int width = 13;
        const int height = 9;
        float[] source = Enumerable.Range(0, width * height)
            .Select(index => (index * 23 % 71) / 70f)
            .ToArray();
        var scalar = new ComputeOptions { Backend = ComputeBackendKind.Scalar };
        var actualOptions = new ComputeOptions { Backend = backend };

        float[] expectedGradient = ImageFilters.GradientMagnitude(source, width, height, scalar);
        float[] expectedLaplacian = ImageFilters.Laplacian(source, width, height, scalar);
        float[] expectedEdges = ImageFilters.EdgeMap(source, width, height, 0.08f, scalar);
        float[] expectedContrast = ImageSpatialOperations.LocalContrast(source, width, height, options: scalar);

        Assert.Multiple(() =>
        {
            Assert.That(ImageFilters.GradientMagnitude(source, width, height, actualOptions), Is.EqualTo(expectedGradient).Within(4e-6f));
            Assert.That(ImageFilters.Laplacian(source, width, height, actualOptions), Is.EqualTo(expectedLaplacian).Within(4e-6f));
            Assert.That(ImageFilters.EdgeMap(source, width, height, 0.08f, actualOptions), Is.EqualTo(expectedEdges));
            Assert.That(ImageSpatialOperations.LocalContrast(source, width, height, options: actualOptions), Is.EqualTo(expectedContrast).Within(2e-6f));
        });
    }

    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    [TestCase(ComputeBackendKind.Simd)]
    [TestCase(ComputeBackendKind.Gpu)]
    public void ImageSpectrumOperations_HaveBackendParity(ComputeBackendKind backend)
    {
        const int width = 16;
        const int height = 8;
        float[] power = Enumerable.Range(0, width * height)
            .Select(index => ((index * 37) % 97) / 96f)
            .ToArray();
        var scalarOptions = new ComputeOptions { Backend = ComputeBackendKind.Scalar };
        var options = new ComputeOptions { Backend = backend };
        float[] expected = ImageSpectrumOperations.CalculateRadialSpectrum(power, width, height, 12, out FrequencyBandEnergy expectedBands, options: scalarOptions);

        float[] actual = ImageSpectrumOperations.CalculateRadialSpectrum(power, width, height, 12, out FrequencyBandEnergy actualBands, options: options);

        Assert.Multiple(() =>
        {
            Assert.That(actual, Is.EqualTo(expected).Within(2e-5f));
            Assert.That(actualBands.Low, Is.EqualTo(expectedBands.Low).Within(2e-4));
            Assert.That(actualBands.Mid, Is.EqualTo(expectedBands.Mid).Within(2e-4));
            Assert.That(actualBands.High, Is.EqualTo(expectedBands.High).Within(2e-4));
        });
    }

    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    [TestCase(ComputeBackendKind.Simd)]
    [TestCase(ComputeBackendKind.Gpu)]
    public void BayerDemosaicAndNoise_HaveBackendParity(ComputeBackendKind backend)
    {
        const int width = 12;
        const int height = 8;
        Rgb[] pixels = Enumerable.Range(0, width * height)
            .Select(index => new Rgb((index * 13 % 89) / 88f, (index * 29 % 97) / 96f, (index * 43 % 101) / 100f))
            .ToArray();
        Image<Rgb> image = Image<Rgb>.Load(pixels, width, height, ColorEncoding.Linear);
        var scalar = new ComputeOptions { Backend = ComputeBackendKind.Scalar };
        var options = new ComputeOptions { Backend = backend };
        float[] expectedMosaic = image.ToBayer(BayerPattern.Grbg, scalar);
        Image<Rgb> expectedDemosaic = BayerOperations.DemosaicBilinear(expectedMosaic, width, height, BayerPattern.Grbg, options: scalar);
        float[] expectedNoise = ImageNoiseOperations.ApplySignalDependentNoise(expectedMosaic, 0.002f, 0.0001f, 42, scalar);

        float[] actualMosaic = image.ToBayer(BayerPattern.Grbg, options);
        Image<Rgb> actualDemosaic = BayerOperations.DemosaicBilinear(actualMosaic, width, height, BayerPattern.Grbg, options: options);
        float[] actualNoise = ImageNoiseOperations.ApplySignalDependentNoise(actualMosaic, 0.002f, 0.0001f, 42, options);

        Assert.Multiple(() =>
        {
            Assert.That(actualMosaic, Is.EqualTo(expectedMosaic).Within(3e-6f));
            Assert.That(MemoryMarshal.Cast<Rgb, float>(actualDemosaic.Pixels.Span).ToArray(), Is.EqualTo(MemoryMarshal.Cast<Rgb, float>(expectedDemosaic.Pixels.Span).ToArray()).Within(5e-6f));
            Assert.That(actualNoise, Is.EqualTo(expectedNoise).Within(5e-6f));
        });
    }

    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    [TestCase(ComputeBackendKind.Gpu)]
    public void LocalEntropy_HasSupportedBackendParity(ComputeBackendKind backend)
    {
        const int width = 9;
        const int height = 7;
        float[] source = Enumerable.Range(0, width * height).Select(index => (index * 11 % 17) / 16f).ToArray();
        float[] expected = ImageSpatialOperations.LocalEntropy(source, width, height, 1, 8, new ComputeOptions { Backend = ComputeBackendKind.Scalar });

        float[] actual = ImageSpatialOperations.LocalEntropy(source, width, height, 1, 8, new ComputeOptions { Backend = backend });

        Assert.That(actual, Is.EqualTo(expected).Within(2e-5f));
    }

    [Test]
    public void LocalEntropy_RejectsExplicitSimdInsteadOfUsingScalarFallback()
    {
        Assert.Throws<ComputeBackendNotSupportedException>(() =>
            ImageSpatialOperations.LocalEntropy(new float[9], 3, 3, options: new ComputeOptions { Backend = ComputeBackendKind.Simd }));
    }

    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    [TestCase(ComputeBackendKind.Simd)]
    [TestCase(ComputeBackendKind.Gpu)]
    public void ResizeBilinear_HasBackendParityForUpsampling(ComputeBackendKind backend)
    {
        float[] source =
        [
            0f, 1f,
            1f, 0f
        ];
        float[] expected = ImageResampler.Resize(source, 2, 2, 7, 5, options: new ComputeOptions { Backend = ComputeBackendKind.Scalar });

        float[] actual = ImageResampler.Resize(source, 2, 2, 7, 5, options: new ComputeOptions { Backend = backend });

        Assert.That(actual, Is.EqualTo(expected).Within(2e-6f));
    }

    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    [TestCase(ComputeBackendKind.Simd)]
    [TestCase(ComputeBackendKind.Gpu)]
    public void BlurSubtractAndDownsample_RespectBackendAndHaveParity(ComputeBackendKind backend)
    {
        const int width = 14;
        const int height = 10;
        float[] source = Enumerable.Range(0, width * height).Select(index => (index * 31 % 109) / 108f).ToArray();
        var scalar = new ComputeOptions { Backend = ComputeBackendKind.Scalar };
        var options = new ComputeOptions { Backend = backend };
        var expectedBlur = new float[source.Length];
        GrayImageOperations.BoxBlur(source, expectedBlur, width, height, 2, options: scalar);
        var actualBlur = new float[source.Length];
        GrayImageOperations.BoxBlur(source, actualBlur, width, height, 2, options: options);
        var expectedResidual = new float[source.Length];
        GrayImageOperations.Subtract(source, expectedBlur, expectedResidual, options: scalar);
        var actualResidual = new float[source.Length];
        GrayImageOperations.Subtract(source, actualBlur, actualResidual, options: options);
        var expectedDownsample = new float[35];
        ImageResampler.Downsample(source, expectedDownsample, width, height, 7, 5, options: scalar);
        var actualDownsample = new float[35];
        ImageResampler.Downsample(source, actualDownsample, width, height, 7, 5, options: options);

        Assert.Multiple(() =>
        {
            Assert.That(actualBlur, Is.EqualTo(expectedBlur).Within(4e-6f));
            Assert.That(actualResidual, Is.EqualTo(expectedResidual).Within(4e-6f));
            Assert.That(actualDownsample, Is.EqualTo(expectedDownsample).Within(4e-6f));
        });
    }

    [TestCase(22, 10, 11, 5)]
    [TestCase(44, 20, 11, 5)]
    [TestCase(23, 11, 7, 4)]
    public void ImageResampler_MatchesAreaReferenceIncludingSimdTail(
        int sourceWidth,
        int sourceHeight,
        int destinationWidth,
        int destinationHeight)
    {
        float[] source = Enumerable.Range(0, sourceWidth * sourceHeight)
            .Select(index => ((index * 29) % 113) / 112f)
            .ToArray();
        var actual = new float[destinationWidth * destinationHeight];

        ImageResampler.Downsample(
            source,
            actual,
            sourceWidth,
            sourceHeight,
            destinationWidth,
            destinationHeight);
        float[] expected = ScalarDownsample(
            source,
            sourceWidth,
            sourceHeight,
            destinationWidth,
            destinationHeight);

        Assert.That(actual, Is.EqualTo(expected).Within(2e-6f));
    }
    [Test]
    public void Load_UsesImageSharpRgb24ByteLayout()
    {
        using var encoded = new MemoryStream();
        using (var source = new SixLabors.ImageSharp.Image<ImageSharpRgb24>(2, 1))
        {
            source[0, 0] = new ImageSharpRgb24(255, 128, 0);
            source[1, 0] = new ImageSharpRgb24(12, 34, 56);
            source.Save(encoded, new PngEncoder());
        }

        encoded.Position = 0;
        using var decoded = ImageSharpImage.Load<ImageSharpRgb24>(encoded);
        var bytes = new byte[decoded.Width * decoded.Height * 3];
        decoded.CopyPixelDataTo(bytes);

        Image image = Image.Load(bytes, decoded.Width, decoded.Height);

        Assert.Multiple(() =>
        {
            Assert.That(image.Width, Is.EqualTo(2));
            Assert.That(image.Height, Is.EqualTo(1));
            Assert.That(image.Length, Is.EqualTo(2));
            Assert.That(image.Pixels[0].Red, Is.EqualTo(1f));
            Assert.That(image.Pixels[0].Green, Is.EqualTo(128f / 255f).Within(1e-6f));
            Assert.That(image.Pixels[0].Blue, Is.Zero);
            Assert.That(image.Pixels[1].Red, Is.EqualTo(12f / 255f).Within(1e-6f));
            Assert.That(image.Pixels[1].Green, Is.EqualTo(34f / 255f).Within(1e-6f));
            Assert.That(image.Pixels[1].Blue, Is.EqualTo(56f / 255f).Within(1e-6f));
        });
    }

    [Test]
    public void Constructor_DoesNotAllocatePixelBuffer()
    {
        var image = new Image(4, 3);

        Assert.Multiple(() =>
        {
            Assert.That(image.Width, Is.EqualTo(4));
            Assert.That(image.Height, Is.EqualTo(3));
            Assert.That(image.Pixels, Is.Empty);
            Assert.That(image.Length, Is.Zero);
        });
    }

    [Test]
    public void Load_RejectsByteCountThatDoesNotMatchDimensions()
    {
        var exception = Assert.Throws<ArgumentException>(
            () => Image.Load([255, 0, 0], 2, 1));

        Assert.That(exception!.ParamName, Is.EqualTo("pixelBytes"));
    }

    [Test]
    public void Grayscale_ComputesNormalizedLuminance()
    {
        Image image = Image.Load(
            [
                255, 0, 0,
                0, 255, 0,
                0, 0, 255,
                255, 255, 255
            ],
            2,
            2);

        float[] grayscale = image.Grayscale();

        Assert.That(
            grayscale,
            Is.EqualTo(new[] { 0.2126f, 0.7152f, 0.0722f, 1f })
                .Within(1e-6f));
    }

    [Test]
    public void Grayscale8_ReturnsOneBytePerPixel()
    {
        Image image = Image.Load(
            [255, 0, 0, 0, 255, 0, 0, 0, 255],
            3,
            1);

        Image<Gray8> grayscale = image.Grayscale8();

        Assert.Multiple(() =>
        {
            Assert.That(grayscale.Pixels.Length, Is.EqualTo(3));
            Assert.That(grayscale.Pixels.Span[0].Value, Is.EqualTo(54));
            Assert.That(grayscale.Pixels.Span[1].Value, Is.EqualTo(182));
            Assert.That(grayscale.Pixels.Span[2].Value, Is.EqualTo(18));
        });
    }

    [Test]
    public void Grayscale_MatchesImageSharpBt709WithinByteQuantization()
    {
        byte[] bytes =
        [
            12, 34, 56,
            210, 120, 30,
            17, 231, 99
        ];
        Image image = Image.Load(bytes, 3, 1);
        using var imageSharp =
            SixLabors.ImageSharp.Image.LoadPixelData<ImageSharpRgb24>(bytes, 3, 1);
        imageSharp.Mutate(context => context.Grayscale());

        float[] actual = image.Grayscale();

        for (int index = 0; index < actual.Length; index++)
        {
            float expected = imageSharp[index, 0].R / 255f;
            Assert.That(actual[index], Is.EqualTo(expected).Within(1f / 255f));
        }
    }

    [Test]
    public void Grayscale_RejectsImageWithoutLoadedPixels()
    {
        var image = new Image(1, 1);

        Assert.Throws<InvalidOperationException>(() => image.Grayscale());
    }

    private static float[] ScalarBoxBlur(float[] source, int width, int height, int radius)
    {
        var result = new float[source.Length];
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            double sum = 0;
            int count = 0;
            for (int yy = Math.Max(0, y - radius); yy <= Math.Min(height - 1, y + radius); yy++)
            for (int xx = Math.Max(0, x - radius); xx <= Math.Min(width - 1, x + radius); xx++)
            {
                sum += source[(yy * width) + xx];
                count++;
            }
            result[(y * width) + x] = (float)(sum / count);
        }
        return result;
    }

    private static float[] ScalarDownsample(
        float[] source,
        int sourceWidth,
        int sourceHeight,
        int destinationWidth,
        int destinationHeight)
    {
        var result = new float[destinationWidth * destinationHeight];
        for (int y = 0; y < destinationHeight; y++)
        {
            int y0 = y * sourceHeight / destinationHeight;
            int y1 = Math.Max(y0 + 1, (y + 1) * sourceHeight / destinationHeight);
            for (int x = 0; x < destinationWidth; x++)
            {
                int x0 = x * sourceWidth / destinationWidth;
                int x1 = Math.Max(x0 + 1, (x + 1) * sourceWidth / destinationWidth);
                double sum = 0;
                int count = 0;
                for (int yy = y0; yy < y1; yy++)
                for (int xx = x0; xx < x1; xx++)
                {
                    sum += source[(yy * sourceWidth) + xx];
                    count++;
                }
                result[(y * destinationWidth) + x] = (float)(sum / count);
            }
        }
        return result;
    }
}
