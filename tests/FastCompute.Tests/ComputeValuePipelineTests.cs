using System.Runtime.InteropServices;
using FastCompute.ImageProcessing;

namespace FastCompute.Tests;

public sealed class ComputeValuePipelineTests
{
    private static readonly ComputeBackendKind[] CpuBackends =
    [
        ComputeBackendKind.Scalar,
        ComputeBackendKind.ParallelCpu,
        ComputeBackendKind.Simd
    ];

    private static readonly ComputeBackendKind[] AllBackends =
    [
        ComputeBackendKind.Scalar,
        ComputeBackendKind.ParallelCpu,
        ComputeBackendKind.Simd,
        ComputeBackendKind.Gpu
    ];

    [TestCaseSource(nameof(CpuBackends))]
    public void RgbProjection_ProducesBackendParityAndHandlesTail(
        ComputeBackendKind backend)
    {
        Rgb[] pixels = CreatePixels(19);
        float[] expected = pixels
            .Select(pixel =>
                (0.2126f * pixel.Red) +
                (0.7152f * pixel.Green) +
                (0.0722f * pixel.Blue))
            .ToArray();

        float[] actual = pixels
            .AsCompute(new ComputeOptions { Backend = backend })
            .Select(Rgb.Luminance)
            .ToArray();

        Assert.That(actual, Is.EqualTo(expected).Within(1e-6f));
    }

    [TestCaseSource(nameof(CpuBackends))]
    public void RgbGrayscale_InPlaceReusesSourceAcrossBackends(
        ComputeBackendKind backend)
    {
        Rgb[] pixels = CreatePixels(19);

        Rgb[] result = pixels
            .AsCompute(new ComputeOptions { Backend = backend })
            .SelectInPlace(Rgb.Grayscale)
            .ToArrayInPlace();

        Assert.That(result, Is.SameAs(pixels));
        Assert.Multiple(() =>
        {
            for (int index = 0; index < result.Length; index++)
            {
                Assert.That(result[index].Green, Is.EqualTo(result[index].Red).Within(1e-6f));
                Assert.That(result[index].Blue, Is.EqualTo(result[index].Red).Within(1e-6f));
            }
        });
    }

    [TestCaseSource(nameof(CpuBackends))]
    public void CustomComputeValue_FusesMapAndProjection(
        ComputeBackendKind backend)
    {
        PairValue[] values = Enumerable.Range(0, 21)
            .Select(index => new PairValue(index, index * 0.5f))
            .ToArray();

        float[] result = values
            .AsCompute(new ComputeOptions { Backend = backend })
            .Select(value => new PairValue(
                value.First * 2f,
                value.Second + 1f))
            .Select(value => value.First + value.Second)
            .ToArray();

        float[] expected = values
            .Select(value => (value.First * 2f) + value.Second + 1f)
            .ToArray();
        Assert.That(result, Is.EqualTo(expected).Within(1e-6f));
    }

    [Test]
    public void ExplicitGpu_ExecutesCompositeProjectionAndMap()
    {
        Rgb[] pixels = CreatePixels(19);
        float[] expected = pixels.Select(Rgb.Luminance.Compile()).ToArray();

        float[] projection = pixels
            .AsCompute(new ComputeOptions { Backend = ComputeBackendKind.Gpu })
            .Select(Rgb.Luminance)
            .ToArray();
        Rgb[] grayscale = pixels
            .AsCompute(new ComputeOptions { Backend = ComputeBackendKind.Gpu })
            .Select(Rgb.Grayscale)
            .ToArray();

        Assert.Multiple(() =>
        {
            Assert.That(projection, Is.EqualTo(expected).Within(1e-6f));
            for (int index = 0; index < grayscale.Length; index++)
            {
                Assert.That(grayscale[index].Red, Is.EqualTo(expected[index]).Within(1e-6f));
                Assert.That(grayscale[index].Green, Is.EqualTo(expected[index]).Within(1e-6f));
                Assert.That(grayscale[index].Blue, Is.EqualTo(expected[index]).Within(1e-6f));
            }
        });
    }

    [TestCaseSource(nameof(CpuBackends))]
    public void EmptyCompositeArray_IsSupported(ComputeBackendKind backend)
    {
        Rgb[] source = [];

        float[] projection = source
            .AsCompute(new ComputeOptions { Backend = backend })
            .Select(Rgb.Luminance)
            .ToArray();
        Rgb[] inPlace = source
            .AsCompute(new ComputeOptions { Backend = backend })
            .SelectInPlace(Rgb.Grayscale)
            .ToArrayInPlace();

        Assert.Multiple(() =>
        {
            Assert.That(projection, Is.Empty);
            Assert.That(inPlace, Is.SameAs(source));
        });
    }

    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    public void UnsupportedExpression_IsRejectedBeforeBackendExecution(
        ComputeBackendKind backend)
    {
        Rgb[] pixels = CreatePixels(1);

        Assert.Throws<NotSupportedException>(
            () => pixels
                .AsCompute(new ComputeOptions { Backend = backend })
                .Select(pixel => MathF.Sqrt(pixel.Red))
                .ToArray());
    }

    [TestCase(ComputeBackendKind.Scalar)]
    [TestCase(ComputeBackendKind.ParallelCpu)]
    public void ByteComponents_ConvertToAnotherRegisteredType(
        ComputeBackendKind backend)
    {
        Rgb24[] source =
        [
            new Rgb24(30, 60, 90),
            new Rgb24(3, 6, 9)
        ];

        Gray8[] result = source
            .AsCompute(new ComputeOptions { Backend = backend })
            .Select(pixel => new Gray8(
                (byte)((pixel.Red + pixel.Green + pixel.Blue) / 3)))
            .ToArray();

        Assert.That(
            result.Select(pixel => pixel.Value),
            Is.EqualTo(new byte[] { 60, 6 }));
    }

    [Test]
    public void ByteComponents_SupportProjectionAndInPlaceMap()
    {
        Gray8[] source = [new Gray8(10), new Gray8(20), new Gray8(30)];

        float[] projected = source
            .AsCompute(new ComputeOptions { Backend = ComputeBackendKind.Scalar })
            .Select(pixel => (float)pixel.Value)
            .ToArray();
        Gray8[] mapped = source
            .AsCompute(new ComputeOptions { Backend = ComputeBackendKind.Scalar })
            .SelectInPlace(pixel => new Gray8((byte)(pixel.Value / 2)))
            .ToArrayInPlace();

        Assert.Multiple(() =>
        {
            Assert.That(projected, Is.EqualTo(new[] { 10f, 20f, 30f }));
            Assert.That(mapped, Is.SameAs(source));
            Assert.That(
                mapped.Select(pixel => pixel.Value),
                Is.EqualTo(new byte[] { 5, 10, 15 }));
        });
    }

    [TestCase(ComputeBackendKind.Simd)]
    [TestCase(ComputeBackendKind.Gpu)]
    public void ByteComponents_PreserveIntegerAndNarrowingSemanticsOnNativeBackends(ComputeBackendKind backend)
    {
        var options = new ComputeOptions { Backend = backend };
        Rgb24[] colors =
        [
            new Rgb24(30, 61, 92),
            new Rgb24(3, 7, 11)
        ];
        Gray8[] averages = colors
            .AsCompute(options)
            .Select(pixel => new Gray8(
                (byte)((pixel.Red + pixel.Green + pixel.Blue) / 3)))
            .ToArray();
        Gray8[] values = [new Gray8(250), new Gray8(21)];
        Gray8[] fusedSource = [new Gray8(250), new Gray8(21)];
        float[] projected = values.AsCompute(options)
            .Select(pixel => (float)pixel.Value)
            .ToArray();
        Gray8[] narrowed = values.AsCompute(options)
            .Select(pixel => new Gray8((byte)(pixel.Value + 10)))
            .ToArray();
        Gray8[] halved = values.AsCompute(options)
            .SelectInPlace(pixel => new Gray8((byte)(pixel.Value / 2)))
            .ToArrayInPlace();
        float[] fused = fusedSource.AsCompute(options)
            .Select(pixel => new Gray8((byte)(pixel.Value + 10)))
            .Select(pixel => (float)pixel.Value)
            .ToArray();
        PairByte[] custom =
        [
            new PairByte(3, 7),
            new PairByte(250, 130),
            new PairByte(11, 19),
            new PairByte(21, 31),
            new PairByte(41, 51),
            new PairByte(61, 71),
            new PairByte(81, 91),
            new PairByte(101, 111),
            new PairByte(121, 131)
        ];
        PairByte[] customResult = custom.AsCompute(options)
            .Select(value => new PairByte(
                (byte)(value.First + 1),
                (byte)(value.Second * 2)))
            .ToArray();
        QuadByte[] fourComponents = Enumerable.Range(0, 9)
            .Select(index => new QuadByte(
                (byte)index,
                (byte)(index + 10),
                (byte)(index + 20),
                (byte)(index + 30)))
            .ToArray();
        QuadByte[] fourComponentResult = fourComponents.AsCompute(options)
            .Select(value => new QuadByte(
                (byte)(value.First + 1),
                (byte)(value.Second + 2),
                (byte)(value.Third + 3),
                (byte)(value.Fourth + 4)))
            .ToArray();

        Assert.Multiple(() =>
        {
            Assert.That(averages.Select(pixel => pixel.Value), Is.EqualTo(new byte[] { 61, 7 }));
            Assert.That(projected, Is.EqualTo(new[] { 250f, 21f }));
            Assert.That(narrowed.Select(pixel => pixel.Value), Is.EqualTo(new byte[] { 4, 31 }));
            Assert.That(halved.Select(pixel => pixel.Value), Is.EqualTo(new byte[] { 125, 10 }));
            Assert.That(fused, Is.EqualTo(new[] { 4f, 31f }));
            Assert.That(customResult[0].First, Is.EqualTo(4));
            Assert.That(customResult[0].Second, Is.EqualTo(14));
            Assert.That(customResult[1].First, Is.EqualTo(251));
            Assert.That(customResult[1].Second, Is.EqualTo(4));
            Assert.That(fourComponentResult[0].First, Is.EqualTo(1));
            Assert.That(fourComponentResult[0].Second, Is.EqualTo(12));
            Assert.That(fourComponentResult[0].Third, Is.EqualTo(23));
            Assert.That(fourComponentResult[0].Fourth, Is.EqualTo(34));
            Assert.That(fourComponentResult[8].Fourth, Is.EqualTo(42));
        });
    }

    [Test]
    public void MixedComponentTypes_RejectUnsupportedExplicitBackend()
    {
        Rgb24[] source = [new Rgb24(1, 2, 3)];

        var exception = Assert.Throws<ComputeBackendNotSupportedException>(
            () => source
                .AsCompute(new ComputeOptions { Backend = ComputeBackendKind.Simd })
                .Select(pixel => new GrayF32(pixel.Red / 255f))
                .ToArray());

        Assert.That(exception!.Backend, Is.EqualTo(ComputeBackendKind.Simd));
    }

    [TestCaseSource(nameof(AllBackends))]
    public void FloatComponentTypes_TransformAcrossAllBackends(
        ComputeBackendKind backend)
    {
        Rgb[] source = CreatePixels(19);
        float[] expected = source.Select(Rgb.Luminance.Compile()).ToArray();

        GrayF32[] result = source
            .AsCompute(new ComputeOptions { Backend = backend })
            .Select(Rgb.GrayscaleF32)
            .ToArray();

        Assert.That(
            result.Select(pixel => pixel.Value),
            Is.EqualTo(expected).Within(1e-6f));
    }

    private static Rgb[] CreatePixels(int count) =>
        Enumerable.Range(0, count)
            .Select(index => new Rgb(
                (index % 7) / 6f,
                (index % 11) / 10f,
                (index % 13) / 12f))
            .ToArray();

    [StructLayout(LayoutKind.Sequential)]
    private readonly struct PairValue : IComputeValue<PairValue>
    {
        public PairValue(float first, float second)
        {
            First = first;
            Second = second;
        }

        public static ComputeValueDescriptor<PairValue> ComputeDescriptor { get; } =
            ComputeValueDescriptor<PairValue>.Create(
                value => value.First,
                value => value.Second);

        public float First { get; }

        public float Second { get; }
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    private readonly struct PairByte(byte first, byte second) : IComputeValue<PairByte>
    {
        public static ComputeValueDescriptor<PairByte> ComputeDescriptor { get; } =
            ComputeValueDescriptor<PairByte>.Create(value => value.First, value => value.Second);

        public byte First { get; } = first;

        public byte Second { get; } = second;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    private readonly struct QuadByte(byte first, byte second, byte third, byte fourth) : IComputeValue<QuadByte>
    {
        public static ComputeValueDescriptor<QuadByte> ComputeDescriptor { get; } =
            ComputeValueDescriptor<QuadByte>.Create(
                value => value.First,
                value => value.Second,
                value => value.Third,
                value => value.Fourth);

        public byte First { get; } = first;
        public byte Second { get; } = second;
        public byte Third { get; } = third;
        public byte Fourth { get; } = fourth;
    }
}
