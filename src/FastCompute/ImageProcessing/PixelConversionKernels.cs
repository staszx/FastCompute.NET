using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace FastCompute.ImageProcessing;

internal static class PixelConversionKernels
{
    private static readonly Vector256<float> RedWeight = Vector256.Create(0.2126f);
    private static readonly Vector256<float> GreenWeight = Vector256.Create(0.7152f);
    private static readonly Vector256<float> BlueWeight = Vector256.Create(0.0722f);
    private static readonly Vector256<float> ByteScale = Vector256.Create(1f / 255f);
    private static readonly Vector256<float> ByteMaximum = Vector256.Create(255f);
    private static readonly Vector256<float> Zero = Vector256<float>.Zero;

    internal static void RgbToGrayF32(ReadOnlySpan<Rgb> source, Span<GrayF32> destination)
    {
        Span<float> output = MemoryMarshal.Cast<GrayF32, float>(destination);
        int index = 0;
        if (Avx2.IsSupported)
        {
            ReadOnlySpan<float> input = MemoryMarshal.Cast<Rgb, float>(source);
            ref float inputReference = ref MemoryMarshal.GetReference(input);
            for (; index <= source.Length - 8; index += 8)
            {
                DeinterleaveRgb(ref inputReference, index, out Vector256<float> red, out Vector256<float> green, out Vector256<float> blue);
                Luminance(red, green, blue).CopyTo(output.Slice(index, 8));
            }
        }
        for (; index < source.Length; index++)
        {
            Rgb pixel = source[index];
            output[index] = Luminance(pixel.Red, pixel.Green, pixel.Blue);
        }
    }

    internal static void RgbToGray8(ReadOnlySpan<Rgb> source, Span<Gray8> destination)
    {
        Span<byte> output = MemoryMarshal.Cast<Gray8, byte>(destination);
        int index = 0;
        if (Avx2.IsSupported)
        {
            ReadOnlySpan<float> input = MemoryMarshal.Cast<Rgb, float>(source);
            ref float inputReference = ref MemoryMarshal.GetReference(input);
            ref byte outputReference = ref MemoryMarshal.GetReference(output);
            for (; index <= source.Length - 8; index += 8)
            {
                DeinterleaveRgb(ref inputReference, index, out Vector256<float> red, out Vector256<float> green, out Vector256<float> blue);
                StoreEightBytes(Quantize(Luminance(red, green, blue)), ref outputReference, index);
            }
        }
        for (; index < source.Length; index++)
        {
            Rgb pixel = source[index];
            output[index] = ToByte(Luminance(pixel.Red, pixel.Green, pixel.Blue));
        }
    }

    internal static void Rgb24ToGrayF32(ReadOnlySpan<Rgb24> source, Span<GrayF32> destination)
    {
        Span<float> output = MemoryMarshal.Cast<GrayF32, float>(destination);
        int index = 0;
        if (Avx2.IsSupported)
        {
            for (; index <= source.Length - 8; index += 8)
            {
                GatherRgb24(source, index, out Vector256<float> red, out Vector256<float> green, out Vector256<float> blue);
                Luminance(red, green, blue).CopyTo(output.Slice(index, 8));
            }
        }
        for (; index < source.Length; index++)
        {
            Rgb24 pixel = source[index];
            output[index] = Luminance(pixel.Red / 255f, pixel.Green / 255f, pixel.Blue / 255f);
        }
    }

    internal static void Rgb24ToGray8(ReadOnlySpan<Rgb24> source, Span<Gray8> destination)
    {
        Span<byte> output = MemoryMarshal.Cast<Gray8, byte>(destination);
        int index = 0;
        if (Avx2.IsSupported)
        {
            ref byte outputReference = ref MemoryMarshal.GetReference(output);
            for (; index <= source.Length - 8; index += 8)
            {
                GatherRgb24(source, index, out Vector256<float> red, out Vector256<float> green, out Vector256<float> blue);
                StoreEightBytes(Quantize(Luminance(red, green, blue)), ref outputReference, index);
            }
        }
        for (; index < source.Length; index++)
        {
            Rgb24 pixel = source[index];
            output[index] = ToByte(Luminance(pixel.Red / 255f, pixel.Green / 255f, pixel.Blue / 255f));
        }
    }

    internal static void GrayF32ToGray8(ReadOnlySpan<float> source, Span<Gray8> destination)
    {
        Span<byte> output = MemoryMarshal.Cast<Gray8, byte>(destination);
        int index = 0;
        if (Avx2.IsSupported)
        {
            ref float sourceReference = ref MemoryMarshal.GetReference(source);
            ref byte outputReference = ref MemoryMarshal.GetReference(output);
            for (; index <= source.Length - 8; index += 8)
            {
                StoreEightBytes(
                    Quantize(Vector256.LoadUnsafe(ref sourceReference, (nuint)index)),
                    ref outputReference,
                    index);
            }
        }
        for (; index < source.Length; index++) output[index] = ToByte(source[index]);
    }

    internal static void Gray8ToGrayF32(ReadOnlySpan<byte> source, Span<GrayF32> destination)
    {
        Span<float> output = MemoryMarshal.Cast<GrayF32, float>(destination);
        int index = 0;
        if (Avx2.IsSupported)
        {
            ref byte sourceReference = ref MemoryMarshal.GetReference(source);
            for (; index <= source.Length - 16; index += 16)
            {
                Vector128<byte> bytes = Vector128.LoadUnsafe(ref sourceReference, (nuint)index);
                Vector256<float> first = Avx.Multiply(
                    Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(bytes)),
                    ByteScale);
                Vector128<byte> upper = Sse2.ShiftRightLogical128BitLane(bytes, 8);
                Vector256<float> second = Avx.Multiply(
                    Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(upper)),
                    ByteScale);
                first.CopyTo(output.Slice(index, 8));
                second.CopyTo(output.Slice(index + 8, 8));
            }
        }
        for (; index < source.Length; index++) output[index] = source[index] / 255f;
    }

    internal static void Gray8ToRgb(ReadOnlySpan<byte> source, Span<Rgb> destination)
    {
        Span<float> output = MemoryMarshal.Cast<Rgb, float>(destination);
        int index = 0;
        if (Avx2.IsSupported)
        {
            ref byte sourceReference = ref MemoryMarshal.GetReference(source);
            ref float outputReference = ref MemoryMarshal.GetReference(output);
            for (; index <= source.Length - 8; index += 8)
            {
                Vector128<byte> bytes = Vector128.LoadUnsafe(ref sourceReference, (nuint)index);
                Vector256<float> gray = Avx.Multiply(
                    Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(bytes)),
                    ByteScale);
                InterleaveRgb(gray, gray, gray, out Vector256<float> first, out Vector256<float> second, out Vector256<float> third);
                nuint offset = (nuint)(index * 3);
                first.StoreUnsafe(ref outputReference, offset);
                second.StoreUnsafe(ref outputReference, offset + 8);
                third.StoreUnsafe(ref outputReference, offset + 16);
            }
        }
        for (; index < source.Length; index++)
        {
            float value = source[index] / 255f;
            destination[index] = new Rgb(value, value, value);
        }
    }

    internal static void Gray8ToRgb24(ReadOnlySpan<byte> source, Span<Rgb24> destination)
    {
        Span<byte> output = MemoryMarshal.Cast<Rgb24, byte>(destination);
        int index = 0;
        if (Ssse3.IsSupported)
        {
            Vector128<byte> firstMask = Vector128.Create(
                (byte)0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5);
            Vector128<byte> secondMask = Vector128.Create(
                (byte)5, 5, 6, 6, 6, 7, 7, 7,
                0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
            ref byte sourceReference = ref MemoryMarshal.GetReference(source);
            ref byte outputReference = ref MemoryMarshal.GetReference(output);
            for (; index <= source.Length - 8; index += 8)
            {
                Vector128<byte> bytes = Vector128.LoadUnsafe(ref sourceReference, (nuint)index);
                nuint outputOffset = (nuint)(index * 3);
                Ssse3.Shuffle(bytes, firstMask).StoreUnsafe(ref outputReference, outputOffset);
                ulong finalBytes = Ssse3.Shuffle(bytes, secondMask).AsUInt64().GetElement(0);
                Unsafe.WriteUnaligned(ref output[(index * 3) + 16], finalBytes);
            }
        }
        for (; index < source.Length; index++)
        {
            byte value = source[index];
            destination[index] = new Rgb24(value, value, value);
        }
    }

    internal static void Rgb24ToRgb(ReadOnlySpan<Rgb24> source, Span<Rgb> destination)
    {
        Span<float> output = MemoryMarshal.Cast<Rgb, float>(destination);
        int index = 0;
        if (Avx2.IsSupported)
        {
            ref float outputReference = ref MemoryMarshal.GetReference(output);
            for (; index <= source.Length - 8; index += 8)
            {
                GatherRgb24(source, index, out Vector256<float> red, out Vector256<float> green, out Vector256<float> blue);
                InterleaveRgb(red, green, blue, out Vector256<float> first, out Vector256<float> second, out Vector256<float> third);
                nuint offset = (nuint)(index * 3);
                first.StoreUnsafe(ref outputReference, offset);
                second.StoreUnsafe(ref outputReference, offset + 8);
                third.StoreUnsafe(ref outputReference, offset + 16);
            }
        }
        for (; index < source.Length; index++)
        {
            Rgb24 pixel = source[index];
            destination[index] = new Rgb(pixel.Red / 255f, pixel.Green / 255f, pixel.Blue / 255f);
        }
    }

    internal static void RgbToRgb24(ReadOnlySpan<Rgb> source, Span<Rgb24> destination)
    {
        int index = 0;
        if (Avx2.IsSupported)
        {
            ReadOnlySpan<float> input = MemoryMarshal.Cast<Rgb, float>(source);
            ref float inputReference = ref MemoryMarshal.GetReference(input);
            for (; index <= source.Length - 8; index += 8)
            {
                DeinterleaveRgb(ref inputReference, index, out Vector256<float> red, out Vector256<float> green, out Vector256<float> blue);
                StoreRgb24(Quantize(red), Quantize(green), Quantize(blue), destination, index);
            }
        }
        for (; index < source.Length; index++)
        {
            Rgb pixel = source[index];
            destination[index] = new Rgb24(ToByte(pixel.Red), ToByte(pixel.Green), ToByte(pixel.Blue));
        }
    }

    internal static void GrayF32ToRgb(ReadOnlySpan<float> source, Span<Rgb> destination)
    {
        Span<float> output = MemoryMarshal.Cast<Rgb, float>(destination);
        int index = 0;
        if (Avx2.IsSupported)
        {
            ref float sourceReference = ref MemoryMarshal.GetReference(source);
            ref float outputReference = ref MemoryMarshal.GetReference(output);
            for (; index <= source.Length - 8; index += 8)
            {
                Vector256<float> gray = Vector256.LoadUnsafe(ref sourceReference, (nuint)index);
                InterleaveRgb(gray, gray, gray, out Vector256<float> first, out Vector256<float> second, out Vector256<float> third);
                nuint offset = (nuint)(index * 3);
                first.StoreUnsafe(ref outputReference, offset);
                second.StoreUnsafe(ref outputReference, offset + 8);
                third.StoreUnsafe(ref outputReference, offset + 16);
            }
        }
        for (; index < source.Length; index++) destination[index] = new Rgb(source[index], source[index], source[index]);
    }

    internal static void GrayF32ToRgb24(ReadOnlySpan<float> source, Span<Rgb24> destination)
    {
        int index = 0;
        if (Avx2.IsSupported)
        {
            ref float sourceReference = ref MemoryMarshal.GetReference(source);
            for (; index <= source.Length - 8; index += 8)
            {
                Vector128<byte> bytes = Quantize(Vector256.LoadUnsafe(ref sourceReference, (nuint)index));
                StoreRgb24(bytes, bytes, bytes, destination, index);
            }
        }
        for (; index < source.Length; index++)
        {
            byte value = ToByte(source[index]);
            destination[index] = new Rgb24(value, value, value);
        }
    }

    internal static void DeinterleaveFloat3(
        ReadOnlySpan<float> input,
        int pixelIndex,
        Span<float> firstComponent,
        Span<float> secondComponent,
        Span<float> thirdComponent)
    {
        ref float inputReference = ref MemoryMarshal.GetReference(input);
        DeinterleaveRgb(ref inputReference, pixelIndex, out Vector256<float> first, out Vector256<float> second, out Vector256<float> third);
        first.CopyTo(firstComponent);
        second.CopyTo(secondComponent);
        third.CopyTo(thirdComponent);
    }

    internal static void DeinterleaveFloat2(
        ReadOnlySpan<float> input,
        int pixelIndex,
        Span<float> firstComponent,
        Span<float> secondComponent)
    {
        ref float inputReference = ref MemoryMarshal.GetReference(input);
        nuint offset = (nuint)(pixelIndex * 2);
        Vector256<float> firstBlock = Vector256.LoadUnsafe(ref inputReference, offset);
        Vector256<float> secondBlock = Vector256.LoadUnsafe(ref inputReference, offset + 8);
        Vector256<int> even = Vector256.Create(0, 2, 4, 6, 0, 2, 4, 6);
        Vector256<int> odd = Vector256.Create(1, 3, 5, 7, 1, 3, 5, 7);
        Avx.Blend(
            Avx2.PermuteVar8x32(firstBlock, even),
            Avx2.PermuteVar8x32(secondBlock, even),
            0xF0).CopyTo(firstComponent);
        Avx.Blend(
            Avx2.PermuteVar8x32(firstBlock, odd),
            Avx2.PermuteVar8x32(secondBlock, odd),
            0xF0).CopyTo(secondComponent);
    }

    internal static void InterleaveFloat2(
        ReadOnlySpan<float> firstComponent,
        ReadOnlySpan<float> secondComponent,
        Span<float> output,
        int pixelIndex)
    {
        Vector256<float> first = Vector256.Create(firstComponent);
        Vector256<float> second = Vector256.Create(secondComponent);
        Vector256<float> low = Avx.UnpackLow(first, second);
        Vector256<float> high = Avx.UnpackHigh(first, second);
        ref float outputReference = ref MemoryMarshal.GetReference(output);
        nuint offset = (nuint)(pixelIndex * 2);
        Avx.Permute2x128(low, high, 0x20).StoreUnsafe(ref outputReference, offset);
        Avx.Permute2x128(low, high, 0x31).StoreUnsafe(ref outputReference, offset + 8);
    }

    internal static void InterleaveFloat3(
        ReadOnlySpan<float> firstComponent,
        ReadOnlySpan<float> secondComponent,
        ReadOnlySpan<float> thirdComponent,
        Span<float> output,
        int pixelIndex)
    {
        Vector256<float> first = Vector256.Create(firstComponent);
        Vector256<float> second = Vector256.Create(secondComponent);
        Vector256<float> third = Vector256.Create(thirdComponent);
        InterleaveRgb(first, second, third, out Vector256<float> firstBlock, out Vector256<float> secondBlock, out Vector256<float> thirdBlock);
        ref float outputReference = ref MemoryMarshal.GetReference(output);
        nuint offset = (nuint)(pixelIndex * 3);
        firstBlock.StoreUnsafe(ref outputReference, offset);
        secondBlock.StoreUnsafe(ref outputReference, offset + 8);
        thirdBlock.StoreUnsafe(ref outputReference, offset + 16);
    }

    internal static void DeinterleaveByte3(
        ReadOnlySpan<byte> input,
        int pixelIndex,
        Span<int> firstComponent,
        Span<int> secondComponent,
        Span<int> thirdComponent)
    {
        GatherByte3(input, pixelIndex, out Vector256<int> first, out Vector256<int> second, out Vector256<int> third);
        first.CopyTo(firstComponent);
        second.CopyTo(secondComponent);
        third.CopyTo(thirdComponent);
    }

    internal static void DeinterleaveByte2(
        ReadOnlySpan<byte> input,
        int pixelIndex,
        Span<int> firstComponent,
        Span<int> secondComponent)
    {
        ref byte source = ref MemoryMarshal.GetReference(input);
        Vector128<byte> packed = Vector128.LoadUnsafe(ref source, (nuint)(pixelIndex * 2));
        Vector128<byte> mask = Vector128.Create(
            (byte)0, 0x80, 0x80, 0x80, 2, 0x80, 0x80, 0x80,
            4, 0x80, 0x80, 0x80, 6, 0x80, 0x80, 0x80);
        Vector256.Create(
            Ssse3.Shuffle(packed, mask).AsInt32(),
            Ssse3.Shuffle(packed, Sse2.Add(mask, Vector128.Create((byte)8))).AsInt32())
            .CopyTo(firstComponent);
        mask = Sse2.Add(mask, Vector128.Create((byte)1));
        Vector256.Create(
            Ssse3.Shuffle(packed, mask).AsInt32(),
            Ssse3.Shuffle(packed, Sse2.Add(mask, Vector128.Create((byte)8))).AsInt32())
            .CopyTo(secondComponent);
    }

    internal static void DeinterleaveByte4(
        ReadOnlySpan<byte> input,
        int pixelIndex,
        Span<int> firstComponent,
        Span<int> secondComponent,
        Span<int> thirdComponent,
        Span<int> fourthComponent)
    {
        ref byte source = ref MemoryMarshal.GetReference(input);
        nuint offset = (nuint)(pixelIndex * 4);
        Vector128<byte> low = Vector128.LoadUnsafe(ref source, offset);
        Vector128<byte> high = Vector128.LoadUnsafe(ref source, offset + 16);
        Vector128<byte> mask = Vector128.Create(
            (byte)0, 0x80, 0x80, 0x80, 4, 0x80, 0x80, 0x80,
            8, 0x80, 0x80, 0x80, 12, 0x80, 0x80, 0x80);
        DeinterleaveByte4Component(low, high, mask, firstComponent);
        mask = Sse2.Add(mask, Vector128.Create((byte)1));
        DeinterleaveByte4Component(low, high, mask, secondComponent);
        mask = Sse2.Add(mask, Vector128.Create((byte)1));
        DeinterleaveByte4Component(low, high, mask, thirdComponent);
        mask = Sse2.Add(mask, Vector128.Create((byte)1));
        DeinterleaveByte4Component(low, high, mask, fourthComponent);
    }

    internal static void StoreByte1(ReadOnlySpan<int> values, Span<byte> output, int pixelIndex)
    {
        ref byte outputReference = ref MemoryMarshal.GetReference(output);
        StoreEightBytes(PackIntegers(Vector256.Create(values)), ref outputReference, pixelIndex);
    }

    internal static void InterleaveByte3(
        ReadOnlySpan<int> firstComponent,
        ReadOnlySpan<int> secondComponent,
        ReadOnlySpan<int> thirdComponent,
        Span<byte> output,
        int pixelIndex) =>
        StoreRgb24(
            PackIntegers(Vector256.Create(firstComponent)),
            PackIntegers(Vector256.Create(secondComponent)),
            PackIntegers(Vector256.Create(thirdComponent)),
            MemoryMarshal.Cast<byte, Rgb24>(output),
            pixelIndex);

    internal static void InterleaveByte2(
        ReadOnlySpan<int> firstComponent,
        ReadOnlySpan<int> secondComponent,
        Span<byte> output,
        int pixelIndex)
    {
        Vector128<byte> first = PackIntegers(Vector256.Create(firstComponent));
        Vector128<byte> second = PackIntegers(Vector256.Create(secondComponent));
        Vector128<byte> packed = Sse2.UnpackLow(first, second);
        ref byte destination = ref MemoryMarshal.GetReference(output);
        packed.StoreUnsafe(ref destination, (nuint)(pixelIndex * 2));
    }

    internal static void InterleaveByte4(
        ReadOnlySpan<int> firstComponent,
        ReadOnlySpan<int> secondComponent,
        ReadOnlySpan<int> thirdComponent,
        ReadOnlySpan<int> fourthComponent,
        Span<byte> output,
        int pixelIndex)
    {
        Vector128<byte> firstSecond = Sse2.UnpackLow(
            PackIntegers(Vector256.Create(firstComponent)),
            PackIntegers(Vector256.Create(secondComponent)));
        Vector128<byte> thirdFourth = Sse2.UnpackLow(
            PackIntegers(Vector256.Create(thirdComponent)),
            PackIntegers(Vector256.Create(fourthComponent)));
        Vector128<byte> low = Sse2.UnpackLow(firstSecond.AsUInt16(), thirdFourth.AsUInt16()).AsByte();
        Vector128<byte> high = Sse2.UnpackHigh(firstSecond.AsUInt16(), thirdFourth.AsUInt16()).AsByte();
        ref byte destination = ref MemoryMarshal.GetReference(output);
        nuint offset = (nuint)(pixelIndex * 4);
        low.StoreUnsafe(ref destination, offset);
        high.StoreUnsafe(ref destination, offset + 16);
    }

    private static void DeinterleaveByte4Component(
        Vector128<byte> low,
        Vector128<byte> high,
        Vector128<byte> mask,
        Span<int> destination) =>
        Vector256.Create(
            Ssse3.Shuffle(low, mask).AsInt32(),
            Ssse3.Shuffle(high, mask).AsInt32())
        .CopyTo(destination);

    private static void DeinterleaveRgb(
        ref float input,
        int pixelIndex,
        out Vector256<float> red,
        out Vector256<float> green,
        out Vector256<float> blue)
    {
        nuint offset = (nuint)(pixelIndex * 3);
        Vector256<float> first = Vector256.LoadUnsafe(ref input, offset);
        Vector256<float> second = Vector256.LoadUnsafe(ref input, offset + 8);
        Vector256<float> third = Vector256.LoadUnsafe(ref input, offset + 16);

        red = Avx.Blend(
            Avx.Blend(
                Avx2.PermuteVar8x32(first, Vector256.Create(0, 3, 6, 0, 0, 0, 0, 0)),
                Avx2.PermuteVar8x32(second, Vector256.Create(0, 0, 0, 1, 4, 7, 0, 0)),
                0x38),
            Avx2.PermuteVar8x32(third, Vector256.Create(0, 0, 0, 0, 0, 0, 2, 5)),
            0xC0);
        green = Avx.Blend(
            Avx.Blend(
                Avx2.PermuteVar8x32(first, Vector256.Create(1, 4, 7, 0, 0, 0, 0, 0)),
                Avx2.PermuteVar8x32(second, Vector256.Create(0, 0, 0, 2, 5, 0, 0, 0)),
                0x18),
            Avx2.PermuteVar8x32(third, Vector256.Create(0, 0, 0, 0, 0, 0, 3, 6)),
            0xE0);
        blue = Avx.Blend(
            Avx.Blend(
                Avx2.PermuteVar8x32(first, Vector256.Create(2, 5, 0, 0, 0, 0, 0, 0)),
                Avx2.PermuteVar8x32(second, Vector256.Create(0, 0, 0, 3, 6, 0, 0, 0)),
                0x1C),
            Avx2.PermuteVar8x32(third, Vector256.Create(0, 0, 0, 0, 0, 1, 4, 7)),
            0xE0);
    }

    private static void GatherRgb24(
        ReadOnlySpan<Rgb24> source,
        int index,
        out Vector256<float> red,
        out Vector256<float> green,
        out Vector256<float> blue)
    {
        GatherByte3(MemoryMarshal.AsBytes(source), index, out Vector256<int> redBytes, out Vector256<int> greenBytes, out Vector256<int> blueBytes);
        red = Avx.Multiply(Avx.ConvertToVector256Single(redBytes), ByteScale);
        green = Avx.Multiply(Avx.ConvertToVector256Single(greenBytes), ByteScale);
        blue = Avx.Multiply(Avx.ConvertToVector256Single(blueBytes), ByteScale);
    }

    private static void GatherByte3(
        ReadOnlySpan<byte> bytes,
        int index,
        out Vector256<int> redBytes,
        out Vector256<int> greenBytes,
        out Vector256<int> blueBytes)
    {
        ref byte input = ref MemoryMarshal.GetReference(bytes);
        nuint offset = (nuint)(index * 3);
        Vector128<byte> first = Vector128.LoadUnsafe(ref input, offset);
        Vector128<byte> second = Vector128.LoadUnsafe(ref input, offset + 8);
        Vector128<byte> lowMask = Vector128.Create(
            (byte)0, 0x80, 0x80, 0x80, 3, 0x80, 0x80, 0x80,
            6, 0x80, 0x80, 0x80, 9, 0x80, 0x80, 0x80);
        Vector128<byte> highMask = Vector128.Create(
            (byte)4, 0x80, 0x80, 0x80, 7, 0x80, 0x80, 0x80,
            10, 0x80, 0x80, 0x80, 13, 0x80, 0x80, 0x80);
        redBytes = Vector256.Create(
            Ssse3.Shuffle(first, lowMask).AsInt32(),
            Ssse3.Shuffle(second, highMask).AsInt32());
        lowMask = Sse2.Add(lowMask, Vector128.Create((byte)1));
        highMask = Sse2.Add(highMask, Vector128.Create((byte)1));
        greenBytes = Vector256.Create(
            Ssse3.Shuffle(first, lowMask).AsInt32(),
            Ssse3.Shuffle(second, highMask).AsInt32());
        lowMask = Sse2.Add(lowMask, Vector128.Create((byte)1));
        highMask = Sse2.Add(highMask, Vector128.Create((byte)1));
        blueBytes = Vector256.Create(
            Ssse3.Shuffle(first, lowMask).AsInt32(),
            Ssse3.Shuffle(second, highMask).AsInt32());
    }

    private static void InterleaveRgb(
        Vector256<float> red,
        Vector256<float> green,
        Vector256<float> blue,
        out Vector256<float> first,
        out Vector256<float> second,
        out Vector256<float> third)
    {
        first = Avx.Blend(
            Avx.Blend(
                Avx2.PermuteVar8x32(red, Vector256.Create(0, 0, 0, 1, 0, 0, 2, 0)),
                Avx2.PermuteVar8x32(green, Vector256.Create(0, 0, 0, 0, 1, 0, 0, 2)),
                0x92),
            Avx2.PermuteVar8x32(blue, Vector256.Create(0, 0, 0, 0, 0, 1, 0, 0)),
            0x24);
        second = Avx.Blend(
            Avx.Blend(
                Avx2.PermuteVar8x32(blue, Vector256.Create(2, 0, 0, 3, 0, 0, 4, 0)),
                Avx2.PermuteVar8x32(red, Vector256.Create(0, 3, 0, 0, 4, 0, 0, 5)),
                0x92),
            Avx2.PermuteVar8x32(green, Vector256.Create(0, 0, 3, 0, 0, 4, 0, 0)),
            0x24);
        third = Avx.Blend(
            Avx.Blend(
                Avx2.PermuteVar8x32(green, Vector256.Create(5, 0, 0, 6, 0, 0, 7, 0)),
                Avx2.PermuteVar8x32(blue, Vector256.Create(0, 5, 0, 0, 6, 0, 0, 7)),
                0x92),
            Avx2.PermuteVar8x32(red, Vector256.Create(0, 0, 6, 0, 0, 7, 0, 0)),
            0x24);
    }

    private static Vector256<float> Luminance(Vector256<float> red, Vector256<float> green, Vector256<float> blue) =>
        Avx.Add(Avx.Add(Avx.Multiply(red, RedWeight), Avx.Multiply(green, GreenWeight)), Avx.Multiply(blue, BlueWeight));

    private static float Luminance(float red, float green, float blue) =>
        (0.2126f * red) + (0.7152f * green) + (0.0722f * blue);

    private static Vector128<byte> Quantize(Vector256<float> values)
    {
        Vector256<float> clamped = Avx.Min(Avx.Max(values, Zero), Vector256.Create(1f));
        Vector256<int> integers = Avx.ConvertToVector256Int32(Avx.Multiply(clamped, ByteMaximum));
        Vector128<ushort> words = Sse41.PackUnsignedSaturate(integers.GetLower(), integers.GetUpper());
        return Sse2.PackUnsignedSaturate(words.AsInt16(), Vector128<short>.Zero);
    }

    private static Vector128<byte> PackIntegers(Vector256<int> values)
    {
        Vector128<ushort> words = Sse41.PackUnsignedSaturate(values.GetLower(), values.GetUpper());
        return Sse2.PackUnsignedSaturate(words.AsInt16(), Vector128<short>.Zero);
    }

    private static void StoreEightBytes(Vector128<byte> values, ref byte destination, int offset)
    {
        ulong bytes = values.AsUInt64().GetElement(0);
        Unsafe.WriteUnaligned(ref Unsafe.Add(ref destination, offset), bytes);
    }

    private static void StoreRgb24(
        Vector128<byte> red,
        Vector128<byte> green,
        Vector128<byte> blue,
        Span<Rgb24> destination,
        int pixelIndex)
    {
        Vector128<byte> redFirstMask = Vector128.Create(
            (byte)0, 0x80, 0x80, 1, 0x80, 0x80, 2, 0x80,
            0x80, 3, 0x80, 0x80, 4, 0x80, 0x80, 5);
        Vector128<byte> greenFirstMask = Vector128.Create(
            (byte)0x80, 0, 0x80, 0x80, 1, 0x80, 0x80, 2,
            0x80, 0x80, 3, 0x80, 0x80, 4, 0x80, 0x80);
        Vector128<byte> blueFirstMask = Vector128.Create(
            (byte)0x80, 0x80, 0, 0x80, 0x80, 1, 0x80, 0x80,
            2, 0x80, 0x80, 3, 0x80, 0x80, 4, 0x80);
        Vector128<byte> first = Sse2.Or(
            Sse2.Or(Ssse3.Shuffle(red, redFirstMask), Ssse3.Shuffle(green, greenFirstMask)),
            Ssse3.Shuffle(blue, blueFirstMask));

        Vector128<byte> redSecondMask = Vector128.Create(
            (byte)0x80, 0x80, 6, 0x80, 0x80, 7, 0x80, 0x80,
            0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
        Vector128<byte> greenSecondMask = Vector128.Create(
            (byte)5, 0x80, 0x80, 6, 0x80, 0x80, 7, 0x80,
            0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
        Vector128<byte> blueSecondMask = Vector128.Create(
            (byte)0x80, 5, 0x80, 0x80, 6, 0x80, 0x80, 7,
            0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
        Vector128<byte> second = Sse2.Or(
            Sse2.Or(Ssse3.Shuffle(red, redSecondMask), Ssse3.Shuffle(green, greenSecondMask)),
            Ssse3.Shuffle(blue, blueSecondMask));

        Span<byte> output = MemoryMarshal.Cast<Rgb24, byte>(destination);
        ref byte outputReference = ref MemoryMarshal.GetReference(output);
        int offset = pixelIndex * 3;
        first.StoreUnsafe(ref outputReference, (nuint)offset);
        Unsafe.WriteUnaligned(ref output[offset + 16], second.AsUInt64().GetElement(0));
    }

    private static byte ToByte(float value) =>
        (byte)Math.Clamp((int)MathF.Round(value * 255f), 0, 255);
}
