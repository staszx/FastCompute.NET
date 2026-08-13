using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace FastCompute.Gpu;

internal static class FourierGpuKernels
{
    public static void Phase(Index1D index, ArrayView<Complex32> source, ArrayView<float> destination) =>
        destination[index] = XMath.Atan2(source[index].Imaginary, source[index].Real);

    public static void BitReverse(Index1D index, ArrayView<Complex32> data, int dimensionLength, int transformCount, int baseStride, int elementStride)
    {
        int transform = index / dimensionLength;
        int position = index - (transform * dimensionLength);
        if (transform >= transformCount) return;
        int reversed = 0;
        int value = position;
        for (int bit = dimensionLength; bit > 1; bit >>= 1)
        {
            reversed = (reversed << 1) | (value & 1);
            value >>= 1;
        }
        if (position >= reversed) return;
        int baseOffset = transform * baseStride;
        int left = baseOffset + (position * elementStride);
        int right = baseOffset + (reversed * elementStride);
        Complex32 temporary = data[left];
        data[left] = data[right];
        data[right] = temporary;
    }

    public static void Stage(Index1D index, ArrayView<Complex32> data, int dimensionLength, int transformCount, int baseStride, int elementStride, int size, int inverse)
    {
        int butterfliesPerTransform = dimensionLength >> 1;
        int transform = index / butterfliesPerTransform;
        if (transform >= transformCount) return;
        int local = index - (transform * butterfliesPerTransform);
        int half = size >> 1;
        int block = local / half;
        int j = local - (block * half);
        int firstPosition = (block * size) + j;
        int baseOffset = transform * baseStride;
        int firstIndex = baseOffset + (firstPosition * elementStride);
        int secondIndex = firstIndex + (half * elementStride);
        float angle = (inverse == 0 ? -2f : 2f) * XMath.PI * j / size;
        float cosine = XMath.Cos(angle);
        float sine = XMath.Sin(angle);
        Complex32 even = data[firstIndex];
        Complex32 sourceOdd = data[secondIndex];
        var odd = new Complex32(
            (sourceOdd.Real * cosine) - (sourceOdd.Imaginary * sine),
            (sourceOdd.Imaginary * cosine) + (sourceOdd.Real * sine));
        data[firstIndex] = new Complex32(even.Real + odd.Real, even.Imaginary + odd.Imaginary);
        data[secondIndex] = new Complex32(even.Real - odd.Real, even.Imaginary - odd.Imaginary);
    }

    public static void Scale(Index1D index, ArrayView<Complex32> data, float scale)
    {
        Complex32 value = data[index];
        data[index] = new Complex32(value.Real * scale, value.Imaginary * scale);
    }
}
