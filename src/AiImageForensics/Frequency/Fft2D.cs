using System.Numerics;

namespace AiImageForensics.Frequency;

internal static class Fft2D
{
    public static void Transform(Complex[] data, int width, int height, CancellationToken cancellationToken)
    {
        if (!BitOperations.IsPow2((uint)width) || !BitOperations.IsPow2((uint)height) || data.Length != checked(width * height))
            throw new ArgumentException("FFT dimensions must be powers of two and match the buffer.");

        var column = new Complex[height];
        for (int y = 0; y < height; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Transform1D(data.AsSpan(y * width, width));
        }
        for (int x = 0; x < width; x++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (int y = 0; y < height; y++) column[y] = data[(y * width) + x];
            Transform1D(column);
            for (int y = 0; y < height; y++) data[(y * width) + x] = column[y];
        }
    }

    private static void Transform1D(Span<Complex> values)
    {
        int length = values.Length;
        for (int i = 1, j = 0; i < length; i++)
        {
            int bit = length >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) (values[i], values[j]) = (values[j], values[i]);
        }
        for (int size = 2; size <= length; size <<= 1)
        {
            double angle = -2 * Math.PI / size;
            Complex step = new(Math.Cos(angle), Math.Sin(angle));
            int half = size >> 1;
            for (int start = 0; start < length; start += size)
            {
                Complex factor = Complex.One;
                for (int j = 0; j < half; j++)
                {
                    Complex even = values[start + j];
                    Complex odd = values[start + j + half] * factor;
                    values[start + j] = even + odd;
                    values[start + j + half] = even - odd;
                    factor *= step;
                }
            }
        }
    }
}
