using System.Runtime.InteropServices;

namespace FastCompute;

/// <summary>Represents a GPU- and SIMD-friendly single-precision complex number.</summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct Complex32 : IEquatable<Complex32>, IComputeValue<Complex32>
{
    /// <summary>Gets the native two-component compute descriptor.</summary>
    public static ComputeValueDescriptor<Complex32> ComputeDescriptor { get; } =
        ComputeValueDescriptor<Complex32>.Create(value => value.Real, value => value.Imaginary);
    /// <summary>Initializes a complex number.</summary>
    public Complex32(float real, float imaginary)
    {
        Real = real;
        Imaginary = imaginary;
    }

    /// <summary>Gets the real component.</summary>
    public float Real { get; }

    /// <summary>Gets the imaginary component.</summary>
    public float Imaginary { get; }

    /// <summary>Gets the magnitude.</summary>
    public float Magnitude => MathF.Sqrt((Real * Real) + (Imaginary * Imaginary));

    /// <inheritdoc />
    public bool Equals(Complex32 other) => Real.Equals(other.Real) && Imaginary.Equals(other.Imaginary);

    /// <inheritdoc />
    public override bool Equals(object? obj) => obj is Complex32 other && Equals(other);

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Real, Imaginary);

    /// <summary>Adds two complex numbers.</summary>
    public static Complex32 operator +(Complex32 left, Complex32 right) =>
        new(left.Real + right.Real, left.Imaginary + right.Imaginary);

    /// <summary>Subtracts two complex numbers.</summary>
    public static Complex32 operator -(Complex32 left, Complex32 right) =>
        new(left.Real - right.Real, left.Imaginary - right.Imaginary);

    /// <summary>Multiplies two complex numbers.</summary>
    public static Complex32 operator *(Complex32 left, Complex32 right) =>
        new(
            (left.Real * right.Real) - (left.Imaginary * right.Imaginary),
            (left.Imaginary * right.Real) + (left.Real * right.Imaginary));

    /// <summary>Scales a complex number.</summary>
    public static Complex32 operator *(Complex32 value, float scale) =>
        new(value.Real * scale, value.Imaginary * scale);
}
