namespace FastCompute.Tests;

public sealed class ComputeTests
{
    [Test]
    public void Run_AppliesArithmeticExpressionWithoutChangingSource()
    {
        float[] source = [1.0f, 2.0f, 3.0f];

        float[] result = Compute.Run(source, value => value * 2.0f + 1.0f);

        Assert.Multiple(() =>
        {
            Assert.That(result, Is.EqualTo(new[] { 3.0f, 5.0f, 7.0f }));
            Assert.That(source, Is.EqualTo(new[] { 1.0f, 2.0f, 3.0f }));
            Assert.That(result, Is.Not.SameAs(source));
        });
    }

    [Test]
    public void Run_AppliesGpuMathExpression()
    {
        float[] source = [-0.5f, 0.0f, 0.5f];

        float[] result = Compute.Run(
            source,
            value => GpuMath.Clamp(
                GpuMath.Sin(value) * GpuMath.Exp(-value * value),
                -1.0f,
                1.0f));

        for (int index = 0; index < source.Length; index++)
        {
            float expected = Math.Clamp(
                MathF.Sin(source[index]) * MathF.Exp(-source[index] * source[index]),
                -1.0f,
                1.0f);
            Assert.That(result[index], Is.EqualTo(expected).Within(1e-6f));
        }
    }

    [Test]
    public void Run_HandlesEmptyAndSingleElementArrays()
    {
        Assert.Multiple(() =>
        {
            Assert.That(Compute.Run([], value => value + 1.0f), Is.Empty);
            Assert.That(Compute.Run([2.0f], value => -value), Is.EqualTo(new[] { -2.0f }));
        });
    }

    [Test]
    public void Run_PreservesIeeeSpecialValues()
    {
        float[] source = [float.NaN, float.PositiveInfinity, float.NegativeInfinity, -0.0f];

        float[] multipliedByZero = Compute.Run(source, value => value * 0.0f);
        float[] addedToZero = Compute.Run(source, value => value + 0.0f);

        Assert.Multiple(() =>
        {
            Assert.That(multipliedByZero[0], Is.NaN);
            Assert.That(multipliedByZero[1], Is.NaN);
            Assert.That(multipliedByZero[2], Is.NaN);
            Assert.That(
                BitConverter.SingleToInt32Bits(addedToZero[3]),
                Is.EqualTo(BitConverter.SingleToInt32Bits(+0.0f)));
        });
    }

    [Test]
    public void Run_UsesFloatDivisionAndSqrtSemantics()
    {
        float[] division = Compute.Run([1.0f, -1.0f], value => value / 0.0f);
        float[] squareRoot = Compute.Run([-1.0f], value => GpuMath.Sqrt(value));

        Assert.Multiple(() =>
        {
            Assert.That(division[0], Is.EqualTo(float.PositiveInfinity));
            Assert.That(division[1], Is.EqualTo(float.NegativeInfinity));
            Assert.That(squareRoot[0], Is.NaN);
        });
    }

    [Test]
    public void Zip_AppliesBinaryExpression()
    {
        float[] result = Compute.Zip(
            [1.0f, 2.0f, 3.0f],
            [4.0f, 5.0f, 6.0f],
            (left, right) => left * right + 1.0f);

        Assert.That(result, Is.EqualTo(new[] { 5.0f, 11.0f, 19.0f }));
    }

    [Test]
    public void Zip_RejectsMismatchedLengths()
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => Compute.Zip([1.0f], [1.0f, 2.0f], (left, right) => left + right))!;

        Assert.That(exception.Message, Does.Contain("equal length"));
    }

    [Test]
    public void Sum_ReturnsSequentialFloatSumAndZeroForEmptyArray()
    {
        Assert.Multiple(() =>
        {
            Assert.That(Compute.Sum([1.0f, 2.0f, 3.0f]), Is.EqualTo(6.0f));
            Assert.That(Compute.Sum([]), Is.EqualTo(0.0f));
        });
    }

    [Test]
    public void UnsupportedMethod_ProducesActionableError()
    {
        GpuExpressionNotSupportedException exception =
            Assert.Throws<GpuExpressionNotSupportedException>(
                () => Compute.Run([1.0f], value => MathF.Sin(value)))!;

        Assert.Multiple(() =>
        {
            Assert.That(exception.NodeType, Is.EqualTo(System.Linq.Expressions.ExpressionType.Call));
            Assert.That(exception.ExpressionFragment, Does.Contain("Sin"));
            Assert.That(exception.Message, Does.Contain("GpuMath"));
        });
    }

    [Test]
    public void CapturedValue_IsRejectedUntilCapturedParametersAreImplemented()
    {
        float multiplier = 2.0f;

        GpuExpressionNotSupportedException exception =
            Assert.Throws<GpuExpressionNotSupportedException>(
                () => Compute.Run([1.0f], value => value * multiplier))!;

        Assert.That(exception.Message, Does.Contain("Captured values"));
    }

    [Test]
    public void ForcedUnavailableBackend_Throws()
    {
        var unavailableBackend = (ComputeBackendKind)int.MaxValue;
        var options = new ComputeOptions { Backend = unavailableBackend };

        ComputeBackendUnavailableException exception =
            Assert.Throws<ComputeBackendUnavailableException>(
                () => Compute.Run([1.0f], value => value, options))!;

        Assert.That(exception.Backend, Is.EqualTo(unavailableBackend));
    }

    [Test]
    public void CancelledOperation_ThrowsBeforePlanning()
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        var options = new ComputeOptions { CancellationToken = cancellation.Token };

        Assert.Throws<OperationCanceledException>(
            () => Compute.Run([1.0f], value => value, options));
    }

    [Test]
    public void FastOptimizationMode_IsExplicitlyUnavailable()
    {
        var options = new ComputeOptions
        {
            OptimizationMode = ComputeOptimizationMode.Fast
        };

        NotSupportedException exception = Assert.Throws<NotSupportedException>(
            () => Compute.Run([1.0f], value => value, options))!;

        Assert.That(exception.Message, Does.Contain("Fast optimization mode"));
    }

    [Test]
    public void PublicMethods_RejectNullArguments()
    {
        Assert.Multiple(() =>
        {
            Assert.Throws<ArgumentNullException>(
                () => Compute.Run(null!, value => value));
            Assert.Throws<ArgumentNullException>(
                () => Compute.Run([1.0f], null!));
            Assert.Throws<ArgumentNullException>(
                () => Compute.Zip(null!, [1.0f], (left, right) => left + right));
            Assert.Throws<ArgumentNullException>(
                () => Compute.Sum(null!));
        });
    }
}
