namespace FastCompute;

/// <summary>Contains an ordinary least-squares line fit.</summary>
public readonly struct LinearRegressionResult
{
    internal LinearRegressionResult(double slope, double intercept, double rSquared)
    {
        Slope = slope;
        Intercept = intercept;
        RSquared = rSquared;
    }

    /// <summary>Gets the fitted slope.</summary>
    public double Slope { get; }
    /// <summary>Gets the fitted intercept.</summary>
    public double Intercept { get; }
    /// <summary>Gets the coefficient of determination.</summary>
    public double RSquared { get; }
}
