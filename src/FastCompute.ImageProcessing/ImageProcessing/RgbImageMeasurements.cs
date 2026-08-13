namespace FastCompute.ImageProcessing;

/// <summary>Extracts reusable measurements from native RGB images.</summary>
public static class RgbImageMeasurements
{
    /// <summary>Calculates same-pixel and one-pixel channel correlations.</summary>
    public static RgbCorrelationMeasurements CalculateCorrelations(Image<Rgb> source, ComputeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        RgbChannels channels = source.SplitRgbChannels(options);
        float[] redRight = Offset(channels.Red, source.Width, source.Height, 1, 0, out float[] redRightBase);
        float[] greenRight = Offset(channels.Green, source.Width, source.Height, 1, 0, out _);
        float[] blueRight = Offset(channels.Blue, source.Width, source.Height, 1, 0, out float[] blueRightBase);
        float[] redDown = Offset(channels.Red, source.Width, source.Height, 0, 1, out float[] redDownBase);
        float[] greenDown = Offset(channels.Green, source.Width, source.Height, 0, 1, out _);
        float[] blueDown = Offset(channels.Blue, source.Width, source.Height, 0, 1, out float[] blueDownBase);
        _ = redRight;
        _ = blueRight;
        _ = redDown;
        _ = blueDown;
        return new RgbCorrelationMeasurements(
            [
                (float)Compute.Correlation(channels.Red, channels.Green, options),
                (float)Compute.Correlation(channels.Red, channels.Blue, options),
                (float)Compute.Correlation(channels.Green, channels.Blue, options)
            ],
            [
                (float)Compute.Correlation(redRightBase, greenRight, options),
                (float)Compute.Correlation(redDownBase, greenDown, options),
                (float)Compute.Correlation(blueRightBase, greenRight, options),
                (float)Compute.Correlation(blueDownBase, greenDown, options)
            ]);
    }

    /// <summary>Calculates per-parity mean absolute residual from a three-neighbour interpolation estimate.</summary>
    public static CfaParityMeasurements CalculateCfaParityResiduals(Image<Rgb> source, ComputeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        RgbChannels channels = source.SplitRgbChannels(options);
        float[][] channelData = [channels.Red, channels.Green, channels.Blue];
        var means = new double[4, 3];
        for (int channel = 0; channel < 3; channel++)
        {
            float[] residual = InterpolationResidual(channelData[channel], source.Width, source.Height, options);
            float[][] parity = BayerOperations.ExtractParitySamples(residual, source.Width, source.Height, options);
            for (int group = 0; group < 4; group++) means[group, channel] = Compute.Mean(parity[group], options);
        }
        return new CfaParityMeasurements { MeanResiduals = means };
    }

    private static float[] InterpolationResidual(float[] channel, int width, int height, ComputeOptions? options)
    {
        float[] left = ShiftClamp(channel, width, height, -1, 0);
        float[] right = ShiftClamp(channel, width, height, 1, 0);
        float[] up = ShiftClamp(channel, width, height, 0, -1);
        float[] sum = Compute.Zip(left, right, (first, second) => first + second, options);
        Compute.ZipInPlace(sum, up, (first, second) => first + second, options);
        Compute.RunInPlace(sum, value => value / 3f, options);
        return Compute.Zip(channel, sum, (center, estimate) => ComputeMath.Abs(center - estimate), options);
    }

    private static float[] ShiftClamp(float[] source, int width, int height, int offsetX, int offsetY)
    {
        var result = GC.AllocateUninitializedArray<float>(source.Length);
        Parallel.For(0, height, y =>
        {
            int sourceY = Math.Clamp(y + offsetY, 0, height - 1);
            for (int x = 0; x < width; x++)
            {
                int sourceX = Math.Clamp(x + offsetX, 0, width - 1);
                result[(y * width) + x] = source[(sourceY * width) + sourceX];
            }
        });
        return result;
    }

    private static float[] Offset(float[] source, int width, int height, int offsetX, int offsetY, out float[] sourceValues)
    {
        int resultWidth = width - Math.Abs(offsetX);
        int resultHeight = height - Math.Abs(offsetY);
        int startX = Math.Max(0, -offsetX);
        int startY = Math.Max(0, -offsetY);
        sourceValues = ImageRegionOperations.Crop(source, width, height, startX, startY, resultWidth, resultHeight);
        return ImageRegionOperations.Crop(source, width, height, startX + offsetX, startY + offsetY, resultWidth, resultHeight);
    }
}
