using System.Runtime.CompilerServices;

namespace FastCompute.ImageProcessing;

internal static class ImageGpuContextExtensions
{
    private static readonly ConditionalWeakTable<ComputeContext, ImageGpuContextServices> Services = new();

    internal static ImageGpuContextServices GetImageGpuServices(this ComputeContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        return Services.GetValue(context, static owner => new ImageGpuContextServices(owner));
    }
}
