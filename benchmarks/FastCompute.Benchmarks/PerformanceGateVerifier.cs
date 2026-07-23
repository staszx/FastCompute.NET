using BenchmarkDotNet.Reports;

namespace FastCompute.Benchmarks;

internal static class PerformanceGateVerifier
{
    internal const string Category = "PerformanceGate";

    private const string BaselineMethod = "ForLoop";
    private const string CandidateMethod = "FastComputeAuto";
    private const double MaximumAcceptedRatio = 1.05;

    internal static bool Passed(
        IReadOnlyList<Summary> summaries,
        TextWriter output)
    {
        bool passed = summaries.Count > 0;

        foreach (Summary summary in summaries)
        {
            BenchmarkReport? baseline = FindReport(summary, BaselineMethod);
            BenchmarkReport? candidate = FindReport(summary, CandidateMethod);

            if (baseline?.ResultStatistics is null ||
                candidate?.ResultStatistics is null)
            {
                output.WriteLine(
                    $"PERF FAIL {summary.Title}: complete results for " +
                    $"{BaselineMethod} and {CandidateMethod} are required.");
                passed = false;
                continue;
            }

            double ratio =
                candidate.ResultStatistics.Mean /
                baseline.ResultStatistics.Mean;
            bool comparisonPassed = ratio <= MaximumAcceptedRatio;

            output.WriteLine(
                $"{(comparisonPassed ? "PERF PASS" : "PERF FAIL")} " +
                $"{summary.Title}: FastCompute/for = {ratio:F3}, " +
                $"required <= {MaximumAcceptedRatio:F2}.");

            passed &= comparisonPassed;
        }

        return passed;
    }

    private static BenchmarkReport? FindReport(
        Summary summary,
        string methodName)
    {
        foreach (BenchmarkReport report in summary.Reports)
        {
            if (string.Equals(
                    report.BenchmarkCase.Descriptor.WorkloadMethod.Name,
                    methodName,
                    StringComparison.Ordinal))
            {
                return report;
            }
        }

        return null;
    }
}
