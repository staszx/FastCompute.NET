using System.Linq.Expressions;
using FastCompute.Expressions;

namespace FastCompute.Tests;

public sealed class StrictComputeOptimizerTests
{
    [Test]
    public void Optimize_FoldsConstantSubtree()
    {
        ParameterExpression parameter = Expression.Parameter(typeof(float), "value");
        Expression<Func<float, float>> expression = Expression.Lambda<Func<float, float>>(
            Expression.Add(
                parameter,
                Expression.Multiply(
                    Expression.Constant(2.0f),
                    Expression.Constant(4.0f))),
            parameter);

        ComputeExpressionPlan plan = ComputeExpressionParser.Parse(expression);
        ComputeExpressionPlan optimized = StrictComputeOptimizer.Optimize(plan);

        Assert.That(
            optimized.Root,
            Is.EqualTo(
                new BinaryNode(
                    BinaryOperation.Add,
                    new ParameterNode(0),
                    new ConstantNode(8.0f))));
    }

    [Test]
    public void Optimize_RemovesMultiplicationByOne()
    {
        Expression<Func<float, float>> expression = value => value * 1.0f;

        ComputeExpressionPlan plan = ComputeExpressionParser.Parse(expression);
        ComputeExpressionPlan optimized = StrictComputeOptimizer.Optimize(plan);

        Assert.That(optimized.Root, Is.EqualTo(new ParameterNode(0)));
    }

    [Test]
    public void Optimize_DoesNotRemoveMultiplicationByZero()
    {
        Expression<Func<float, float>> expression = value => value * 0.0f;

        ComputeExpressionPlan plan = ComputeExpressionParser.Parse(expression);
        ComputeExpressionPlan optimized = StrictComputeOptimizer.Optimize(plan);

        Assert.That(
            optimized.Root,
            Is.TypeOf<BinaryNode>().And.Property(nameof(BinaryNode.Operation))
                .EqualTo(BinaryOperation.Multiply));
    }

    [Test]
    public void Optimize_DoesNotRemoveAdditionOfPositiveZero()
    {
        Expression<Func<float, float>> expression = value => value + 0.0f;

        ComputeExpressionPlan plan = ComputeExpressionParser.Parse(expression);
        ComputeExpressionPlan optimized = StrictComputeOptimizer.Optimize(plan);

        Assert.That(
            optimized.Root,
            Is.TypeOf<BinaryNode>().And.Property(nameof(BinaryNode.Operation))
                .EqualTo(BinaryOperation.Add));
    }
}
