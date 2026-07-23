using System.Linq.Expressions;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using FastCompute.Expressions;

namespace FastCompute.Backends.Simd;

internal static class SimdExpressionCompiler
{
    private static readonly MethodInfo BroadcastMethod =
        GetMethod(nameof(SimdVectorOperations.Broadcast), 1);
    private static readonly IReadOnlyDictionary<UnaryOperation, MethodInfo> UnaryMethods =
        new Dictionary<UnaryOperation, MethodInfo>
        {
            [UnaryOperation.Negate] =
                GetMethod(nameof(SimdVectorOperations.Negate), 1)
        };
    private static readonly IReadOnlyDictionary<BinaryOperation, MethodInfo> BinaryMethods =
        new Dictionary<BinaryOperation, MethodInfo>
        {
            [BinaryOperation.Add] =
                GetMethod(nameof(SimdVectorOperations.Add), 2),
            [BinaryOperation.Subtract] =
                GetMethod(nameof(SimdVectorOperations.Subtract), 2),
            [BinaryOperation.Multiply] =
                GetMethod(nameof(SimdVectorOperations.Multiply), 2),
            [BinaryOperation.Divide] =
                GetMethod(nameof(SimdVectorOperations.Divide), 2)
        };
    private static readonly IReadOnlyDictionary<ComputeFunction, MethodInfo> FunctionMethods =
        new Dictionary<ComputeFunction, MethodInfo>
        {
            [ComputeFunction.Min] =
                GetMethod(nameof(SimdVectorOperations.Minimum), 2),
            [ComputeFunction.Max] =
                GetMethod(nameof(SimdVectorOperations.Maximum), 2),
            [ComputeFunction.Abs] =
                GetMethod(nameof(SimdVectorOperations.Absolute), 1),
            [ComputeFunction.Clamp] =
                GetMethod(nameof(SimdVectorOperations.Clamp), 3)
        };

    internal static bool Supports(ComputeExpressionPlan plan) =>
        Supports(plan.Root);

    internal static Func<Vector256<float>, Vector256<float>> CompileUnary(
        ComputeExpressionPlan plan)
    {
        var parameter = Expression.Parameter(typeof(Vector256<float>), "value");
        Expression body = Build(plan.Root, [parameter]);
        return Expression
            .Lambda<Func<Vector256<float>, Vector256<float>>>(body, parameter)
            .Compile();
    }

    internal static Func<
        Vector256<float>,
        Vector256<float>,
        Vector256<float>> CompileBinary(
        ComputeExpressionPlan plan)
    {
        var left = Expression.Parameter(typeof(Vector256<float>), "left");
        var right = Expression.Parameter(typeof(Vector256<float>), "right");
        Expression body = Build(plan.Root, [left, right]);
        return Expression
            .Lambda<
                Func<
                    Vector256<float>,
                    Vector256<float>,
                    Vector256<float>>>(body, left, right)
            .Compile();
    }

    private static bool Supports(ComputeNode node) => node switch
    {
        ParameterNode => true,
        ConstantNode => true,
        UnaryNode { Operation: UnaryOperation.Negate } unary =>
            Supports(unary.Operand),
        BinaryNode binary =>
            Supports(binary.Left) && Supports(binary.Right),
        FunctionNode function when FunctionMethods.ContainsKey(function.Function) =>
            function.Arguments.All(Supports),
        _ => false
    };

    private static Expression Build(
        ComputeNode node,
        IReadOnlyList<ParameterExpression> parameters) =>
        node switch
        {
            ParameterNode parameter => parameters[parameter.Index],
            ConstantNode constant => Expression.Call(
                BroadcastMethod,
                Expression.Constant(constant.Value)),
            UnaryNode unary => Expression.Call(
                UnaryMethods[unary.Operation],
                Build(unary.Operand, parameters)),
            BinaryNode binary => Expression.Call(
                BinaryMethods[binary.Operation],
                Build(binary.Left, parameters),
                Build(binary.Right, parameters)),
            FunctionNode function => Expression.Call(
                FunctionMethods[function.Function],
                function.Arguments
                    .Select(argument => Build(argument, parameters))),
            _ => throw new ComputeException(
                $"IR node '{node.GetType().Name}' is not supported by SIMD.")
        };

    private static MethodInfo GetMethod(string name, int parameterCount) =>
        typeof(SimdVectorOperations)
            .GetMethods(BindingFlags.Static | BindingFlags.NonPublic)
            .Single(method =>
                method.Name == name &&
                method.GetParameters().Length == parameterCount);
}

internal static class SimdVectorOperations
{
    private static readonly Vector256<float> AbsoluteValueMask =
        Vector256.Create(BitConverter.Int32BitsToSingle(int.MaxValue));
    private static readonly Vector256<float> SignMask =
        Vector256.Create(-0.0f);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> Broadcast(float value) =>
        Vector256.Create(value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> Negate(Vector256<float> value) =>
        Avx.Xor(value, SignMask);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> Add(
        Vector256<float> left,
        Vector256<float> right) =>
        Avx.Add(left, right);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> Subtract(
        Vector256<float> left,
        Vector256<float> right) =>
        Avx.Subtract(left, right);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> Multiply(
        Vector256<float> left,
        Vector256<float> right) =>
        Avx.Multiply(left, right);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> Divide(
        Vector256<float> left,
        Vector256<float> right) =>
        Avx.Divide(left, right);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> Absolute(Vector256<float> value) =>
        Avx.And(value, AbsoluteValueMask);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> Clamp(
        Vector256<float> value,
        Vector256<float> minimum,
        Vector256<float> maximum)
    {
        Vector256<float> invalidBounds = Avx.Compare(
            minimum,
            maximum,
            FloatComparisonMode.OrderedGreaterThanNonSignaling);
        if (Avx.MoveMask(invalidBounds) != 0)
        {
            throw new ArgumentException(
                "The minimum value cannot be greater than the maximum value.");
        }

        return Maximum(Minimum(value, maximum), minimum);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> Minimum(
        Vector256<float> left,
        Vector256<float> right)
    {
        Vector256<float> result = Avx.Min(left, right);
        Vector256<float> equal = Avx.Compare(
            left,
            right,
            FloatComparisonMode.OrderedEqualNonSignaling);
        result = Avx.BlendVariable(result, Avx.Or(left, right), equal);
        return PropagateNaN(result, left, right);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> Maximum(
        Vector256<float> left,
        Vector256<float> right)
    {
        Vector256<float> result = Avx.Max(left, right);
        Vector256<float> equal = Avx.Compare(
            left,
            right,
            FloatComparisonMode.OrderedEqualNonSignaling);
        result = Avx.BlendVariable(result, Avx.And(left, right), equal);
        return PropagateNaN(result, left, right);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> PropagateNaN(
        Vector256<float> result,
        Vector256<float> left,
        Vector256<float> right)
    {
        Vector256<float> unordered = Avx.Compare(
            left,
            right,
            FloatComparisonMode.UnorderedNonSignaling);
        return Avx.BlendVariable(result, Avx.Add(left, right), unordered);
    }
}
