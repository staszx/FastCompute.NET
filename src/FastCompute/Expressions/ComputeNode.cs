namespace FastCompute.Expressions;

internal enum ComputeDataType
{
    Float32
}

internal enum UnaryOperation
{
    Negate
}

internal enum BinaryOperation
{
    Add,
    Subtract,
    Multiply,
    Divide
}

internal enum ComputeFunction
{
    Abs,
    Min,
    Max,
    Clamp,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Exp,
    Log,
    Log10,
    Pow,
    Floor,
    Ceiling,
    Round
}

internal abstract record ComputeNode(ComputeDataType DataType);

internal sealed record ParameterNode(int Index)
    : ComputeNode(ComputeDataType.Float32);

internal sealed record ConstantNode(float Value)
    : ComputeNode(ComputeDataType.Float32);

internal sealed record UnaryNode(UnaryOperation Operation, ComputeNode Operand)
    : ComputeNode(ComputeDataType.Float32);

internal sealed record BinaryNode(
    BinaryOperation Operation,
    ComputeNode Left,
    ComputeNode Right)
    : ComputeNode(ComputeDataType.Float32);

internal sealed record FunctionNode(
    ComputeFunction Function,
    IReadOnlyList<ComputeNode> Arguments)
    : ComputeNode(ComputeDataType.Float32);

internal sealed record ComputeExpressionPlan(
    int ParameterCount,
    ComputeNode Root);
