using System.Linq.Expressions;

namespace FastCompute;

/// <summary>
/// Represents an error reported by FastCompute.
/// </summary>
public class ComputeException : Exception
{
    /// <summary>Initializes an exception with a message.</summary>
    public ComputeException(string message)
        : base(message)
    {
    }

    /// <summary>Initializes an exception with a message and an inner exception.</summary>
    public ComputeException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}

/// <summary>
/// Indicates that a requested execution backend is not available.
/// </summary>
public sealed class ComputeBackendUnavailableException : ComputeException
{
    /// <summary>Initializes an exception for the requested backend.</summary>
    public ComputeBackendUnavailableException(ComputeBackendKind backend)
        : base($"The requested compute backend '{backend}' is not available.")
    {
        Backend = backend;
    }

    /// <summary>Gets the unavailable backend.</summary>
    public ComputeBackendKind Backend { get; }
}

/// <summary>
/// Indicates that a backend does not support the requested operation kind.
/// </summary>
public sealed class ComputeBackendNotSupportedException : ComputeException
{
    /// <summary>Initializes an exception for an unsupported backend operation.</summary>
    public ComputeBackendNotSupportedException(
        ComputeBackendKind backend,
        string operation,
        string supportedBackends)
        : base(
            $"The compute backend '{backend}' does not support {operation}. " +
            $"Supported backends: {supportedBackends}.")
    {
        Backend = backend;
        Operation = operation;
    }

    /// <summary>Gets the backend that rejected the operation.</summary>
    public ComputeBackendKind Backend { get; }

    /// <summary>Gets the rejected operation description.</summary>
    public string Operation { get; }
}

/// <summary>
/// Indicates that a compute expression could not be compiled.
/// </summary>
public sealed class ComputeCompilationException : ComputeException
{
    /// <summary>Initializes a compilation exception.</summary>
    public ComputeCompilationException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}

/// <summary>
/// Indicates that an expression contains a construct unsupported by compute backends.
/// </summary>
public sealed class GpuExpressionNotSupportedException : ComputeException
{
    /// <summary>Initializes an unsupported-expression exception.</summary>
    public GpuExpressionNotSupportedException(
        ExpressionType nodeType,
        string expressionFragment,
        string description,
        IReadOnlyList<string> allowedAlternatives)
        : base(CreateMessage(expressionFragment, description, allowedAlternatives))
    {
        NodeType = nodeType;
        ExpressionFragment = expressionFragment;
        AllowedAlternatives = allowedAlternatives.ToArray();
    }

    /// <summary>Gets the unsupported expression node type.</summary>
    public ExpressionType NodeType { get; }

    /// <summary>Gets the unsupported portion of the expression.</summary>
    public string ExpressionFragment { get; }

    /// <summary>Gets the supported alternatives.</summary>
    public IReadOnlyList<string> AllowedAlternatives { get; }

    private static string CreateMessage(
        string expressionFragment,
        string description,
        IReadOnlyList<string> allowedAlternatives)
    {
        string alternatives = string.Join(" ", allowedAlternatives);
        return $"The expression '{expressionFragment}' is not supported. {description} {alternatives}".TrimEnd();
    }
}

/// <summary>
/// Indicates that compute buffers cannot participate in the same operation.
/// </summary>
public sealed class ComputeBufferMismatchException : ComputeException
{
    /// <summary>Initializes a buffer mismatch exception.</summary>
    public ComputeBufferMismatchException(string message)
        : base(message)
    {
    }
}
