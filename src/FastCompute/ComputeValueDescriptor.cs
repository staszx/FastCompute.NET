using System.Linq.Expressions;
using System.Reflection;
using System.Runtime.InteropServices;

namespace FastCompute;

/// <summary>Identifies the physical scalar type used by compute components.</summary>
public enum ComputeComponentType
{
    /// <summary>A 32-bit IEEE floating-point component.</summary>
    Float32,

    /// <summary>An unsigned 8-bit integer component.</summary>
    Byte
}

/// <summary>
/// Describes an unmanaged value as a tightly packed sequence of homogeneous
/// scalar components that FastCompute can process independently.
/// </summary>
/// <typeparam name="T">The described value type.</typeparam>
public sealed class ComputeValueDescriptor<T>
    where T : unmanaged
{
    private const int MaximumComponentCount = 16;
    private readonly IReadOnlyDictionary<MemberInfo, int> componentIndexes;

    private ComputeValueDescriptor(
        IReadOnlyList<MemberInfo> components,
        IReadOnlyDictionary<MemberInfo, int> componentIndexes,
        ComputeComponentType componentType,
        int componentSize)
    {
        Components = components;
        this.componentIndexes = componentIndexes;
        ComponentType = componentType;
        ComponentSize = componentSize;
    }

    /// <summary>
    /// Gets the ordered component members.
    /// </summary>
    public IReadOnlyList<MemberInfo> Components { get; }

    /// <summary>
    /// Gets the number of floating-point components in one value.
    /// </summary>
    public int ComponentCount => Components.Count;

    /// <summary>Gets the physical type shared by all components.</summary>
    public ComputeComponentType ComponentType { get; }

    /// <summary>Gets the physical size of one component in bytes.</summary>
    public int ComponentSize { get; }

    /// <summary>
    /// Creates and validates a descriptor from component selectors in physical
    /// memory order.
    /// </summary>
    /// <remarks>
    /// The described type must use sequential, tightly packed storage containing
    /// only the selected component fields. Auto-properties are supported when
    /// their compiler-generated backing fields follow the same layout.
    /// </remarks>
    /// <param name="components">Selectors ordered by their physical layout.</param>
    /// <returns>A validated descriptor.</returns>
    public static ComputeValueDescriptor<T> Create(
        params Expression<Func<T, float>>[] components) =>
        CreateCore(components, ComputeComponentType.Float32);

    /// <summary>
    /// Creates and validates a descriptor for tightly packed byte components.
    /// </summary>
    public static ComputeValueDescriptor<T> Create(
        params Expression<Func<T, byte>>[] components) =>
        CreateCore(components, ComputeComponentType.Byte);

    private static ComputeValueDescriptor<T> CreateCore<TComponent>(
        Expression<Func<T, TComponent>>[] components,
        ComputeComponentType componentType)
        where TComponent : unmanaged
    {
        ArgumentNullException.ThrowIfNull(components);
        if (components.Length is 0 or > MaximumComponentCount)
        {
            throw new ArgumentOutOfRangeException(
                nameof(components),
                components.Length,
                $"A compute value must contain between 1 and {MaximumComponentCount} components.");
        }

        int componentSize = Marshal.SizeOf<TComponent>();
        if (Marshal.SizeOf<T>() != checked(components.Length * componentSize))
        {
            throw new ArgumentException(
                $"'{typeof(T).Name}' must consist of exactly {components.Length} " +
                $"tightly packed {typeof(TComponent).Name} components.",
                nameof(components));
        }

        var members = new MemberInfo[components.Length];
        var indexes = new Dictionary<MemberInfo, int>(components.Length);
        for (int index = 0; index < components.Length; index++)
        {
            Expression body = components[index]?.Body ??
                throw new ArgumentException(
                    "Component selectors cannot contain null values.",
                    nameof(components));
            if (body is not MemberExpression memberExpression ||
                memberExpression.Expression != components[index].Parameters[0] ||
                memberExpression.Type != typeof(TComponent))
            {
                throw new ArgumentException(
                    $"Every component selector must directly select a " +
                    $"{typeof(TComponent).Name} field or property.",
                    nameof(components));
            }

            MemberInfo member = memberExpression.Member;
            int offset = GetMemberOffset(member, typeof(TComponent));
            int expectedOffset = checked(index * componentSize);
            if (offset != expectedOffset)
            {
                throw new ArgumentException(
                    $"Component '{member.Name}' has byte offset {offset}; " +
                    $"the descriptor requires offset {expectedOffset}.",
                    nameof(components));
            }

            if (!indexes.TryAdd(member, index))
            {
                throw new ArgumentException(
                    $"Component '{member.Name}' was selected more than once.",
                    nameof(components));
            }

            members[index] = member;
        }

        return new ComputeValueDescriptor<T>(
            members,
            indexes,
            componentType,
            componentSize);
    }

    internal int GetComponentIndex(MemberInfo member)
    {
        if (!componentIndexes.TryGetValue(member, out int index))
        {
            throw new NotSupportedException(
                $"Member '{member.Name}' is not a registered component of '{typeof(T).Name}'.");
        }

        return index;
    }

    private static int GetMemberOffset(MemberInfo member, Type componentType)
    {
        FieldInfo field = member switch
        {
            FieldInfo directField when directField.FieldType == componentType =>
                directField,
            PropertyInfo property when
                property.PropertyType == componentType &&
                property.GetMethod is not null =>
                typeof(T).GetField(
                    $"<{property.Name}>k__BackingField",
                    BindingFlags.Instance | BindingFlags.NonPublic) ??
                throw new ArgumentException(
                    $"Property '{property.Name}' must be an auto-property backed by a " +
                    $"{componentType.Name} field."),
            _ => throw new ArgumentException(
                $"Member '{member.Name}' is not a {componentType.Name} field or " +
                "readable auto-property.")
        };

        return checked((int)Marshal.OffsetOf<T>(field.Name));
    }
}
