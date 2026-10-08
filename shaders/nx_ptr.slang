// NxPtr<T>: a device address shaders read and write T through, the Slang
// half of the pair whose C++ half is nx_ptr.hpp. Included by core.slang.

public uint64_t stride_of<T>()
{
    uint64_t size = uint64_t(sizeof(T));
    uint64_t align = uint64_t(alignof(T));
    return (size + align - 1) & ~(align - 1);
}

F load_raw<F>(uint64_t address)
{
    // Mali/Intel Vulkan crash pipeline compilation on an access chain that
    // yields a physical pointer-to-struct; copying as uint words never forms
    // one. Do not "simplify" back to *(F*).
    static_assert(sizeof(F) % sizeof(uint) == 0, "type must be 32-bit-addressable");
    static const uint word_count = sizeof(F) / sizeof(uint);
    static const uint vector_count = word_count / 4u;
    static const uint vector_words = vector_count * 4u;
    uint words[word_count];
    uint4* vectors = (uint4*)address;
    [ForceUnroll] for (uint v = 0; v < vector_count; ++v) {
        const uint4 word = vectors[v];
        words[v * 4u + 0u] = word.x;
        words[v * 4u + 1u] = word.y;
        words[v * 4u + 2u] = word.z;
        words[v * 4u + 3u] = word.w;
    }
    uint* scalars = (uint*)address;
    [ForceUnroll] for (uint w = vector_words; w < word_count; ++w)
        words[w] = scalars[w];
    return reinterpret<F>(words);
}

void store_raw<F>(uint64_t address, in F value)
{
    static_assert(sizeof(F) % sizeof(uint) == 0, "type must be 32-bit-addressable");
    static const uint word_count = sizeof(F) / sizeof(uint);
    static const uint vector_count = word_count / 4u;
    static const uint vector_words = vector_count * 4u;
    uint words[word_count] = reinterpret<uint[word_count]>(value);
    uint4* vectors = (uint4*)address;
    [ForceUnroll] for (uint v = 0; v < vector_count; ++v) {
        const uint w = v * 4u;
        vectors[v] = uint4(words[w], words[w + 1u], words[w + 2u],
                           words[w + 3u]);
    }
    uint* scalars = (uint*)address;
    [ForceUnroll] for (uint w = vector_words; w < word_count; ++w)
        scalars[w] = words[w];
    return;
}

public struct NxPtr<T>
{
    public uint64_t value;

    public bool is_valid() { return value != 0; }

    uint64_t element_address(uint index) { return value + uint64_t(index) * stride_of<T>(); }
    uint64_t field_address(uint index, uint field_offset) { return element_address(index) + uint64_t(field_offset); }

    public T load(uint index) { return load_raw<T>(element_address(index)); }
    public void store(uint index, in T newValue) { store_raw<T>(element_address(index), newValue); }

    public __subscript(uint index) -> T
    {
        get { return load(index); }
        // A store mutates external memory, not the address; nonmutating lets
        // addr[i] = x work on a read-only (push-constant) address.
        [nonmutating] set { store(index, newValue); }
    }

    public F get_field<F>(uint index, uint field_offset) { return load_raw<F>(field_address(index, field_offset)); }
    public void set_field<F>(uint index, uint field_offset, in F fieldValue) { store_raw<F>(field_address(index, field_offset), fieldValue); }
}

public NxPtr<T> operator+<T>(NxPtr<T> a, uint n)
{
    NxPtr<T> r = a;
    r.value = a.value + uint64_t(n) * stride_of<T>();
    return r;
}
public NxPtr<T> operator+<T>(uint n, NxPtr<T> a) { return a + n; }
public NxPtr<T> operator-<T>(NxPtr<T> a, uint n)
{
    NxPtr<T> r = a;
    r.value = a.value - uint64_t(n) * stride_of<T>();
    return r;
}
public bool operator==<T>(NxPtr<T> a, NxPtr<T> b) { return a.value == b.value; }
public bool operator!=<T>(NxPtr<T> a, NxPtr<T> b) { return a.value != b.value; }

// Atomics on 32-bit words, for counters and cursors the GPU claims from.
public extension NxPtr<uint>
{
    /// Adds @p value to word @p index and returns what it held before.
    public uint atomic_add(uint index, uint value)
    {
        uint* word = (uint*)element_address(index);
        uint before;
        InterlockedAdd(*word, value, before);
        return before;
    }

    /// Keeps the greater of word @p index and @p value in it, and returns
    /// what it held before.
    public uint atomic_max(uint index, uint value)
    {
        uint* word = (uint*)element_address(index);
        uint before;
        InterlockedMax(*word, value, before);
        return before;
    }
}

public extension NxPtr<int>
{
    /// Adds @p value to word @p index and returns what it held before.
    public int atomic_add(uint index, int value)
    {
        int* word = (int*)element_address(index);
        int before;
        InterlockedAdd(*word, value, before);
        return before;
    }
}
