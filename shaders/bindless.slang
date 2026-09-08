
module bindless;

public static const uint kInvalidIndex = 0xFFFFFFFF;

#if defined(NX_TARGET_METAL)

// Declaration order fixes the byte offsets the C++ side writes slots at, so it
// must not change without updating create_bindless_heap().
public struct BindlessTables {
    public Texture2D textures[65536];
    public RWTexture2D<float4> rw_textures[8192];
    public SamplerState samplers[1024];
};

public ParameterBlock<BindlessTables> g_bindless;

public Texture2D get_texture(uint index) { return g_bindless.textures[index]; }
public RWTexture2D<float4> get_rw_texture(uint index) {
    return g_bindless.rw_textures[index];
}
public SamplerState get_sampler(uint index) {
    return g_bindless.samplers[index];
}

#else

[[vk::binding(0, 0)]]
SamplerState g_samplers[];

[[vk::binding(1, 0)]]
Texture2D g_textures[];

[[vk::binding(2, 0)]]
RWTexture2D<float4> g_rw_textures[];

public Texture2D get_texture(uint index) { return g_textures[index]; }
public RWTexture2D<float4> get_rw_texture(uint index) {
    return g_rw_textures[index];
}
public SamplerState get_sampler(uint index) { return g_samplers[index]; }

#endif

public float4 sample_texture(uint texture_index, uint sampler_index, float2 uv) {
    return get_texture(texture_index).Sample(get_sampler(sampler_index), uv);
}

public float4 sample_texture_lod(uint texture_index, uint sampler_index,
                                 float2 uv, float lod) {
    return get_texture(texture_index).SampleLevel(get_sampler(sampler_index), uv,
                                                  lod);
}

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
    uint words[word_count];
    uint* source = (uint*)address;
    [ForceUnroll] for (uint w = 0; w < word_count; ++w) words[w] = source[w];
    return reinterpret<F>(words);
}

void store_raw<F>(uint64_t address, in F value)
{
    static_assert(sizeof(F) % sizeof(uint) == 0, "type must be 32-bit-addressable");
    static const uint word_count = sizeof(F) / sizeof(uint);
    uint words[word_count] = reinterpret<uint[word_count]>(value);
    uint* destination = (uint*)address;
    [ForceUnroll] for (uint w = 0; w < word_count; ++w) destination[w] = words[w];
    return;
}

public struct GpuPtr<T>
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

    public uint atomic_add(uint index, uint field_offset, uint operand) { uint* p = (uint*)field_address(index, field_offset); uint prev; InterlockedAdd(p[0], operand, prev); return prev; }
    public uint atomic_min(uint index, uint field_offset, uint operand) { uint* p = (uint*)field_address(index, field_offset); uint prev; InterlockedMin(p[0], operand, prev); return prev; }
    public uint atomic_max(uint index, uint field_offset, uint operand) { uint* p = (uint*)field_address(index, field_offset); uint prev; InterlockedMax(p[0], operand, prev); return prev; }
    public uint atomic_and(uint index, uint field_offset, uint operand) { uint* p = (uint*)field_address(index, field_offset); uint prev; InterlockedAnd(p[0], operand, prev); return prev; }
    public uint atomic_or (uint index, uint field_offset, uint operand) { uint* p = (uint*)field_address(index, field_offset); uint prev; InterlockedOr (p[0], operand, prev); return prev; }
    public uint atomic_xor(uint index, uint field_offset, uint operand) { uint* p = (uint*)field_address(index, field_offset); uint prev; InterlockedXor(p[0], operand, prev); return prev; }
    public uint atomic_exchange(uint index, uint field_offset, uint operand) { uint* p = (uint*)field_address(index, field_offset); uint prev; InterlockedExchange(p[0], operand, prev); return prev; }
    public uint atomic_compare_exchange(uint index, uint field_offset, uint compare, uint operand) { uint* p = (uint*)field_address(index, field_offset); uint prev; InterlockedCompareExchange(p[0], compare, operand, prev); return prev; }

    public int atomic_add(uint index, uint field_offset, int operand) { int* p = (int*)field_address(index, field_offset); int prev; InterlockedAdd(p[0], operand, prev); return prev; }
    public int atomic_exchange(uint index, uint field_offset, int operand) { int* p = (int*)field_address(index, field_offset); int prev; InterlockedExchange(p[0], operand, prev); return prev; }
    public int atomic_compare_exchange(uint index, uint field_offset, int compare, int operand) { int* p = (int*)field_address(index, field_offset); int prev; InterlockedCompareExchange(p[0], compare, operand, prev); return prev; }
}

public GpuPtr<T> operator+<T>(GpuPtr<T> a, uint n)
{
    GpuPtr<T> r = a;
    r.value = a.value + uint64_t(n) * stride_of<T>();
    return r;
}
public GpuPtr<T> operator+<T>(uint n, GpuPtr<T> a) { return a + n; }
public GpuPtr<T> operator-<T>(GpuPtr<T> a, uint n)
{
    GpuPtr<T> r = a;
    r.value = a.value - uint64_t(n) * stride_of<T>();
    return r;
}
public bool operator==<T>(GpuPtr<T> a, GpuPtr<T> b) { return a.value == b.value; }
public bool operator!=<T>(GpuPtr<T> a, GpuPtr<T> b) { return a.value != b.value; }
