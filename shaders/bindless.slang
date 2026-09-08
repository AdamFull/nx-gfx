
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
