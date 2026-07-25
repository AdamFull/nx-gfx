// The shader side of the bindless binding contract. Import this module in
// every shader; reach resources through the accessors below rather than
// through the tables, which are declared differently per backend.
//
// Vulkan gets descriptor indexing: unbounded arrays in set 0, whose set and
// binding numbers must match rhi/bindless.h.
//
// Metal has no descriptor indexing at all, so the same resources are a
// ParameterBlock, which Slang lowers to an argument buffer. That needs Tier 2
// hardware and fixed array bounds - the sizes below must match the
// MAX_BINDLESS_* constants in rhi/bindless.h, because the C++ side computes
// the slot offsets into the buffer from them.
//
// NX_TARGET_METAL is defined by cmake/NxShaders.cmake when compiling for
// metallib.

module bindless;

public static const uint kInvalidIndex = 0xFFFFFFFF;

#if defined(NX_TARGET_METAL)

public struct BindlessTables {
    public Texture2D textures[65536];            // MAX_BINDLESS_TEXTURES
    public RWTexture2D<float4> rw_textures[8192]; // MAX_BINDLESS_RW_TEXTURES
    public SamplerState samplers[1024];           // MAX_BINDLESS_SAMPLERS
};

// Slang assigns this the first free buffer index after the push block, so it
// lands at [[buffer(1)]] with push constants at [[buffer(0)]]. The device
// binds it once per encoder and never rebinds.
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

// Push-constant indices are uniform across a draw, so no NonUniformResourceIndex
// is needed here. Wrap indices that vary per invocation at the call site.
public float4 sample_texture(uint texture_index, uint sampler_index, float2 uv) {
    return get_texture(texture_index).Sample(get_sampler(sampler_index), uv);
}

public float4 sample_texture_lod(uint texture_index, uint sampler_index,
                                 float2 uv, float lod) {
    return get_texture(texture_index).SampleLevel(get_sampler(sampler_index), uv,
                                                  lod);
}
