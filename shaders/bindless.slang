
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
