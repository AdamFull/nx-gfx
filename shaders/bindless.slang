// The shader side of the bindless binding contract. Set and binding numbers
// must match rhi/bindless.h. Import this module in every shader.

module bindless;

[[vk::binding(0, 0)]]
public SamplerState g_samplers[];

[[vk::binding(1, 0)]]
public Texture2D g_textures[];

[[vk::binding(2, 0)]]
public RWTexture2D<float4> g_rw_textures[];

public static const uint kInvalidIndex = 0xFFFFFFFF;

// Push-constant indices are uniform across a draw, so no NonUniformResourceIndex
// is needed here. Wrap indices that vary per invocation at the call site.
public float4 sample_texture(uint texture_index, uint sampler_index, float2 uv) {
    return g_textures[texture_index].Sample(g_samplers[sampler_index], uv);
}

public float4 sample_texture_lod(uint texture_index, uint sampler_index,
                                 float2 uv, float lod) {
    return g_textures[texture_index].SampleLevel(g_samplers[sampler_index], uv,
                                                 lod);
}
