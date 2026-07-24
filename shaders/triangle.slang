// Sample shader: a vertex-coloured triangle that tints itself with a bindless
// texture when one is supplied. Vertices are generated from SV_VertexID, so
// there is no vertex buffer - the vertex-pulling model the 2D renderer uses.

import bindless;

struct PushConstants {
    uint texture_index;
    uint sampler_index;
};

[[vk::push_constant]]
PushConstants pc;

struct VertexOutput {
    float4 position : SV_Position;
    float3 color : COLOR0;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vs_main(uint vertex_id: SV_VertexID) {
    const float2 positions[3] = {
        float2(0.0, -0.6),
        float2(0.6, 0.6),
        float2(-0.6, 0.6),
    };
    const float3 colors[3] = {
        float3(1.0, 0.0, 0.0),
        float3(0.0, 1.0, 0.0),
        float3(0.0, 0.0, 1.0),
    };

    VertexOutput output;
    output.position = float4(positions[vertex_id], 0.0, 1.0);
    output.color = colors[vertex_id];
    output.uv = positions[vertex_id] * 0.5 + 0.5;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input): SV_Target {
    float3 rgb = input.color;
    if (pc.texture_index != kInvalidIndex) {
        rgb *= sample_texture(pc.texture_index, pc.sampler_index, input.uv).rgb;
    }
    return float4(rgb, 1.0);
}
