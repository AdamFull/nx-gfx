// Source for VIDEO_CONVERT_FRAG_SPV in vk_video_convert_spv.inc.
// Samples the decoded picture through the ycbcr conversion baked into the
// immutable sampler at set 0, binding 0, so texture() returns RGB. See the .inc
// header to regenerate.
#version 450

layout(set = 0, binding = 0) uniform sampler2D u_tex;
layout(push_constant) uniform Push {
  vec2 uv_scale;
} pc;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

void main() {
  o_color = texture(u_tex, v_uv * pc.uv_scale);
}
