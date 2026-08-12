// Source for VIDEO_CONVERT_VERT_SPV in vk_video_convert_spv.inc.
// A fullscreen triangle covering [0,1] in uv. See the .inc header to regenerate.
#version 450

layout(location = 0) out vec2 v_uv;

void main() {
  vec2 uv = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
  v_uv = uv;
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
