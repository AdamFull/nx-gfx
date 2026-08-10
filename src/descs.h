#pragma once

/**
 * @file descs.h
 * @brief Resource creation descriptors (namespace nxe::rhi).
 *
 */

#include "core/foundation/core/foundation.h"
#include "core/foundation/platform/native_window.h"
#include "core/foundation/strings/utf8_string_view.h"
#include "core/rendering/rhi/handles.h"
#include "core/rendering/rhi/types.h"

namespace nxe::rhi {

struct Extent2D {
  u32 width = 0;
  u32 height = 0;
};

struct Extent3D {
  u32 width = 0;
  u32 height = 0;
  u32 depth = 1;
};

struct Offset2D {
  i32 x = 0;
  i32 y = 0;
};

struct Rect2D {
  Offset2D offset;
  Extent2D extent;
};

struct Viewport {
  f32 x = 0.f, y = 0.f;
  f32 width = 0.f, height = 0.f;
  f32 min_depth = 0.f, max_depth = 1.f;
};

union ClearValue {
  f32 color[4];
  struct {
    f32 depth;
    u32 stencil;
  } depth_stencil;
};

[[nodiscard]] constexpr ClearValue
clear_color(const f32 r, const f32 g, const f32 b, const f32 a = 1.f) noexcept {
  ClearValue v{};
  v.color[0] = r;
  v.color[1] = g;
  v.color[2] = b;
  v.color[3] = a;
  return v;
}

[[nodiscard]] constexpr ClearValue clear_depth(const f32 depth = 1.f,
                                               const u32 stencil = 0) noexcept {
  ClearValue v{};
  v.depth_stencil.depth = depth;
  v.depth_stencil.stencil = stencil;
  return v;
}

struct TextureDesc {
  nx::string_view name = "texture";
  Format format = Format::Unknown;
  TextureDimension dimension = TextureDimension::Tex2D;
  u32 width = 1;
  u32 height = 1;
  /// Depth for Tex3D, array length for Tex2DArray, 6 (or 6*N) for TexCube.
  u32 depth_or_layers = 1;
  u32 mip_levels = 1;
  /// MSAA sample count; 1 disables it. 2D rendering rarely wants more.
  u32 sample_count = 1;
  TextureUsageFlags usage = TextureUsage::Sampled;
  MemoryUsage memory = MemoryUsage::DeviceLocal;
  /// State the texture is in immediately after creation.
  ResourceState initial_state = ResourceState::Undefined;
};

/// What a resource needs from a heap. Query it before packing, then place the
/// resource at an offset that satisfies the alignment.
struct MemoryRequirements {
  u64 size = 0;
  u64 alignment = 1;
  u32 memory_type_bits = ~0u;
};

/// Memory the caller sub-allocates itself. Resources placed at non-overlapping
/// offsets are independent; resources placed at overlapping offsets alias, and
/// must not be in use at the same time.
struct HeapDesc {
  nx::string_view name = "heap";
  u64 size = 0;
  u64 alignment = 256;
  /// Intersection of the memory_type_bits of everything to be placed here.
  u32 memory_type_bits = ~0u;
  MemoryUsage memory = MemoryUsage::DeviceLocal;
};

struct BufferDesc {
  nx::string_view name = "buffer";
  u64 size = 0;
  BufferUsageFlags usage = BufferUsage::Storage;
  MemoryUsage memory = MemoryUsage::DeviceLocal;
  /// Keep the mapping alive for the buffer's lifetime. Upload-heap ring
  /// buffers want this; mapping per frame is pure overhead.
  bool persistently_mapped = false;
};

struct SamplerDesc {
  nx::string_view name = "sampler";
  FilterMode min_filter = FilterMode::Linear;
  FilterMode mag_filter = FilterMode::Linear;
  FilterMode mip_filter = FilterMode::Linear;
  AddressMode address_u = AddressMode::ClampToEdge;
  AddressMode address_v = AddressMode::ClampToEdge;
  AddressMode address_w = AddressMode::ClampToEdge;
  f32 mip_lod_bias = 0.f;
  f32 max_anisotropy = 1.f; ///< 1 disables anisotropic filtering.
  f32 min_lod = 0.f;
  f32 max_lod = 1000.f;
  /// Set for shadow/percentage-closer sampling; Never disables comparison.
  CompareOp compare = CompareOp::Never;
};

struct RenderAttachment {
  TextureHandle texture;
  u32 mip_level = 0;
  u32 base_layer = 0;
  LoadOp load = LoadOp::Clear;
  StoreOp store = StoreOp::Store;
  ClearValue clear = {};
  /// MSAA resolve target; null handle disables resolve.
  TextureHandle resolve_target;
};

inline constexpr u32 MAX_COLOR_ATTACHMENTS = 8;

struct RenderPassDesc {
  nx::string_view name = "pass";
  RenderAttachment color[MAX_COLOR_ATTACHMENTS] = {};
  u32 color_count = 0;
  RenderAttachment depth_stencil = {};
  bool has_depth_stencil = false;
  /// Render area; zero extent means the full size of the first attachment.
  Rect2D render_area = {};
};

/// Swapchain creation / recreation parameters.
struct SwapchainDesc {
  /// Native surface, straight from IWindow::native_handle().
  nx::NativeWindowHandle window;
  u32 width = 0;
  u32 height = 0;
  PresentMode present_mode = PresentMode::Fifo;
  /// Preferred backbuffer format; the backend picks the closest supported one
  /// and reports the result via Swapchain::format().
  Format preferred_format = Format::BGRA8_UNORM;
};

#if defined(NX_RHI_METAL)
inline constexpr nx::string_view SHADER_BINARY_EXT = ".metallib";
#else
inline constexpr nx::string_view SHADER_BINARY_EXT = ".spv";
#endif

/// A compiled shader blob (SPIR-V on Vulkan, metallib on Metal). One blob may
/// hold several entry points; a pipeline names the one it wants.
struct ShaderDesc {
  nx::string_view name = "shader";
  const void *code = nullptr;
  usize code_size = 0;
  nx::string_view reflection_json;
};

/// A shader stage in a pipeline: which blob, and which entry point in it.
struct ShaderEntry {
  ShaderHandle shader;
  nx::string_view entry_point = "main";
};

/// Colour target blending, one per attachment.
struct BlendState {
  bool enabled = false;
  BlendMode mode = BlendMode::Opaque;
  /// Per-channel write mask, bit 0..3 = R,G,B,A.
  u8 write_mask = 0xF;
};

struct GraphicsPipelineDesc {
  nx::string_view name = "pipeline";
  ShaderEntry vertex;
  ShaderEntry fragment;

  /// Attachment formats, for dynamic rendering. Must match the render pass the
  /// pipeline is used in.
  Format color_formats[MAX_COLOR_ATTACHMENTS] = {};
  u32 color_count = 0;
  Format depth_format = Format::Unknown;
  BlendState blend[MAX_COLOR_ATTACHMENTS] = {};

  PrimitiveTopology topology = PrimitiveTopology::TriangleList;
  CullMode cull = CullMode::None;
  FrontFace front_face = FrontFace::CounterClockwise;

  bool depth_test = false;
  bool depth_write = false;
  CompareOp depth_compare = CompareOp::LessOrEqual;

  u32 sample_count = 1;
};

struct ComputePipelineDesc {
  nx::string_view name = "pipeline";
  ShaderEntry compute;
  u32 group_size[3] = {0, 0, 0};
};

/// One indirect draw, as the GPU writes it. Layout is fixed by both APIs, so
/// a compute pass fills these directly and the CPU never sees the count.
struct DrawIndirectCommand {
  u32 vertex_count = 0;
  u32 instance_count = 0;
  u32 first_vertex = 0;
  u32 first_instance = 0;
};

struct DrawIndexedIndirectCommand {
  u32 index_count = 0;
  u32 instance_count = 0;
  u32 first_index = 0;
  i32 vertex_offset = 0;
  u32 first_instance = 0;
};

struct DispatchIndirectCommand {
  u32 groups_x = 0;
  u32 groups_y = 0;
  u32 groups_z = 0;
};

/// Barrier describing a texture's state transition. The render graph is the
/// intended producer of these; hand-written barriers are expert-mode.
struct TextureBarrier {
  TextureHandle texture;
  ResourceState from = ResourceState::Undefined;
  ResourceState to = ResourceState::Undefined;
  bool discard = false;
  /// UINT32_MAX means "all mips" / "all layers".
  u32 base_mip = 0;
  u32 mip_count = ~0u;
  u32 base_layer = 0;
  u32 layer_count = ~0u;
};

struct BufferBarrier {
  BufferHandle buffer;
  ResourceState from = ResourceState::Undefined;
  ResourceState to = ResourceState::Undefined;
  u64 offset = 0;
  u64 size = ~0ull; ///< ~0 means "to the end".
};

/// What the device can actually do. Queried once after creation; the engine
/// refuses to start if the mandatory bindless features are missing.
struct DeviceCaps {
  nx::string_view device_name;
  u32 api_version = 0;
  u32 driver_version = 0;
  u32 vendor_id = 0;
  bool discrete = false;

  /// Mandatory: the whole binding model depends on these.
  bool descriptor_indexing = false;
  bool buffer_device_address = false;
  bool dynamic_rendering = false;
  bool synchronization2 = false;
  bool timeline_semaphores = false;

  /// Optional.
  bool async_compute = false;
  bool dedicated_transfer = false;
  bool draw_indirect_count = false;
  /// Lazily-allocated / memoryless attachments; free transient targets on
  /// tile-based mobile GPUs.
  bool memoryless_attachments = false;
  bool timestamps = false;

  u32 max_texture_2d = 0;
  u32 max_bindless_sampled_images = 0;
  u32 max_bindless_storage_images = 0;
  u32 max_bindless_samplers = 0;
  u32 max_push_constant_size = 0;
  u64 device_local_memory = 0;
  f32 timestamp_period_ns = 0.f;
};

} // namespace nxe::rhi
