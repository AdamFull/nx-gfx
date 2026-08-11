#include "core/rendering/rhi/types.h"

namespace nxe::rhi {
namespace {

struct FormatInfo {
  nx::string_view name;
  u32 texel_size; ///< Bytes per texel, or per block when compressed.
  u8 block_width;
  u8 block_height;
  bool depth;
  bool stencil;
  bool srgb;
};

// Indexed by Format. Kept in the same order as the enum; the static_assert
// below catches any drift.
constexpr FormatInfo FORMAT_TABLE[] = {
    {"Unknown", 0, 1, 1, false, false, false},

    {"R8_UNORM", 1, 1, 1, false, false, false},
    {"R8_SNORM", 1, 1, 1, false, false, false},
    {"R8_UINT", 1, 1, 1, false, false, false},
    {"R8_SINT", 1, 1, 1, false, false, false},
    {"RG8_UNORM", 2, 1, 1, false, false, false},
    {"RG8_UINT", 2, 1, 1, false, false, false},
    {"RGBA8_UNORM", 4, 1, 1, false, false, false},
    {"RGBA8_SRGB", 4, 1, 1, false, false, true},
    {"RGBA8_SNORM", 4, 1, 1, false, false, false},
    {"RGBA8_UINT", 4, 1, 1, false, false, false},
    {"BGRA8_UNORM", 4, 1, 1, false, false, false},
    {"BGRA8_SRGB", 4, 1, 1, false, false, true},

    {"R16_UNORM", 2, 1, 1, false, false, false},
    {"R16_FLOAT", 2, 1, 1, false, false, false},
    {"R16_UINT", 2, 1, 1, false, false, false},
    {"R16_SINT", 2, 1, 1, false, false, false},
    {"RG16_FLOAT", 4, 1, 1, false, false, false},
    {"RG16_UNORM", 4, 1, 1, false, false, false},
    {"RGBA16_FLOAT", 8, 1, 1, false, false, false},
    {"RGBA16_UNORM", 8, 1, 1, false, false, false},

    {"R32_FLOAT", 4, 1, 1, false, false, false},
    {"R32_UINT", 4, 1, 1, false, false, false},
    {"R32_SINT", 4, 1, 1, false, false, false},
    {"RG32_FLOAT", 8, 1, 1, false, false, false},
    {"RG32_UINT", 8, 1, 1, false, false, false},
    {"RGBA32_FLOAT", 16, 1, 1, false, false, false},
    {"RGBA32_UINT", 16, 1, 1, false, false, false},

    {"RGB10A2_UNORM", 4, 1, 1, false, false, false},
    {"RG11B10_FLOAT", 4, 1, 1, false, false, false},

    {"D16_UNORM", 2, 1, 1, true, false, false},
    {"D32_FLOAT", 4, 1, 1, true, false, false},
    {"D24_UNORM_S8_UINT", 4, 1, 1, true, true, false},
    {"D32_FLOAT_S8_UINT", 8, 1, 1, true, true, false},

    {"BC1_RGBA_UNORM", 8, 4, 4, false, false, false},
    {"BC1_RGBA_SRGB", 8, 4, 4, false, false, true},
    {"BC3_RGBA_UNORM", 16, 4, 4, false, false, false},
    {"BC3_RGBA_SRGB", 16, 4, 4, false, false, true},
    {"BC4_R_UNORM", 8, 4, 4, false, false, false},
    {"BC5_RG_UNORM", 16, 4, 4, false, false, false},
    {"BC7_RGBA_UNORM", 16, 4, 4, false, false, false},
    {"BC7_RGBA_SRGB", 16, 4, 4, false, false, true},

    {"ASTC_4x4_UNORM", 16, 4, 4, false, false, false},
    {"ASTC_4x4_SRGB", 16, 4, 4, false, false, true},
    {"ASTC_6x6_UNORM", 16, 6, 6, false, false, false},
    {"ASTC_6x6_SRGB", 16, 6, 6, false, false, true},
    {"ASTC_8x8_UNORM", 16, 8, 8, false, false, false},
    {"ASTC_8x8_SRGB", 16, 8, 8, false, false, true},

    {"BC1_RGB_UNORM", 8, 4, 4, false, false, false},
    {"BC1_RGB_SRGB", 8, 4, 4, false, false, true},
    {"BC2_RGBA_UNORM", 16, 4, 4, false, false, false},
    {"BC2_RGBA_SRGB", 16, 4, 4, false, false, true},
    {"BC6H_RGB_UFLOAT", 16, 4, 4, false, false, false},
    {"BC6H_RGB_SFLOAT", 16, 4, 4, false, false, false},
};
static_assert(std::size(FORMAT_TABLE) == nx::cast<usize>(Format::Count),
              "FORMAT_TABLE is out of sync with Format");

constexpr nx::string_view STATE_NAMES[] = {"Undefined",
                                           "General",
                                           "CopySrc",
                                           "CopyDst",
                                           "ShaderReadOnly",
                                           "ShaderWriteOnly",
                                           "ShaderReadWrite",
                                           "ColorAttachment",
                                           "ColorAttachmentRead",
                                           "ColorAttachmentWrite",
                                           "DepthStencilRead",
                                           "DepthStencilWrite",
                                           "DepthStencilReadWrite",
                                           "VertexBuffer",
                                           "IndexBuffer",
                                           "IndirectBuffer",
                                           "UniformBuffer",
                                           "HostRead",
                                           "HostWrite",
                                           "Present"};
static_assert(std::size(STATE_NAMES) == nx::cast<usize>(ResourceState::Count),
              "STATE_NAMES is out of sync with ResourceState");

constexpr nx::string_view QUEUE_NAMES[] = {"Graphics", "Compute", "Copy"};
static_assert(std::size(QUEUE_NAMES) == nx::cast<usize>(QueueType::Count),
              "QUEUE_NAMES is out of sync with QueueType");

const FormatInfo &info_of(const Format format) noexcept {
  const auto i = nx::cast<usize>(format);
  return FORMAT_TABLE[i < nx::cast<usize>(Format::Count) ? i : 0];
}

[[nodiscard]] bool same_ignoring_case(const nx::string_view a,
                                      const nx::string_view b) noexcept {
  if (a.size() != b.size())
    return false;
  for (usize i = 0; i < a.size(); ++i) {
    const char x = a[i] | ('a' - 'A');
    const char y = b[i] | ('a' - 'A');
    if (x != y)
      return false;
  }
  return true;
}

struct UsageName {
  nx::string_view name;
  u16 bit;
};

constexpr UsageName TEXTURE_USAGE_NAMES[] = {
    {"sampled", nx::cast<u16>(TextureUsage::Sampled)},
    {"storage", nx::cast<u16>(TextureUsage::Storage)},
    {"render_target", nx::cast<u16>(TextureUsage::RenderTarget)},
    {"depth_stencil", nx::cast<u16>(TextureUsage::DepthStencil)},
    {"copy_src", nx::cast<u16>(TextureUsage::CopySrc)},
    {"copy_dst", nx::cast<u16>(TextureUsage::CopyDst)},
    {"transient", nx::cast<u16>(TextureUsage::Transient)},
};

constexpr UsageName BUFFER_USAGE_NAMES[] = {
    {"vertex", nx::cast<u16>(BufferUsage::Vertex)},
    {"index", nx::cast<u16>(BufferUsage::Index)},
    {"uniform", nx::cast<u16>(BufferUsage::Uniform)},
    {"storage", nx::cast<u16>(BufferUsage::Storage)},
    {"indirect", nx::cast<u16>(BufferUsage::Indirect)},
    {"copy_src", nx::cast<u16>(BufferUsage::CopySrc)},
    {"copy_dst", nx::cast<u16>(BufferUsage::CopyDst)},
    {"device_address", nx::cast<u16>(BufferUsage::DeviceAddress)},
};

} // namespace

nx::string_view to_string(const Format format) noexcept {
  return info_of(format).name;
}

bool format_is_float(const Format format) noexcept {
  switch (format) {
  case Format::R16_FLOAT:
  case Format::RG16_FLOAT:
  case Format::RGBA16_FLOAT:
  case Format::R32_FLOAT:
  case Format::RG32_FLOAT:
  case Format::RGBA32_FLOAT:
  case Format::RG11B10_FLOAT:
    return true;
  default:
    return false;
  }
}

Format format_as_srgb(const Format format) noexcept {
  switch (format) {
  case Format::RGBA8_UNORM:
    return Format::RGBA8_SRGB;
  case Format::BGRA8_UNORM:
    return Format::BGRA8_SRGB;
  case Format::BC1_RGBA_UNORM:
    return Format::BC1_RGBA_SRGB;
  case Format::BC1_RGB_UNORM:
    return Format::BC1_RGB_SRGB;
  case Format::BC2_RGBA_UNORM:
    return Format::BC2_RGBA_SRGB;
  case Format::BC3_RGBA_UNORM:
    return Format::BC3_RGBA_SRGB;
  case Format::BC7_RGBA_UNORM:
    return Format::BC7_RGBA_SRGB;
  case Format::ASTC_4x4_UNORM:
    return Format::ASTC_4x4_SRGB;
  case Format::ASTC_6x6_UNORM:
    return Format::ASTC_6x6_SRGB;
  case Format::ASTC_8x8_UNORM:
    return Format::ASTC_8x8_SRGB;
  default:
    // Single-channel and two-channel formats have no sRGB twin, and would not
    // want one: they carry coverage, a mask or data, not colour.
    return format;
  }
}

Format format_from_string(const nx::string_view name) noexcept {
  for (usize i = 0; i < nx::cast<usize>(Format::Count); ++i)
    if (same_ignoring_case(FORMAT_TABLE[i].name, name))
      return nx::cast<Format>(i);
  return Format::Unknown;
}

TextureUsage texture_usage_from_string(const nx::string_view name) noexcept {
  for (const UsageName &one : TEXTURE_USAGE_NAMES)
    if (same_ignoring_case(one.name, name))
      return nx::cast<TextureUsage>(one.bit);
  return TextureUsage::None;
}

BufferUsage buffer_usage_from_string(const nx::string_view name) noexcept {
  for (const UsageName &one : BUFFER_USAGE_NAMES)
    if (same_ignoring_case(one.name, name))
      return nx::cast<BufferUsage>(one.bit);
  return BufferUsage::None;
}

nx::string_view to_string(const ResourceState state) noexcept {
  const auto i = nx::cast<usize>(state);
  return i < nx::cast<usize>(ResourceState::Count) ? STATE_NAMES[i] : "Invalid";
}

nx::string_view to_string(const QueueType queue) noexcept {
  const auto i = nx::cast<usize>(queue);
  return i < nx::cast<usize>(QueueType::Count) ? QUEUE_NAMES[i] : "Invalid";
}

u32 format_texel_size(const Format format) noexcept {
  return info_of(format).texel_size;
}

void format_block_extent(const Format format, u32 &width,
                         u32 &height) noexcept {
  const FormatInfo &info = info_of(format);
  width = info.block_width;
  height = info.block_height;
}

bool format_is_depth(const Format format) noexcept {
  return info_of(format).depth;
}

bool format_is_stencil(const Format format) noexcept {
  return info_of(format).stencil;
}

bool format_is_srgb(const Format format) noexcept {
  return info_of(format).srgb;
}

bool format_is_compressed(const Format format) noexcept {
  const FormatInfo &info = info_of(format);
  return info.block_width > 1 || info.block_height > 1;
}

u64 format_row_pitch(const Format format, const u32 width) noexcept {
  const FormatInfo &info = info_of(format);
  const u32 blocks_x = (width + info.block_width - 1) / info.block_width;
  return nx::cast<u64>(blocks_x) * info.texel_size;
}

u64 format_subresource_size(const Format format, const u32 width,
                            const u32 height, const u32 depth) noexcept {
  const FormatInfo &info = info_of(format);
  const u32 blocks_y = (height + info.block_height - 1) / info.block_height;
  return format_row_pitch(format, width) * blocks_y * (depth == 0 ? 1u : depth);
}

} // namespace nxe::rhi
