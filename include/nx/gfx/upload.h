#pragma once

#include "nx/gfx/types.h"

#include <limits>
#include <span>

namespace nx::gfx {

struct upload_ticket {
  u64 value = 0;
  bool failed = false;

  [[nodiscard]] bool valid() const noexcept { return value != 0; }
  [[nodiscard]] bool ok() const noexcept { return !failed; }

  [[nodiscard]] static upload_ticket failure() noexcept { return {0, true}; }
};

/// Backends expose array layers through 32-bit descriptors, but current
/// desktop GPU APIs cap the usable count at 2048.
inline constexpr u32 MAX_IMAGE_ARRAY_LAYERS = 2048;

struct image_subresource {
  u64 offset = 0; ///< Byte offset into the upload's bytes.
  u64 size = 0;   ///< Byte length of this subresource.
  u32 mip = 0;
  u32 layer = 0;
  u32 width = 0;
  u32 height = 0;
  u32 depth = 1;
  u32 row_pitch = 0; ///< Bytes per block-row.

  u32 x = 0;
  u32 y = 0;
  u32 z = 0;
};

/// Texels for a texture upload, whatever container they were decoded from.
struct texture_data {
  gfx::format format = gfx::format::Unknown;
  std::span<const u8> bytes;
  std::span<const image_subresource> subresources;
};

/// Physical texture bounds used to validate a raw upload independently of a
/// graphics backend. The RHI accepts padded rows, but every described byte
/// range and destination block must be valid before a backend records work.
struct image_upload_desc {
  gfx::format format = gfx::format::Unknown;
  u32 width = 0;
  u32 height = 0;
  u32 depth = 1;
  u32 mip_levels = 1;
  u32 array_layers = 1;
};

[[nodiscard]] bool
validate_image_upload(const image_upload_desc &target, std::span<const u8> src,
                      std::span<const image_subresource> layout) noexcept;

namespace detail {

[[nodiscard]] inline bool checked_mul(const u64 a, const u64 b,
                                      u64 &result) noexcept {
  if (a != 0 && b > std::numeric_limits<u64>::max() / a)
    return false;
  result = a * b;
  return true;
}

} // namespace detail

/// Handle to an in-flight readback. `data` is valid once upload_complete()
/// (or after wait_upload()) and stays valid until a later read_* call reuses
/// the same staging - copy it out promptly.
struct readback_result {
  upload_ticket ticket;
  const u8 *data = nullptr;
  u64 size = 0;

  [[nodiscard]] std::span<const u8> bytes() const noexcept {
    return {data, nx::cast<usize>(size)};
  }
};

struct uploader_stats {
  u64 total_submits = 0; ///< Lifetime queue submissions (one per batch).
  u64 completed_submits = 0;
  u64 ring_used = 0; ///< Staging-ring bytes not yet reclaimed.
  u64 ring_capacity = 0;
  usize extra_buffers = 0;
  usize inflight_submits = 0;
};

// Any thread. Staging, the copy queue and queue-family transfers are the
// device's; a frame waits on what was uploaded before it began.
[[nodiscard]] upload_ticket
upload_texture(device &dev, texture_handle texture, std::span<const u8> src,
               std::span<const image_subresource> layout,
               resource_state final_state = resource_state::ShaderReadOnly);
[[nodiscard]] upload_ticket
upload_texture(device &dev, texture_handle texture, const texture_data &data,
               resource_state final_state = resource_state::ShaderReadOnly);
[[nodiscard]] upload_ticket
upload_buffer(device &dev, buffer_handle buffer, u64 dst_offset,
              std::span<const u8> src,
              resource_state final_state = resource_state::ShaderReadOnly);
[[nodiscard]] upload_ticket
clear_buffer(device &dev, buffer_handle buffer,
             resource_state final_state = resource_state::ShaderReadOnly);

// The source must not be in use on the graphics queue: read after
// wait_idle(), or from a resource only uploads touch.
[[nodiscard]] readback_result read_buffer(device &dev, buffer_handle buffer,
                                          u64 offset, u64 size);
[[nodiscard]] readback_result read_texture(device &dev, texture_handle texture,
                                           u32 mip = 0, u32 layer = 0);

[[nodiscard]] bool upload_complete(device &dev, upload_ticket ticket) noexcept;
void wait_upload(device &dev, upload_ticket ticket) noexcept;
/// Submits every queued upload and waits for all of them.
void wait_uploads(device &dev) noexcept;
[[nodiscard]] uploader_stats upload_stats(device &dev) noexcept;

} // namespace nx::gfx
