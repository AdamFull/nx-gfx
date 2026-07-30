#pragma once

/**
 * @file upload_ring.h
 * @brief Device buffers that resize themselves (namespace nxe::rhi).
 */

#include "rendering/rhi/device.h"

#include "foundation/strings/utf8_string.h"

#include <span>

namespace nxe::rhi {

/// Rounds up to a power of two, with a floor that stops a buffer that is still
/// filling from being recreated on every addition.
[[nodiscard]] u64 grow_capacity(u64 needed) noexcept;

/// A device buffer that is recreated whenever it has to be bigger.
///
/// Not RAII: the device outlives every buffer taken from it and is what frees
/// them, so the owner calls release() from its own shutdown().
class GrowableBuffer {
public:
  /// Recreates the buffer if it is smaller than @p bytes. @p desc carries the
  /// usage, memory and mapping; its size is replaced by @p bytes rounded up.
  [[nodiscard]] bool ensure(Device &device, u64 bytes, BufferDesc desc);
  void release(Device &device) noexcept;

  [[nodiscard]] BufferHandle handle() const noexcept { return m_handle; }
  [[nodiscard]] u64 capacity() const noexcept { return m_capacity; }
  [[nodiscard]] bool valid() const noexcept { return m_handle.valid(); }

private:
  BufferHandle m_handle;
  u64 m_capacity = 0;
};

class UploadRing {
public:
  UploadRing() = default;
  ~UploadRing();

  UploadRing(const UploadRing &) = delete;
  UploadRing &operator=(const UploadRing &) = delete;

  /// @p arrays names one buffer set each, and is what indexes write() from
  /// then on. Names are the buffers' debug names, so they read as
  /// "scene instances 2" once the frame is appended.
  [[nodiscard]] bool init(Device *device,
                          std::span<const nx::string_view> arrays);
  void shutdown() noexcept;

  /// Moves to the next frame's buffers. Once per frame, before its writes.
  void next_frame() noexcept;

  /// Copies @p bytes of @p data into this frame's buffer for @p array and
  /// returns its device address, or 0 if it could not be grown or mapped.
  [[nodiscard]] u64 write(u32 array, const void *data, u64 bytes);

  /// A buffer of at least @p bytes whose contents are left alone. For an array
  /// the shader dereferences even on a frame that has nothing to put in it: a
  /// zero address would fault where an empty buffer just reads no elements.
  [[nodiscard]] u64 reserve(u32 array, u64 bytes);

  [[nodiscard]] bool valid() const noexcept { return m_device != nullptr; }
  [[nodiscard]] u32 frame_count() const noexcept { return m_frames; }

private:
  [[nodiscard]] GrowableBuffer *acquire(u32 array, u64 bytes);

  Device *m_device = nullptr;
  /// Frame-major: frame * m_arrays + array.
  nx::vector<GrowableBuffer> m_buffers;
  /// Built once at init. Formatting these per frame would be a heap allocation
  /// per array per frame to name a buffer that almost always already exists.
  nx::vector<nx::string> m_names;
  u32 m_arrays = 0;
  u32 m_frames = 0;
  u32 m_frame = 0;
};

} // namespace nxe::rhi
