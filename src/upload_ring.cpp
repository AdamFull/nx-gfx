#include "core/rendering/rhi/upload_ring.h"

#include "core/foundation/diagnostics/log.h"
#include "core/foundation/strings/format.h"

#include <cstring>

namespace nxe::rhi {
namespace {

const nx::log::Category log_rhi = nx::log::category("rhi");

inline constexpr u64 CAPACITY_FLOOR = 4096;

} // namespace

u64 grow_capacity(const u64 needed) noexcept {
  u64 capacity = CAPACITY_FLOOR;
  while (capacity < needed)
    capacity *= 2;
  return capacity;
}

bool GrowableBuffer::ensure(Device &device, const u64 bytes, BufferDesc desc) {
  if (m_handle.valid() && m_capacity >= bytes)
    return true;
  if (m_handle.valid())
    device.destroy_buffer(m_handle);

  m_capacity = grow_capacity(bytes);
  desc.size = m_capacity;
  m_handle = device.create_buffer(desc);
  if (!m_handle.valid()) {
    nx::loge(log_rhi, "could not allocate '{}' ({} bytes)", desc.name,
             m_capacity);
    m_capacity = 0;
    return false;
  }
  return true;
}

void GrowableBuffer::release(Device &device) noexcept {
  if (m_handle.valid())
    device.destroy_buffer(m_handle);
  m_handle = {};
  m_capacity = 0;
}

UploadRing::~UploadRing() { shutdown(); }

bool UploadRing::init(Device *const device,
                      const std::span<const nx::string_view> arrays) {
  shutdown();
  if (device == nullptr || arrays.empty())
    return false;

  const u32 frames = device->frames_in_flight();
  if (frames == 0)
    return false;

  m_device = device;
  m_arrays = nx::cast<u32>(arrays.size());
  m_frames = frames;
  m_frame = 0;
  m_buffers.resize(nx::cast<usize>(m_arrays) * m_frames);
  m_names.reserve(m_buffers.size());
  for (u32 frame = 0; frame < m_frames; ++frame)
    for (const nx::string_view array : arrays)
      m_names.push_back(nx::format("{} {}", array, frame));
  return true;
}

void UploadRing::shutdown() noexcept {
  if (m_device != nullptr)
    for (GrowableBuffer &buffer : m_buffers)
      buffer.release(*m_device);
  m_buffers.clear();
  m_names.clear();
  m_device = nullptr;
  m_arrays = 0;
  m_frames = 0;
  m_frame = 0;
}

void UploadRing::next_frame() noexcept {
  if (m_frames != 0)
    m_frame = (m_frame + 1) % m_frames;
}

GrowableBuffer *UploadRing::acquire(const u32 array, const u64 bytes) {
  if (m_device == nullptr || array >= m_arrays)
    return nullptr;

  const usize slot = nx::cast<usize>(m_frame) * m_arrays + array;
  GrowableBuffer &buffer = m_buffers[slot];
  if (!buffer.ensure(
          *m_device, bytes,
          {
              .name = m_names[slot].view(),
              // DeviceAddress is what keeps these bindless: without
              // it every array would need a descriptor of its own.
              .usage = BufferUsage::Storage | BufferUsage::DeviceAddress,
              .memory = MemoryUsage::Upload,
              .persistently_mapped = true,
          }))
    return nullptr;
  return &buffer;
}

u64 UploadRing::write(const u32 array, const void *const data,
                      const u64 bytes) {
  GrowableBuffer *const buffer = acquire(array, bytes == 0 ? 1 : bytes);
  if (buffer == nullptr)
    return 0;

  if (bytes > 0) {
    void *const mapped = m_device->buffer_mapped(buffer->handle());
    if (mapped == nullptr)
      return 0;
    std::memcpy(mapped, data, nx::cast<usize>(bytes));
  }
  return m_device->buffer_address(buffer->handle());
}

u64 UploadRing::reserve(const u32 array, const u64 bytes) {
  const GrowableBuffer *const buffer = acquire(array, bytes == 0 ? 1 : bytes);
  return buffer != nullptr ? m_device->buffer_address(buffer->handle()) : 0;
}

} // namespace nxe::rhi
