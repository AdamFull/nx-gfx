#include "rendering/rhi/upload_ring.h"

#include "foundation/diagnostics/log.h"
#include "foundation/strings/format.h"

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
  if (handle_.valid() && capacity_ >= bytes)
    return true;
  if (handle_.valid())
    device.destroy_buffer(handle_);

  capacity_ = grow_capacity(bytes);
  desc.size = capacity_;
  handle_ = device.create_buffer(desc);
  if (!handle_.valid()) {
    nx::loge(log_rhi, "could not allocate '{}' ({} bytes)", desc.name,
             capacity_);
    capacity_ = 0;
    return false;
  }
  return true;
}

void GrowableBuffer::release(Device &device) noexcept {
  if (handle_.valid())
    device.destroy_buffer(handle_);
  handle_ = {};
  capacity_ = 0;
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

  device_ = device;
  arrays_ = nx::cast<u32>(arrays.size());
  frames_ = frames;
  frame_ = 0;
  buffers_.resize(nx::cast<usize>(arrays_) * frames_);
  names_.reserve(buffers_.size());
  for (u32 frame = 0; frame < frames_; ++frame)
    for (const nx::string_view array : arrays)
      names_.push_back(nx::format("{} {}", array, frame));
  return true;
}

void UploadRing::shutdown() noexcept {
  if (device_ != nullptr)
    for (GrowableBuffer &buffer : buffers_)
      buffer.release(*device_);
  buffers_.clear();
  names_.clear();
  device_ = nullptr;
  arrays_ = 0;
  frames_ = 0;
  frame_ = 0;
}

void UploadRing::next_frame() noexcept {
  if (frames_ != 0)
    frame_ = (frame_ + 1) % frames_;
}

GrowableBuffer *UploadRing::acquire(const u32 array, const u64 bytes) {
  if (device_ == nullptr || array >= arrays_)
    return nullptr;

  const usize slot = nx::cast<usize>(frame_) * arrays_ + array;
  GrowableBuffer &buffer = buffers_[slot];
  if (!buffer.ensure(
          *device_, bytes,
          {
              .name = names_[slot].view(),
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
    void *const mapped = device_->buffer_mapped(buffer->handle());
    if (mapped == nullptr)
      return 0;
    std::memcpy(mapped, data, nx::cast<usize>(bytes));
  }
  return device_->buffer_address(buffer->handle());
}

u64 UploadRing::reserve(const u32 array, const u64 bytes) {
  const GrowableBuffer *const buffer = acquire(array, bytes == 0 ? 1 : bytes);
  return buffer != nullptr ? device_->buffer_address(buffer->handle()) : 0;
}

} // namespace nxe::rhi
