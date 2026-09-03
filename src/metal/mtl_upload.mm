
#include "core/rendering/rhi/metal/mtl_internal.h"

#include "core/foundation/core/scope_guard.h"
#include "core/foundation/vfs/vfs.h"

#include <limits>

namespace nxe::rhi {
namespace {

using Staging = MetalUploadState::Staging;
using LoadJob = MetalUploadState::LoadJob;
using LoadSlot = MetalUploadState::LoadSlot;
using Source = MetalUploadState::Source;

u64 round_staging_size(const u64 bytes) noexcept {
  constexpr u64 MIN = 64u * 1024u;
  u64 size = MIN;
  while (size < bytes)
    size *= 2;
  return size;
}

Staging acquire_staging(MetalDeviceState &state, const u64 bytes,
                        const bool host_read = false) {
  MetalUploadState &up = state.upload;

  usize best = up.free_staging.size();
  for (usize i = 0; i < up.free_staging.size(); ++i) {
    if (up.free_staging[i].capacity < bytes ||
        up.free_staging[i].host_read != host_read)
      continue;
    if (best == up.free_staging.size() ||
        up.free_staging[i].capacity < up.free_staging[best].capacity)
      best = i;
  }

  if (best != up.free_staging.size()) {
    const Staging staging = up.free_staging[best];
    up.free_staging[best] = up.free_staging.back();
    up.free_staging.pop_back();
    return staging;
  }

  Staging staging;
  staging.capacity = round_staging_size(bytes);
  staging.host_read = host_read;
  @autoreleasepool {
    const MTLResourceOptions options =
        MTLResourceStorageModeShared |
        (host_read ? MTLResourceCPUCacheModeDefaultCache
                   : MTLResourceCPUCacheModeWriteCombined);

    staging.buffer = [state.device newBufferWithLength:staging.capacity
                                               options:options];
    if (staging.buffer == nil) {
      nx::loge(log_rhi, "uploader: could not allocate {} bytes of staging",
               staging.capacity);
      staging.capacity = 0;
      return staging;
    }
    staging.buffer.label = host_read ? @"nx2d readback" : @"nx2d staging";
  }
  return staging;
}

bool ensure_batch(MetalDeviceState &state) {
  MetalUploadState &up = state.upload;
  if (up.blit != nil)
    return true;
  if (state.queue == nil)
    return false;

  @autoreleasepool {
    up.batch = [state.queue commandBuffer];
    if (up.batch == nil)
      return false;
    up.batch.label = @"nx2d uploads";
    begin_command_buffer(up.batch_slot);

    up.blit = [up.batch blitCommandEncoder];
    if (up.blit == nil) {
      up.batch = nil;
      return false;
    }
    up.blit.label = @"nx2d uploads";
    encoder_wait_fence(state, up.batch_slot, up.blit);
  }
  return true;
}

u64 submit_batch(MetalDeviceState &state) {
  MetalUploadState &up = state.upload;
  if (up.blit == nil)
    return up.submitted;

  const u64 value = ++up.submitted;

  @autoreleasepool {
    encoder_update_fence(state, up.batch_slot, up.blit);
    [up.blit endEncoding];

    [up.batch encodeSignalEvent:state.upload_event value:value];

    std::atomic<u64> *completed = &up.completed;
    [up.batch addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
      if (buffer.error != nil) {
        nx::loge(log_rhi, "uploader: batch failed: {}",
                 [[buffer.error localizedDescription] UTF8String]);
      }
      // Batches complete in submission order on one queue, so a plain store of
      // the highest value is correct and monotonic.
      completed->store(value, std::memory_order_release);
    }];
    [up.batch commit];
  }

  MetalUploadState::InFlight in_flight;
  in_flight.value = value;
  in_flight.cmd = up.batch;
  in_flight.staging = std::move(up.batch_staging);
  up.in_flight.push_back(std::move(in_flight));

  up.batch_staging.clear();
  up.blit = nil;
  up.batch = nil;
  up.bytes_staged = 0;
  return value;
}

bool record_texture_copies(MetalDeviceState &state, const MetalTexture &texture,
                           const std::span<const u8> src,
                           const std::span<const ImageSubresource> layout) {
  MetalUploadState &up = state.upload;

  if (!validate_image_upload(
          {texture.format, texture.width, texture.height,
           nx::cast<u32>([texture.texture depth]), texture.mip_levels,
           texture.layers},
          src, layout)) {
    nx::loge(log_rhi, "upload_texture: malformed upload layout");
    return false;
  }

  u64 total = 0;
  for (const ImageSubresource &sub : layout) {
    if (sub.size > std::numeric_limits<u64>::max() - total)
      return false;
    total += sub.size;
  }
  if (total == 0)
    return false;

  Staging staging = acquire_staging(state, total);
  if (staging.buffer == nil)
    return false;

  auto *dst = static_cast<u8 *>([staging.buffer contents]);
  u64 cursor = 0;

  u32 block_w = 1, block_h = 1;
  format_block_extent(texture.format, block_w, block_h);

  for (const ImageSubresource &sub : layout) {
    std::memcpy(dst + cursor, src.data() + sub.offset,
                nx::cast<usize>(sub.size));

    // bytesPerImage must be 0 for a 2D copy, and the full slice size for 3D.
    // Passing a non-zero value for 2D trips Metal's validation.
    const u64 rows = (nx::cast<u64>(sub.height) + block_h - 1) / block_h;
    const u64 bytes_per_image = sub.depth > 1 ? sub.row_pitch * rows : 0;

    [up.blit copyFromBuffer:staging.buffer
               sourceOffset:cursor
          sourceBytesPerRow:sub.row_pitch
        sourceBytesPerImage:bytes_per_image
                 sourceSize:MTLSizeMake(sub.width, sub.height,
                                        nx::max<u32>(1, sub.depth))
                  toTexture:texture.texture
           destinationSlice:sub.layer
           destinationLevel:sub.mip
          destinationOrigin:MTLOriginMake(sub.x, sub.y, sub.z)];

    cursor += sub.size;
  }

  up.bytes_staged += total;
  up.batch_staging.push_back(staging);
  return true;
}

LoadSlot *acquire_load_slot(MetalUploadState &up, u32 &out_id,
                            u32 &out_generation) {
  nx::scoped_lock lock(up.load_mutex);
  for (usize i = 1; i < MetalUploadState::MAX_LOADS; ++i) {
    LoadSlot &slot = up.loads[i];
    if (slot.in_use)
      continue;
    slot.in_use = true;
    ++slot.generation;
    slot.texture = {};
    slot.decoded = {};
    slot.status.store(LoadStatus::Pending, std::memory_order_release);
    out_id = nx::cast<u32>(i);
    out_generation = slot.generation;
    return &slot;
  }
  return nullptr;
}

LoadSlot *resolve_load(MetalUploadState &up, const LoadRequest req) noexcept {
  if (req.id == 0 || req.id >= MetalUploadState::MAX_LOADS)
    return nullptr;
  LoadSlot &slot = up.loads[req.id];
  return slot.generation == req.generation ? &slot : nullptr;
}

void worker_main(MetalDeviceState *state) {
  MetalUploadState &up = state->upload;

  while (up.worker_running.load(std::memory_order_acquire)) {
    LoadJob job;
    if (!up.job_queue.try_pop(job)) {
      up.job_signal.wait();
      continue;
    }

    LoadSlot &slot = up.loads[job.id];
    if (slot.generation != job.generation)
      continue;

    slot.status.store(LoadStatus::Decoding, std::memory_order_release);

    nx::result<ImageData, ImageError> decoded =
        nx::result<ImageData, ImageError>(nx::err(ImageError::Unknown));

    if (job.source == Source::Path) {
      if (auto bytes = nx::vfs::read(slot.path.view())) {
        decoded = load_image_from_memory(
            std::span<const u8>(bytes->data(), bytes->size()));
      } else {
        decoded = nx::result<ImageData, ImageError>(nx::err(ImageError::Io));
      }
    } else {
      decoded = load_image_from_memory(
          std::span<const u8>(slot.encoded.data(), slot.encoded.size()));
    }

    if (!decoded) {
      nx::logw(log_rhi, "uploader: decode failed for request {}: {}", job.id,
               to_string(decoded.error()));
      slot.status.store(LoadStatus::Failed, std::memory_order_release);
      continue;
    }

    slot.decoded = std::move(*decoded);
    // The encoded bytes are dead once decoded; drop them before the render
    // thread picks the job up so the peak footprint is one copy, not two.
    slot.encoded = {};
    slot.status.store(LoadStatus::Uploading, std::memory_order_release);
    (void)up.done_queue.try_push(job);
  }
}

}

bool uploader_init(MetalDeviceState &state) {
  MetalUploadState &up = state.upload;

  up.worker_running.store(true, std::memory_order_release);
  up.worker = nx::thread(
      [&state] { worker_main(&state); },
      nx::thread_options{.name = "nx-upload",
                         .priority = nx::thread_priority::BelowNormal});

  if (!up.worker.joinable()) {
    nx::loge(log_rhi, "uploader: could not start the decode worker");
    up.worker_running.store(false, std::memory_order_release);
    return false;
  }
  return true;
}

void uploader_shutdown(MetalDeviceState &state) noexcept {
  MetalUploadState &up = state.upload;

  if (up.worker.joinable()) {
    up.worker_running.store(false, std::memory_order_release);
    up.job_signal.set();
    up.worker.join();
  }

  if (up.blit != nil)
    (void)submit_batch(state);

  @autoreleasepool {
    for (MetalUploadState::InFlight &batch : up.in_flight)
      batch.staging.clear();
    up.in_flight.clear();
    up.batch_staging.clear();
    up.free_staging.clear();
    up.blit = nil;
    up.batch = nil;
  }

  for (LoadSlot &slot : up.loads) {
    slot.status.store(LoadStatus::None, std::memory_order_release);
    slot.in_use = false;
    slot.decoded = {};
    slot.encoded = {};
    slot.path.clear();
  }
}

void uploader_reclaim(MetalDeviceState &state) noexcept {
  MetalUploadState &up = state.upload;
  const u64 completed = up.completed.load(std::memory_order_acquire);

  usize kept = 0;
  for (usize i = 0; i < up.in_flight.size(); ++i) {
    if (up.in_flight[i].value <= completed) {
      for (const Staging &staging : up.in_flight[i].staging)
        up.free_staging.push_back(staging);
      continue;
    }
    if (kept != i)
      up.in_flight[kept] = std::move(up.in_flight[i]);
    ++kept;
  }
  while (up.in_flight.size() > kept)
    up.in_flight.pop_back();
}

void uploader_pump(MetalDeviceState &state) {
  MetalUploadState &up = state.upload;
  MetalDevice *device = state.owner;
  if (device == nullptr)
    return;

  LoadJob job;
  while (up.done_queue.try_pop(job)) {
    LoadSlot &slot = up.loads[job.id];
    if (slot.generation != job.generation)
      continue;

    if (!slot.decoded.valid()) {
      slot.status.store(LoadStatus::Failed, std::memory_order_release);
      ++up.loads_failed;
      continue;
    }

    const TextureHandle handle =
        device->create_texture(slot.decoded.texture_desc("loaded", job.usage));
    if (!handle.valid()) {
      slot.status.store(LoadStatus::Failed, std::memory_order_release);
      slot.decoded = {};
      ++up.loads_failed;
      continue;
    }

    const bool uploaded =
        device->uploader().upload_texture(handle, slot.decoded).ok();
    slot.decoded = {};
    // Without this the load reaches Ready and hands out a texture holding
    // whatever was in that memory, which is worse than reporting the failure.
    if (!uploaded) {
      device->destroy_texture(handle);
      slot.status.store(LoadStatus::Failed, std::memory_order_release);
      ++up.loads_failed;
      continue;
    }

    slot.texture = handle;
    slot.status.store(LoadStatus::Ready, std::memory_order_release);
    ++up.loads_ready;
  }

  if (up.blit != nil)
    (void)submit_batch(state);
}

UploadTicket MetalUploader::upload_texture(const TextureHandle handle,
                                           const std::span<const u8> src,
                                           const std::span<const ImageSubresource> layout,
                                           const ResourceState final_state) {
  if (m_state == nullptr || layout.empty())
    return UploadTicket::failure();

  MetalTexture *texture = m_state->textures.try_get(handle);
  if (texture == nullptr || texture->texture == nil ||
      !(texture->usage & TextureUsage::CopyDst)) {
    nx::loge(log_rhi, "upload_texture: handle is not live");
    return UploadTicket::failure();
  }

  if (!ensure_batch(*m_state))
    return UploadTicket::failure();

  if (!record_texture_copies(*m_state, *texture, src, layout))
    return UploadTicket::failure();

  texture->state = final_state;

  return UploadTicket{m_state->upload.submitted + 1};
}

UploadTicket MetalUploader::upload_texture(const TextureHandle handle,
                                           const ImageData &image,
                                           const ResourceState final_state) {
  const MetalTexture *const texture =
      m_state != nullptr ? m_state->textures.try_get(handle) : nullptr;
  if (texture == nullptr || image.format != texture->format) {
    nx::loge(log_rhi, "upload_texture: image format does not match texture");
    return UploadTicket::failure();
  }
  return upload_texture(
      handle, std::span<const u8>(image.pixels.data(), image.pixels.size()),
      std::span<const ImageSubresource>(image.subresources.data(),
                                        image.subresources.size()),
      final_state);
}

UploadTicket MetalUploader::upload_buffer(const BufferHandle handle,
                                          const u64 dst_offset,
                                          const std::span<const u8> src,
                                          const ResourceState final_state) {
  (void)final_state;

  if (m_state == nullptr || src.empty())
    return UploadTicket::failure();

  const MetalBuffer *buffer = m_state->buffers.try_get(handle);
  if (buffer == nullptr || buffer->buffer == nil) {
    nx::loge(log_rhi, "upload_buffer: handle is not live");
    return UploadTicket::failure();
  }
  if (dst_offset > buffer->size || src.size() > buffer->size - dst_offset) {
    nx::loge(log_rhi, "upload_buffer: {} bytes at offset {} overruns a {} byte "
                      "buffer",
             src.size(), dst_offset, buffer->size);
    return UploadTicket::failure();
  }

  if (!ensure_batch(*m_state))
    return UploadTicket::failure();

  MetalUploadState &up = m_state->upload;
  Staging staging = acquire_staging(*m_state, src.size());
  if (staging.buffer == nil)
    return UploadTicket::failure();

  std::memcpy([staging.buffer contents], src.data(), src.size());
  [up.blit copyFromBuffer:staging.buffer
             sourceOffset:0
                 toBuffer:buffer->buffer
        destinationOffset:dst_offset
                     size:src.size()];

  up.bytes_staged += src.size();
  up.batch_staging.push_back(staging);
  return UploadTicket{up.submitted + 1};
}

UploadTicket MetalUploader::clear_buffer(const BufferHandle handle,
                                         const ResourceState final_state) {
  (void)final_state;

  if (m_state == nullptr)
    return UploadTicket::failure();

  const MetalBuffer *buffer = m_state->buffers.try_get(handle);
  if (buffer == nullptr || buffer->buffer == nil) {
    nx::loge(log_rhi, "clear_buffer: handle is not live");
    return UploadTicket::failure();
  }
  if (!(buffer->usage & BufferUsage::CopyDst)) {
    nx::loge(log_rhi, "clear_buffer: buffer has no copy-destination usage");
    return UploadTicket::failure();
  }
  if (!ensure_batch(*m_state))
    return UploadTicket::failure();

  MetalUploadState &up = m_state->upload;
  [up.blit fillBuffer:buffer->buffer
                range:NSMakeRange(0, nx::cast<NSUInteger>(buffer->size))
                value:0];
  return UploadTicket{up.submitted + 1};
}

LoadRequest MetalUploader::load_texture(const nx::string_view path,
                                        const TextureUsageFlags usage) {
  if (m_state == nullptr)
    return {};

  MetalUploadState &up = m_state->upload;
  u32 id = 0, generation = 0;
  LoadSlot *slot = acquire_load_slot(up, id, generation);
  if (slot == nullptr) {
    nx::logw(log_rhi, "uploader: the load table is full ({} slots)",
             MetalUploadState::MAX_LOADS);
    return {};
  }

  slot->path = nx::string(path);
  slot->usage = usage;

  const LoadJob job{id, generation, Source::Path, usage};
  if (!up.job_queue.try_push(job)) {
    slot->status.store(LoadStatus::Failed, std::memory_order_release);
    return {};
  }
  up.job_signal.set();
  return LoadRequest{id, generation};
}

LoadRequest
MetalUploader::load_texture_from_memory(const std::span<const u8> bytes,
                                        const TextureUsageFlags usage) {
  if (m_state == nullptr || bytes.empty())
    return {};

  MetalUploadState &up = m_state->upload;
  u32 id = 0, generation = 0;
  LoadSlot *slot = acquire_load_slot(up, id, generation);
  if (slot == nullptr) {
    nx::logw(log_rhi, "uploader: the load table is full ({} slots)",
             MetalUploadState::MAX_LOADS);
    return {};
  }

  slot->encoded = nx::blob<u8>(bytes.size());
  std::memcpy(slot->encoded.data(), bytes.data(), bytes.size());
  slot->usage = usage;

  const LoadJob job{id, generation, Source::Memory, usage};
  if (!up.job_queue.try_push(job)) {
    slot->status.store(LoadStatus::Failed, std::memory_order_release);
    return {};
  }
  up.job_signal.set();
  return LoadRequest{id, generation};
}

LoadStatus MetalUploader::load_status(const LoadRequest req) const noexcept {
  if (m_state == nullptr)
    return LoadStatus::None;
  LoadSlot *slot = resolve_load(m_state->upload, req);
  if (slot == nullptr)
    return LoadStatus::None;
  return slot->status.load(std::memory_order_acquire);
}

TextureHandle MetalUploader::loaded_texture(const LoadRequest req) const noexcept {
  if (m_state == nullptr)
    return {};
  LoadSlot *slot = resolve_load(m_state->upload, req);
  if (slot == nullptr ||
      slot->status.load(std::memory_order_acquire) != LoadStatus::Ready)
    return {};
  return slot->texture;
}

void MetalUploader::release(const LoadRequest req) noexcept {
  if (m_state == nullptr)
    return;

  MetalUploadState &up = m_state->upload;
  nx::scoped_lock lock(up.load_mutex);
  LoadSlot *slot = resolve_load(up, req);
  if (slot == nullptr)
    return;

  // The contract is to release only once the status has settled, so refuse rather than corrupt: the
  // slot stays claimed and the caller can release it again after polling.
  switch (slot->status.load(std::memory_order_acquire)) {
  case LoadStatus::Pending:
  case LoadStatus::Decoding:
  case LoadStatus::Uploading:
    nx::loge(log_rhi, "uploader: release() on request {} while it is still in "
                      "flight; release only once it is Ready or Failed",
             req.id);
    return;
  default:
    break;
  }

  ++slot->generation;
  slot->in_use = false;
  slot->status.store(LoadStatus::None, std::memory_order_release);
  slot->texture = {};
  slot->decoded = {};
  slot->encoded = {};
  slot->path.clear();
}

bool MetalUploader::is_complete(const UploadTicket ticket) const noexcept {
  if (m_state == nullptr || !ticket.valid())
    return true;
  return m_state->upload.completed.load(std::memory_order_acquire) >=
         ticket.value;
}

ReadbackResult MetalUploader::read_buffer(const BufferHandle handle,
                                          const u64 offset, const u64 size) {
  if (m_state == nullptr || size == 0)
    return {};

  const MetalBuffer *buffer = m_state->buffers.try_get(handle);
  if (buffer == nullptr || buffer->buffer == nil) {
    nx::loge(log_rhi, "read_buffer: handle is not live");
    return {};
  }
  if (!(buffer->usage & BufferUsage::CopySrc)) {
    nx::loge(log_rhi, "read_buffer: buffer was not created with CopySrc");
    return {};
  }
  if (offset > buffer->size || size > buffer->size - offset) {
    nx::loge(log_rhi, "read_buffer: {} bytes at offset {} overruns a {} byte "
                      "buffer",
             size, offset, buffer->size);
    return {};
  }

  MetalUploadState &up = m_state->upload;
  Staging staging = acquire_staging(*m_state, size, /*host_read=*/true);
  if (staging.buffer == nil)
    return {};

  if (!ensure_batch(*m_state)) {
    up.free_staging.push_back(staging);
    return {};
  }

  [up.blit copyFromBuffer:buffer->buffer
             sourceOffset:offset
                 toBuffer:staging.buffer
        destinationOffset:0
                     size:size];

  up.batch_staging.push_back(staging);

  ReadbackResult result;
  result.data = static_cast<const u8 *>([staging.buffer contents]);
  result.size = size;
  // Readbacks submit on their own: a caller that polls the ticket would
  // otherwise wait on a batch nothing ever commits.
  result.ticket = UploadTicket{submit_batch(*m_state)};
  return result;
}

ReadbackResult MetalUploader::read_texture(const TextureHandle handle,
                                           const u32 mip, const u32 layer) {
  if (m_state == nullptr)
    return {};

  const MetalTexture *texture = m_state->textures.try_get(handle);
  if (texture == nullptr || texture->texture == nil) {
    nx::loge(log_rhi, "read_texture: handle is not live");
    return {};
  }
  if (mip >= texture->mip_levels || layer >= texture->layers) {
    nx::loge(log_rhi, "read_texture: mip {} layer {} is out of range for a "
                      "{} mip / {} layer texture",
             mip, layer, texture->mip_levels, texture->layers);
    return {};
  }

  const u32 width = nx::max<u32>(1, texture->width >> mip);
  const u32 height = nx::max<u32>(1, texture->height >> mip);
  const u64 row_pitch = format_row_pitch(texture->format, width);
  const u64 size = format_subresource_size(texture->format, width, height);
  if (size == 0)
    return {};

  MetalUploadState &up = m_state->upload;
  Staging staging = acquire_staging(*m_state, size, /*host_read=*/true);
  if (staging.buffer == nil)
    return {};

  if (!ensure_batch(*m_state)) {
    up.free_staging.push_back(staging);
    return {};
  }

  u32 block_w = 1, block_h = 1;
  format_block_extent(texture->format, block_w, block_h);
  const u64 rows = (nx::cast<u64>(height) + block_h - 1) / block_h;

  [up.blit copyFromTexture:texture->texture
                     sourceSlice:layer
                     sourceLevel:mip
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(width, height, 1)
                        toBuffer:staging.buffer
               destinationOffset:0
          destinationBytesPerRow:row_pitch
        destinationBytesPerImage:row_pitch * rows];

  up.batch_staging.push_back(staging);

  ReadbackResult result;
  result.data = static_cast<const u8 *>([staging.buffer contents]);
  result.size = size;
  result.ticket = UploadTicket{submit_batch(*m_state)};
  return result;
}

void MetalUploader::pump() {
  if (m_state != nullptr)
    uploader_pump(*m_state);
}

UploaderStats MetalUploader::stats() const noexcept {
  if (m_state == nullptr)
    return {};

  const MetalUploadState &up = m_state->upload;
  UploaderStats out;
  out.total_submits = up.submitted;
  out.completed_submits = up.completed.load(std::memory_order_acquire);
  out.ring_used = up.bytes_staged;
  for (const Staging &staging : up.free_staging)
    out.ring_capacity += staging.capacity;
  for (const Staging &staging : up.batch_staging)
    out.ring_capacity += staging.capacity;
  out.extra_buffers = up.free_staging.size();
  out.inflight_submits = up.in_flight.size();
  out.loads_ready = up.loads_ready;
  out.loads_failed = up.loads_failed;
  return out;
}

namespace {

void wait_for_value(MetalDeviceState &state, const u64 value) noexcept {
  MetalUploadState &up = state.upload;
  if (up.completed.load(std::memory_order_acquire) >= value)
    return;

  @autoreleasepool {
    for (const MetalUploadState::InFlight &batch : up.in_flight) {
      if (batch.value <= value && batch.cmd != nil)
        [batch.cmd waitUntilCompleted];
    }
  }

  while (up.completed.load(std::memory_order_acquire) < value)
    nx::this_thread::yield();

  uploader_reclaim(state);
}

}

void MetalUploader::wait(const UploadTicket ticket) noexcept {
  if (m_state == nullptr || !ticket.valid() || is_complete(ticket))
    return;

  MetalUploadState &up = m_state->upload;
  if (ticket.value > up.submitted && up.blit != nil)
    (void)submit_batch(*m_state);

  wait_for_value(*m_state, ticket.value);
}

void MetalUploader::flush() noexcept {
  if (m_state == nullptr)
    return;

  MetalUploadState &up = m_state->upload;
  const u64 value = up.blit != nil ? submit_batch(*m_state) : up.submitted;
  if (value == 0)
    return;

  wait_for_value(*m_state, value);
}

}
