/**
 * @file mtl_upload.mm
 * @brief Placeholder for the Metal asynchronous uploader.
 *
 * NOT IMPLEMENTED. Every entry point fails cleanly - uploads report a null
 * ticket (which UploadTicket::valid() already treats as "nothing to wait for")
 * and loads report LoadStatus::Failed rather than hanging a caller that polls
 * for Ready. The signatures match VulkanUploader so that code above the RHI
 * compiles and runs against either backend; it simply draws nothing that needs
 * uploaded data.
 *
 * The real implementation lands with the resource pass: a staging ring in
 * MTLStorageModeShared memory, a blit encoder per batch, an MTLSharedEvent for
 * the ticket timeline, and the same decode worker thread the Vulkan uploader
 * uses.
 */

#include "rendering/rhi/metal/mtl_internal.h"

namespace nxe::rhi {
namespace {

/// Logged once per process rather than per call: the sample app queues a load
/// every run, and a warning per frame would bury everything else.
void warn_once(const char *what) noexcept {
  static bool warned = false;
  if (warned)
    return;
  warned = true;
  nx::logw(log_rhi, "uploader: {} is not implemented on Metal yet", what);
}

} // namespace

UploadTicket MetalUploader::upload_texture(TextureHandle,
                                           std::span<const u8>,
                                           std::span<const ImageSubresource>,
                                           ResourceState) {
  warn_once("upload_texture");
  return {};
}

UploadTicket MetalUploader::upload_texture(TextureHandle, const ImageData &,
                                           ResourceState) {
  warn_once("upload_texture");
  return {};
}

UploadTicket MetalUploader::upload_buffer(BufferHandle, u64,
                                          std::span<const u8>, ResourceState) {
  warn_once("upload_buffer");
  return {};
}

LoadRequest MetalUploader::load_texture(nx::string_view, TextureUsageFlags) {
  warn_once("load_texture");
  return {};
}

LoadRequest MetalUploader::load_texture_from_memory(std::span<const u8>,
                                                    TextureUsageFlags) {
  warn_once("load_texture");
  return {};
}

LoadStatus MetalUploader::load_status(const LoadRequest req) const noexcept {
  // An invalid request is "no such request"; anything else would have had to
  // come from one of the stubs above, which never hand out a valid id.
  return req.valid() ? LoadStatus::Failed : LoadStatus::None;
}

TextureHandle MetalUploader::loaded_texture(LoadRequest) const noexcept {
  return {};
}

void MetalUploader::release(LoadRequest) noexcept {}

bool MetalUploader::is_complete(UploadTicket) const noexcept {
  // Nothing was ever submitted, so nothing is outstanding.
  return true;
}

void MetalUploader::wait(UploadTicket) noexcept {}

void MetalUploader::flush() noexcept {}

} // namespace nxe::rhi
