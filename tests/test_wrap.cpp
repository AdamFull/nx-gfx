#include "framework/nxtest.h"

#include "nx/gfx/gfx.h"
#include "nx/gfx/native.h"
#include "owned_device.h"

#if defined(NX_RHI_VULKAN)

#include "nx/gfx/vulkan.h"

#include <algorithm>
#include <atomic>

namespace gfx = nx::gfx;

namespace {

/// An image of the test's own, freed by the release callback as an owner
/// that may be gone by then would free it.
struct Foreign {
  VkDevice device = VK_NULL_HANDLE;
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  std::atomic<u32> released{0};
  VkImage released_image = VK_NULL_HANDLE;

  static void release(void *const user, const VkImage image) noexcept {
    Foreign *const self = static_cast<Foreign *>(user);
    self->released_image = image;
    vkDestroyImage(self->device, self->image, nullptr);
    vkFreeMemory(self->device, self->memory, nullptr);
    self->released.fetch_add(1, std::memory_order_release);
  }
};

[[nodiscard]] bool make_foreign(gfx::device &device, Foreign &out) {
  out.device = gfx::vk::logical_device(device);
  const std::span<const u32> families = gfx::vk::queue_families(device);
  VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ci.imageType = VK_IMAGE_TYPE_2D;
  ci.format = VK_FORMAT_R8G8B8A8_UNORM;
  ci.extent = {4, 4, 1};
  ci.mipLevels = 1;
  ci.arrayLayers = 1;
  ci.samples = VK_SAMPLE_COUNT_1_BIT;
  ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
             VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  ci.sharingMode = families.size() > 1 ? VK_SHARING_MODE_CONCURRENT
                                       : VK_SHARING_MODE_EXCLUSIVE;
  ci.queueFamilyIndexCount = nx::cast<u32>(families.size());
  ci.pQueueFamilyIndices = families.data();
  if (vkCreateImage(out.device, &ci, nullptr, &out.image) != VK_SUCCESS)
    return false;

  VkMemoryRequirements mr;
  vkGetImageMemoryRequirements(out.device, out.image, &mr);
  VkPhysicalDeviceMemoryProperties props;
  vkGetPhysicalDeviceMemoryProperties(gfx::vk::physical_device(device), &props);
  u32 type = ~0u;
  for (u32 i = 0; i < props.memoryTypeCount && type == ~0u; ++i)
    if ((mr.memoryTypeBits & (1u << i)) &&
        (props.memoryTypes[i].propertyFlags &
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
      type = i;
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = type;
  return type != ~0u &&
         vkAllocateMemory(out.device, &ai, nullptr, &out.memory) ==
             VK_SUCCESS &&
         vkBindImageMemory(out.device, out.image, out.memory, 0) == VK_SUCCESS;
}

[[nodiscard]] gfx::vk::wrapped_image wrapped(Foreign &foreign) {
  return {.name = "wrapped",
          .image = foreign.image,
          .format = gfx::format::RGBA8_UNORM,
          .width = 4,
          .height = 4,
          .usage = gfx::texture_usage::Sampled | gfx::texture_usage::CopySrc,
          .release = &Foreign::release,
          .user = &foreign};
}

[[nodiscard]] bool run_frame(gfx::device &device) {
  return gfx::begin_headless_frame(device) != nullptr &&
         gfx::end_headless_frame(device);
}

} // namespace

TEST_CASE("gfx wrap: the device's queue families are distinct and include "
          "the graphics family") {
  nx::gfx::test::OwnedDevice device;
  if (!device.init({.application_name = "nx wrap tests"}))
    SKIP("no usable device");
  const std::span<const u32> families = gfx::vk::queue_families(device);
  REQUIRE(!families.empty());
  CHECK(std::find(families.begin(), families.end(),
                  gfx::vk::graphics_family(device)) != families.end());
  for (usize i = 0; i < families.size(); ++i)
    for (usize j = i + 1; j < families.size(); ++j)
      CHECK(families[i] != families[j]);
}

TEST_CASE("gfx wrap: an adopted image is drawn, read back, and handed back "
          "once its retirement is over") {
  nx::gfx::test::OwnedDevice device;
  if (!device.init({.application_name = "nx wrap tests"}))
    SKIP("no usable device");
  Foreign foreign;
  REQUIRE(make_foreign(device, foreign));

  const gfx::texture_handle texture = gfx::vk::wrap(device, wrapped(foreign));
  REQUIRE(texture.valid());
  const gfx::native_texture native = gfx::native(device, texture);
  CHECK(native.image == foreign.image);
  CHECK(native.view != nullptr);

  gfx::command_list *cmd = gfx::begin_headless_frame(device);
  REQUIRE(cmd != nullptr);
  gfx::barrier(*cmd,
               gfx::texture_barrier{.texture = texture,
                                    .from = gfx::resource_state::Undefined,
                                    .to = gfx::resource_state::CopyDst});
  const auto buffer = static_cast<VkCommandBuffer>(gfx::begin_native(*cmd));
  REQUIRE(buffer != VK_NULL_HANDLE);
  const VkClearColorValue colour{{1.f, 0.f, 1.f, 1.f}};
  const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdClearColorImage(buffer, foreign.image,
                       static_cast<VkImageLayout>(gfx::native_layout(
                           device, gfx::resource_state::CopyDst)),
                       &colour, 1, &range);
  gfx::end_native(*cmd);
  gfx::barrier(*cmd, gfx::texture_barrier{.texture = texture,
                                          .from = gfx::resource_state::CopyDst,
                                          .to = gfx::resource_state::CopySrc});
  REQUIRE(gfx::end_headless_frame(device));
  gfx::wait_submitted(device);
  gfx::set_texture_state(device, texture, gfx::resource_state::CopySrc);

  const gfx::readback_result read = gfx::read_texture(device, texture);
  REQUIRE(read.data != nullptr);
  gfx::wait_upload(device, read.ticket);
  REQUIRE(read.size >= 4u);
  CHECK(read.data[0] == 255u);
  CHECK(read.data[1] == 0u);
  CHECK(read.data[2] == 255u);
  CHECK(read.data[3] == 255u);

  gfx::destroy_texture(device, texture);
  CHECK(foreign.released.load(std::memory_order_acquire) == 0u);
  for (u32 frame = 0;
       frame < 16 && foreign.released.load(std::memory_order_acquire) == 0;
       ++frame)
    REQUIRE(run_frame(device));
  CHECK(foreign.released.load(std::memory_order_acquire) == 1u);
  CHECK(foreign.released_image == native.image);
  REQUIRE(run_frame(device));
  CHECK(foreign.released.load(std::memory_order_acquire) == 1u);
}

TEST_CASE("gfx wrap: a device torn down with an adopted image hands it back") {
  nx::gfx::test::OwnedDevice device;
  if (!device.init({.application_name = "nx wrap tests"}))
    SKIP("no usable device");
  Foreign foreign;
  REQUIRE(make_foreign(device, foreign));
  REQUIRE(gfx::vk::wrap(device, wrapped(foreign)).valid());
  device.shutdown();
  CHECK(foreign.released.load(std::memory_order_acquire) == 1u);
}

TEST_CASE("gfx wrap: what cannot be adopted is refused and never handed back") {
  nx::gfx::test::OwnedDevice device;
  if (!device.init({.application_name = "nx wrap tests"}))
    SKIP("no usable device");
  Foreign foreign;
  REQUIRE(make_foreign(device, foreign));

  gfx::vk::wrapped_image uploadable = wrapped(foreign);
  uploadable.usage |= gfx::texture_usage::CopyDst;
  CHECK(!gfx::vk::wrap(device, uploadable).valid());
  gfx::vk::wrapped_image unsampled = wrapped(foreign);
  unsampled.usage = gfx::texture_usage::CopySrc;
  CHECK(!gfx::vk::wrap(device, unsampled).valid());
  gfx::vk::wrapped_image no_image = wrapped(foreign);
  no_image.image = VK_NULL_HANDLE;
  CHECK(!gfx::vk::wrap(device, no_image).valid());

  REQUIRE(run_frame(device));
  CHECK(foreign.released.load(std::memory_order_acquire) == 0u);
  Foreign::release(&foreign, foreign.image);
}

#endif
