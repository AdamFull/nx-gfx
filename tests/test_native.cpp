#include "framework/nxtest.h"
#include "pipeline_test_utils.h"

#include "core/foundation/diagnostics/log.h"
#include "core/foundation/platform/filesystem.h"
#include "nx/gfx/gfx.h"
#include "nx/gfx/native.h"
#include "owned_device.h"

#if defined(NX_RHI_VULKAN)

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <atomic>
#include <cstring>

namespace gfx = nx::gfx;

namespace {

/// The texel the read shader writes out: a float4.
struct Rgba {
  float r, g, b, a;
};

constexpr const char *INSTANCE_EXTENSION = "VK_KHR_get_surface_capabilities2";
constexpr const char *DEVICE_EXTENSION = "VK_EXT_memory_budget";
constexpr const char *MISSING = "VK_NX_not_an_extension";

/// Asks for one extension a loader or device commonly has and one none has,
/// at each level; optionally a core feature the adapter lacks.
class Asker final : public gfx::native_device_client {
public:
  bool impossible = false;
  bool asked_instance = false;
  bool saw_adapter = false;
  bool lacking = false;
  bool instance_available = false;
  bool device_available = false;

  void instance_extensions(const gfx::native_api api,
                           gfx::native_names &out) override {
    asked_instance = api == gfx::native_api::Vulkan;
    out.emplace_back(INSTANCE_EXTENSION);
    out.emplace_back(MISSING);
  }

  void device_extensions(const gfx::native_device &adapter,
                         gfx::native_names &out) override {
    saw_adapter = asked_instance && adapter.instance != nullptr &&
                  adapter.adapter != nullptr && adapter.device == nullptr &&
                  adapter.get_instance_proc_addr != nullptr;
    const auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        adapter.get_instance_proc_addr);
    const auto instance = static_cast<VkInstance>(adapter.instance);
    const auto physical = static_cast<VkPhysicalDevice>(adapter.adapter);
    instance_available = listed(
        [&](u32 *count, VkExtensionProperties *into) {
          return reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
              gipa(nullptr, "vkEnumerateInstanceExtensionProperties"))(
              nullptr, count, into);
        },
        INSTANCE_EXTENSION);
    device_available = listed(
        [&](u32 *count, VkExtensionProperties *into) {
          return reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
              gipa(instance, "vkEnumerateDeviceExtensionProperties"))(
              physical, nullptr, count, into);
        },
        DEVICE_EXTENSION);
    out.emplace_back(DEVICE_EXTENSION);
    out.emplace_back(MISSING);
  }

  void device_features(const gfx::native_device &adapter,
                       void *const features) override {
    if (!impossible)
      return;
    const auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        adapter.get_instance_proc_addr);
    const auto get = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures>(
        gipa(static_cast<VkInstance>(adapter.instance),
             "vkGetPhysicalDeviceFeatures"));
    VkPhysicalDeviceFeatures has{};
    get(static_cast<VkPhysicalDevice>(adapter.adapter), &has);
    const auto *const bits = reinterpret_cast<const VkBool32 *>(&has);
    auto *const wants = reinterpret_cast<VkBool32 *>(
        &static_cast<VkPhysicalDeviceFeatures2 *>(features)->features);
    for (usize i = 0; i < sizeof(has) / sizeof(VkBool32) && !lacking; ++i)
      if (bits[i] == VK_FALSE) {
        wants[i] = VK_TRUE;
        lacking = true;
      }
  }

private:
  template <typename Enumerate>
  [[nodiscard]] static bool listed(Enumerate enumerate, const char *name) {
    u32 count = 0;
    if (enumerate(&count, nullptr) != VK_SUCCESS)
      return false;
    nx::vector<VkExtensionProperties> all(count);
    if (enumerate(&count, all.data()) != VK_SUCCESS)
      return false;
    for (const VkExtensionProperties &one : all)
      if (std::strcmp(one.extensionName, name) == 0)
        return true;
    return false;
  }
};

struct TestDevice {
  nx::gfx::test::OwnedDevice device;
  bool ready = false;

  explicit TestDevice(gfx::native_device_client *const client = nullptr) {
    gfx::device_desc desc{};
    desc.application_name = "nx native tests";
    gfx::native_device_client *const clients[] = {client};
    if (client != nullptr)
      desc.native_clients = clients;
    ready = device.init(desc);
  }
  ~TestDevice() {
    if (ready)
      device.shutdown();
  }
  TestDevice(const TestDevice &) = delete;
  TestDevice &operator=(const TestDevice &) = delete;
};

class ValidationErrors final : public nx::log::Sink {
public:
  void write(const nx::log::Record &r) override {
    const nx::string_view message(r.message, r.length);
    if (message.find("VUID") != nx::string_view::npos)
      m_count.fetch_add(1, std::memory_order_relaxed);
  }
  [[nodiscard]] u32 count() const noexcept {
    return m_count.load(std::memory_order_relaxed);
  }

private:
  std::atomic<u32> m_count{0};
};

struct Push {
  u64 out = 0;
  u32 texture = 0;
};

} // namespace

TEST_CASE("rhi native: a client's extensions are enabled where the device has "
          "them, and the rest left out") {
  Asker asker;
  TestDevice fixture(&asker);
  if (!fixture.ready)
    SKIP("no usable RHI device");
  const gfx::device &device = fixture.device;
  CHECK(asker.asked_instance);
  CHECK(asker.saw_adapter);
  CHECK(gfx::native_extension(device, INSTANCE_EXTENSION) ==
        asker.instance_available);
  CHECK(gfx::native_extension(device, DEVICE_EXTENSION) ==
        asker.device_available);
  CHECK(!gfx::native_extension(device, MISSING));
  CHECK(gfx::native_extension(device, VK_KHR_SWAPCHAIN_EXTENSION_NAME));
}

TEST_CASE("rhi native: a device its client asks the impossible of is made "
          "without what the client asked for") {
  Asker asker;
  asker.impossible = true;
  TestDevice fixture(&asker);
  if (!asker.lacking)
    SKIP("the adapter has every core feature");
  REQUIRE(fixture.ready);
  CHECK(!gfx::native_extension(fixture.device, DEVICE_EXTENSION));
  CHECK(gfx::native_extension(fixture.device, VK_KHR_SWAPCHAIN_EXTENSION_NAME));
}

TEST_CASE("rhi native: commands recorded into the device's own command "
          "buffer land, and its own work after them still finds its textures") {
  TestDevice fixture;
  if (!fixture.ready)
    SKIP("no usable RHI device");
  gfx::device &device = fixture.device;
  const nx::string base = nx::string(NX_GFX_TEST_SHADER_DIR) + "/native_read";
  auto code = nx::fs::file_read(nx::fs::path_view(base + ".spv"));
  auto refl = nx::fs::file_read_text(nx::fs::path_view(base + ".refl.json"));
  if (!code.has_value() || !refl.has_value())
    SKIP("test shaders are not built in this configuration");
  REQUIRE(gfx::BINDLESS_SET == 0);

  const gfx::native_device native = gfx::native(device);
  REQUIRE(native.api == gfx::native_api::Vulkan);
  REQUIRE(native.device != nullptr);
  REQUIRE(native.queue != nullptr);
  const auto vk_device = static_cast<VkDevice>(native.device);
  const auto gdpa =
      reinterpret_cast<PFN_vkGetDeviceProcAddr>(native.get_device_proc_addr);
  REQUIRE(gdpa != nullptr);
  const auto load = [&](const char *const name) {
    const PFN_vkVoidFunction found = gdpa(vk_device, name);
    REQUIRE(found != nullptr);
    return found;
  };
  const auto clear =
      reinterpret_cast<PFN_vkCmdClearColorImage>(load("vkCmdClearColorImage"));
  const auto bind_sets = reinterpret_cast<PFN_vkCmdBindDescriptorSets>(
      load("vkCmdBindDescriptorSets"));
  const auto create_set_layout =
      reinterpret_cast<PFN_vkCreateDescriptorSetLayout>(
          load("vkCreateDescriptorSetLayout"));
  const auto destroy_set_layout =
      reinterpret_cast<PFN_vkDestroyDescriptorSetLayout>(
          load("vkDestroyDescriptorSetLayout"));
  const auto create_pool = reinterpret_cast<PFN_vkCreateDescriptorPool>(
      load("vkCreateDescriptorPool"));
  const auto destroy_pool = reinterpret_cast<PFN_vkDestroyDescriptorPool>(
      load("vkDestroyDescriptorPool"));
  const auto allocate_sets = reinterpret_cast<PFN_vkAllocateDescriptorSets>(
      load("vkAllocateDescriptorSets"));
  const auto create_layout = reinterpret_cast<PFN_vkCreatePipelineLayout>(
      load("vkCreatePipelineLayout"));
  const auto destroy_layout = reinterpret_cast<PFN_vkDestroyPipelineLayout>(
      load("vkDestroyPipelineLayout"));

  // A set of the library's own, bound where the device keeps its bindless
  // one, as a vendor library binds its own.
  const VkDescriptorSetLayoutBinding binding{
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
  const VkDescriptorSetLayoutCreateInfo set_info{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &binding};
  VkDescriptorSetLayout foreign_set_layout = VK_NULL_HANDLE;
  REQUIRE(create_set_layout(vk_device, &set_info, nullptr,
                            &foreign_set_layout) == VK_SUCCESS);
  const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
  const VkDescriptorPoolCreateInfo pool_info{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 1,
      .pPoolSizes = &pool_size};
  VkDescriptorPool foreign_pool = VK_NULL_HANDLE;
  REQUIRE(create_pool(vk_device, &pool_info, nullptr, &foreign_pool) ==
          VK_SUCCESS);
  const VkDescriptorSetAllocateInfo allocate_info{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = foreign_pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &foreign_set_layout};
  VkDescriptorSet foreign_set = VK_NULL_HANDLE;
  REQUIRE(allocate_sets(vk_device, &allocate_info, &foreign_set) == VK_SUCCESS);
  const VkPipelineLayoutCreateInfo layout_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &foreign_set_layout};
  VkPipelineLayout foreign_layout = VK_NULL_HANDLE;
  REQUIRE(create_layout(vk_device, &layout_info, nullptr, &foreign_layout) ==
          VK_SUCCESS);

  ValidationErrors *const errors = nx::allocate<ValidationErrors>();
  nx::log::add_sink(errors);
  {
    const gfx::shader_handle shader =
        nx::gfx::test::create_shader(device,
                                     {.name = "native read",
                                      .code = code->data(),
                                      .code_size = code->size()},
                                     refl->view());
    REQUIRE(shader.valid());
    const gfx::pipeline_handle read = nx::gfx::test::compile_pipeline(
        device, gfx::compute_pipeline_desc{
                    .name = "native read",
                    .compute = {.shader = shader, .entry_point = "read_main"}});
    REQUIRE(read.valid());

    const gfx::texture_handle texture =
        gfx::create_texture(device, {
                                        .name = "native cleared",
                                        .format = gfx::format::RGBA32_FLOAT,
                                        .width = 4,
                                        .height = 4,
                                        .usage = gfx::texture_usage::Sampled |
                                                 gfx::texture_usage::CopyDst,
                                    });
    REQUIRE(texture.valid());
    const gfx::native_texture image = gfx::native(device, texture);
    CHECK(image.image != nullptr);
    CHECK(image.view != nullptr);
    CHECK(image.format == VK_FORMAT_R32G32B32A32_SFLOAT);
    CHECK(image.width == 4);
    CHECK(image.height == 4);
    CHECK(gfx::native(device, gfx::texture_handle{}).image == nullptr);

    const gfx::buffer_handle out = gfx::create_buffer(
        device, {
                    .name = "native read out",
                    .size = sizeof(Rgba),
                    .usage = gfx::buffer_usage::Storage |
                             gfx::buffer_usage::DeviceAddress,
                    .memory = gfx::memory_usage::Readback,
                    .persistently_mapped = true,
                });
    REQUIRE(out.valid());
    std::memset(gfx::buffer_mapped(device, out), 0, sizeof(Rgba));

    gfx::command_list *cmd = nullptr;
    REQUIRE(((cmd = gfx::begin_headless_frame(device)) != nullptr));
    gfx::barrier(*cmd,
                 gfx::texture_barrier{.texture = texture,
                                      .from = gfx::resource_state::Undefined,
                                      .to = gfx::resource_state::CopyDst});
    const auto buffer = static_cast<VkCommandBuffer>(gfx::begin_native(*cmd));
    REQUIRE(buffer != VK_NULL_HANDLE);
    const VkClearColorValue colour{{0.25f, 0.5f, 0.75f, 1.0f}};
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    clear(buffer, static_cast<VkImage>(image.image),
          static_cast<VkImageLayout>(
              gfx::native_layout(device, gfx::resource_state::CopyDst)),
          &colour, 1, &range);
    bind_sets(buffer, VK_PIPELINE_BIND_POINT_COMPUTE, foreign_layout, 0, 1,
              &foreign_set, 0, nullptr);
    gfx::end_native(*cmd);
    gfx::barrier(
        *cmd, gfx::texture_barrier{.texture = texture,
                                   .from = gfx::resource_state::CopyDst,
                                   .to = gfx::resource_state::ShaderReadOnly});
    const Push push{.out = gfx::buffer_address(device, out),
                    .texture = NxTexture2D<Rgba>::from_indices(
                                   gfx::texture_index(device, texture), 0)
                                   .value};
    gfx::bind_pipeline(*cmd, read);
    gfx::push_constants(*cmd, &push, sizeof(push));
    gfx::dispatch(*cmd, 1);
    REQUIRE(gfx::end_headless_frame(device));
    gfx::wait_idle(device);

    Rgba got;
    std::memcpy(&got, gfx::buffer_mapped(device, out), sizeof(got));
    CHECK((got.r == 0.25f && got.g == 0.5f && got.b == 0.75f && got.a == 1.0f));

    gfx::destroy_buffer(device, out);
    gfx::destroy_texture(device, texture);
    gfx::destroy_pipeline(device, read);
    gfx::destroy_shader(device, shader);
  }
  nx::log::flush();
  const u32 counted = errors->count();
  nx::log::remove_sink(errors);
  CHECK(counted == 0u);
  destroy_layout(vk_device, foreign_layout, nullptr);
  destroy_pool(vk_device, foreign_pool, nullptr);
  destroy_set_layout(vk_device, foreign_set_layout, nullptr);
}

#endif
