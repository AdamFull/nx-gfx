#include "framework/nxtest.h"

#include "core/foundation/serialization/json.h"
#include "nx/gfx/crash.h"

#if defined(NX_RHI_VULKAN)
#include "src/vulkan/vk_gpu_crash.h"
#endif

namespace gfx = nx::gfx;

TEST_CASE("gpu crash: structured report preserves object identities") {
  gfx::gpu_crash_report report;
  report.backend = gfx::gpu_backend::Vulkan;
  report.device = "test gpu";
  report.operation = "submit";
  report.result = "device lost";
  report.description = "fault";
  report.frame = 42;
  report.addresses.push_back({"invalid write", 0x1234, 16});
  report.objects.push_back({"buffer", "paper doll instances", 0xabcd});
  report.bindings.push_back(
      {"buffer", "paper doll instances", 0xabcd, 0x1000, 4096, true, false});
  report.vendor_faults.push_back({"page fault", 7, 9});
  report.encoders.push_back(
      {"paper doll composition", "faulted", {nx::string("draw sprites")}});

  const nx::string encoded = gfx::gpu_crash_json(report);
  auto parsed = nx::json::parse(encoded.view());
  REQUIRE(parsed);
  REQUIRE(parsed->is_object());
  CHECK((*parsed)["schema_version"].get_u64() == 1u);
  CHECK((*parsed)["backend"].get_string() == "Vulkan");
  CHECK((*parsed)["frame"].get_u64() == 42u);

  const nx::json::value &object = (*parsed)["objects"][0];
  CHECK(object["object_type"].get_string() == "buffer");
  CHECK(object["object_name"].get_string() == "paper doll instances");
  CHECK(object["object_handle"].get_string() == "0x000000000000abcd");

  const nx::json::value &binding = (*parsed)["address_bindings"][0];
  CHECK(binding["object_type"].get_string() == "buffer");
  CHECK(binding["object_name"].get_string() == "paper doll instances");
  CHECK(binding["object_handle"].get_string() == "0x000000000000abcd");
  CHECK(binding["address"].get_string() == "0x0000000000001000");

  const nx::json::value &encoder = (*parsed)["command_encoders"][0];
  CHECK(encoder["label"].get_string() == "paper doll composition");
  CHECK(encoder["debug_signposts"][0].get_string() == "draw sprites");
}

TEST_CASE("gpu crash: a report names the files it carries beyond its JSON") {
  gfx::gpu_crash_report report;
  report.attachments.push_back(
      {.name = "gpu-aftermath.nv-gpudmp", .bytes = nx::blob<u8>(4, u8{1})});
  report.attachments.push_back({.name = "gpu-aftermath.json", .bytes = {}});

  const nx::string encoded = gfx::gpu_crash_json(report);
  auto parsed = nx::json::parse(encoded.view());
  REQUIRE(parsed);
  const nx::json::value &attachments = (*parsed)["attachments"];
  REQUIRE(attachments.is_array());
  REQUIRE(attachments.as_array().size() == 2u);
  CHECK(attachments[0].get_string() == "gpu-aftermath.nv-gpudmp");
  CHECK(attachments[1].get_string() == "gpu-aftermath.json");
}

#if defined(NX_RHI_VULKAN)
TEST_CASE(
    "gpu crash: Vulkan device loss is terminal even when reports are off") {
  gfx::detail::VulkanGpuCrashReporter reporter;
  reporter.configure(gfx::gpu_crash_mode::Disabled, false);

  CHECK(reporter.status() == gfx::gpu_device_status::Ready);
  CHECK(reporter.check(VK_ERROR_DEVICE_LOST, "test", __FILE__, __LINE__, 0) ==
        VK_ERROR_DEVICE_LOST);
  CHECK(reporter.status() == gfx::gpu_device_status::Lost);

  reporter.reset();
  CHECK(reporter.status() == gfx::gpu_device_status::Ready);
}

TEST_CASE("gpu crash: Nsight Aftermath not started keeps nothing and adds "
          "nothing to a report") {
  gfx::detail::VulkanAftermath aftermath;
  CHECK_FALSE(aftermath.active());
  const u32 spirv[] = {0x07230203u, 0x00010300u, 0u, 1u, 0u};
  aftermath.shader(spirv, sizeof(spirv), "test shader");
  gfx::gpu_crash_report report;
  aftermath.collect(report);
  CHECK(report.attachments.empty());
  CHECK(report.vendor_faults.empty());
  CHECK(report.description.empty());
}
#endif
