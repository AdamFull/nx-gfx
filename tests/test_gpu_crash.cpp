#include "framework/nxtest.h"

#include "core/foundation/serialization/json.h"
#include "nx/gfx/crash.h"

#if defined(NX_RHI_VULKAN)
#include "src/vulkan/vk_breadcrumbs.h"
#include "src/vulkan/vk_gpu_crash.h"
#endif

namespace gfx = nx::gfx;

TEST_CASE(
    "gpu crash: breadcrumbs say which groups were done, which running and "
    "which never reached, however far the numbers have wrapped") {
  namespace g = gfx::detail;
  using State = g::VulkanBreadcrumbs::State;
  constexpr u32 COUNT = g::VulkanBreadcrumbs::ENTRIES;
  nx::vector<u32> words(usize{COUNT} * 2u, 0u);
  const auto word = [](const u32 number, const u32 label) {
    return number << 12u | label;
  };
  const auto started = [&](const u32 number, const u32 label) {
    words[usize{number % COUNT} * 2u] = word(number, label);
  };
  const auto done = [&](const u32 number, const u32 label) {
    words[usize{number % COUNT} * 2u + 1u] = word(number, label);
  };
  SUBCASE("nothing marked") {
    CHECK(g::VulkanBreadcrumbs::read(words, 1, 8).empty());
  }
  SUBCASE("a pass stuck: those before it done, those after it not reached") {
    for (u32 n = 1; n <= 5; ++n) {
      started(n, n + 10u);
      done(n, n + 10u);
    }
    started(6, 16);
    // Six more recorded and never reached.
    const auto read = g::VulkanBreadcrumbs::read(words, 13, 100);
    REQUIRE(read.size() == 9u);
    CHECK(read[0].state == State::Done);
    CHECK(read[0].label == 11u);
    CHECK(read[4].state == State::Done);
    CHECK(read[5].state == State::Running);
    CHECK(read[5].label == 16u);
    CHECK(read[6].state == State::NotStarted);
    CHECK(read[8].state == State::NotStarted);
  }
  SUBCASE("only the last asked for, the newest kept") {
    for (u32 n = 1; n <= 20; ++n) {
      started(n, n);
      done(n, n);
    }
    const auto read = g::VulkanBreadcrumbs::read(words, 21, 4);
    REQUIRE(read.size() == 4u);
    CHECK(read[0].label == 17u);
    CHECK(read[3].label == 20u);
  }
  SUBCASE("a slot still holding an older group's word is not this group's") {
    // Group COUNT + 5 lands on group 5's slot and was never reached, and
    // group 5's words are still there.
    for (u32 n = 6; n < COUNT + 5u; ++n) {
      started(n, 1);
      done(n, 1);
    }
    started(5, 2);
    done(5, 2);
    const auto read = g::VulkanBreadcrumbs::read(words, COUNT + 6u, 2);
    REQUIRE(read.size() == 2u);
    CHECK(read[0].state == State::Done);
    CHECK(read[1].state == State::NotStarted);
  }
}

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
