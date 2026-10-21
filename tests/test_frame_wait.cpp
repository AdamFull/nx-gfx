#include "framework/nxtest.h"

#include "core/foundation/threading/thread.h"
#include "nx/gfx/gfx.h"
#include "nx/gfx/native.h"
#include "owned_device.h"

#if defined(NX_RHI_VULKAN)

#include "nx/gfx/vulkan.h"

#include <atomic>
#include <chrono>

namespace gfx = nx::gfx;

namespace {

[[nodiscard]] bool run_frame(gfx::device &device) {
  return gfx::begin_headless_frame(device) != nullptr &&
         gfx::end_headless_frame(device);
}

/// The host signal, under its core name where the device has Vulkan 1.2:
/// the device loads only the timeline commands it calls itself.
[[nodiscard]] PFN_vkSignalSemaphore host_signal(gfx::device &device) {
  const auto gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
      gfx::native(device).get_device_proc_addr);
  const VkDevice vk = gfx::vk::logical_device(device);
  PFN_vkVoidFunction found = gdpa(vk, "vkSignalSemaphore");
  if (found == nullptr)
    found = gdpa(vk, "vkSignalSemaphoreKHR");
  return reinterpret_cast<PFN_vkSignalSemaphore>(found);
}

void signal(gfx::device &device, const VkSemaphore semaphore, const u64 value) {
  const PFN_vkSignalSemaphore fn = host_signal(device);
  REQUIRE(fn != nullptr);
  VkSemaphoreSignalInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
  info.semaphore = semaphore;
  info.value = value;
  REQUIRE(fn(gfx::vk::logical_device(device), &info) == VK_SUCCESS);
}

} // namespace

TEST_CASE("gfx frame wait: a frame does not finish before the client's "
          "timeline reaches the value named") {
  nx::gfx::test::OwnedDevice device;
  if (!device.init({.application_name = "nx frame wait tests"}))
    SKIP("no usable device");
  const VkSemaphore timeline = gfx::vk::acquire_frame_wait(device);
  REQUIRE(timeline != VK_NULL_HANDLE);

  const PFN_vkSignalSemaphore fn = host_signal(device);
  REQUIRE(fn != nullptr);
  const VkDevice vk = gfx::vk::logical_device(device);

  gfx::vk::wait_before_frame(device, {timeline, 1});
  std::atomic<bool> signalled{false};
  {
    nx::thread client([&] {
      nx::this_thread::sleep_for(std::chrono::milliseconds(100));
      signalled.store(true, std::memory_order_release);
      VkSemaphoreSignalInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
      info.semaphore = timeline;
      info.value = 1;
      (void)fn(vk, &info);
    });
    REQUIRE(run_frame(device));
    CHECK(signalled.load(std::memory_order_acquire));
  }

  // A value already met costs the next frames nothing.
  REQUIRE(run_frame(device));
  gfx::vk::wait_before_frame(device, {timeline, 1});
  REQUIRE(run_frame(device));
  gfx::vk::release_frame_wait(device, timeline);
}

TEST_CASE("gfx frame wait: once released, frames no longer wait, and the "
          "device destroys the semaphore after them") {
  nx::gfx::test::OwnedDevice device;
  if (!device.init({.application_name = "nx frame wait tests"}))
    SKIP("no usable device");
  const VkSemaphore timeline = gfx::vk::acquire_frame_wait(device);
  REQUIRE(timeline != VK_NULL_HANDLE);
  signal(device, timeline, 1);
  gfx::vk::wait_before_frame(device, {timeline, 1});
  REQUIRE(run_frame(device));

  // Never signalled: a frame that still waited on it would never finish.
  gfx::vk::wait_before_frame(device, {timeline, 2});
  gfx::vk::release_frame_wait(device, timeline);
  for (u32 frame = 0; frame < 4; ++frame)
    REQUIRE(run_frame(device));

  // Its place is free again: the device holds at most eight.
  VkSemaphore held[8] = {};
  for (VkSemaphore &one : held) {
    one = gfx::vk::acquire_frame_wait(device);
    CHECK(one != VK_NULL_HANDLE);
  }
  CHECK(gfx::vk::acquire_frame_wait(device) == VK_NULL_HANDLE);
  for (const VkSemaphore one : held)
    gfx::vk::release_frame_wait(device, one);
}

TEST_CASE("gfx frame wait: a device torn down with frame waits still held "
          "frees them") {
  nx::gfx::test::OwnedDevice device;
  if (!device.init({.application_name = "nx frame wait tests"}))
    SKIP("no usable device");
  const VkSemaphore timeline = gfx::vk::acquire_frame_wait(device);
  REQUIRE(timeline != VK_NULL_HANDLE);
  signal(device, timeline, 3);
  gfx::vk::wait_before_frame(device, {timeline, 3});
  REQUIRE(run_frame(device));
  REQUIRE(gfx::vk::acquire_frame_wait(device) != VK_NULL_HANDLE);
  device.shutdown();
}

#endif
