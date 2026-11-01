#include "framework/nxtest.h"

#include "core/foundation/platform/os.h"
#include "nx/gfx/gfx.h"
#include "owned_device.h"

#if defined(NX_RHI_VULKAN)

#include "nx/gfx/native.h"
#include "nx/gfx/vulkan.h"
#include "src/vulkan/vk_gpu_markers.h"

#include <array>
#include <cstring>

namespace gfx = nx::gfx;

namespace {

[[nodiscard]] bool has_timestamps(gfx::device &device) {
  return gfx::caps(device).timestamps;
}

/// One frame's work, and enough frames after it that its slot comes round again
/// and is read.
template <typename Work> void run_and_read(gfx::device &device, Work &&work) {
  gfx::command_list *const list = gfx::begin_headless_frame(device);
  REQUIRE(list != nullptr);
  work(*list);
  REQUIRE(gfx::end_headless_frame(device));
  for (int i = 0; i < 4; ++i) {
    REQUIRE(gfx::begin_headless_frame(device) != nullptr);
    REQUIRE(gfx::end_headless_frame(device));
  }
}

[[nodiscard]] usize read_all(gfx::device &device,
                             nx::vector<gfx::gpu_marker> &out) {
  out.resize(16384);
  const usize count = gfx::read_gpu_markers(device, {out.data(), out.size()});
  out.resize(count);
  return count;
}

} // namespace

TEST_CASE("gpu markers: off, debug groups time nothing") {
  nx::gfx::test::OwnedDevice owned;
  if (!owned.init({.application_name = "nx gpu marker tests"}))
    SKIP("no usable device");
  gfx::device &device = owned;
  run_and_read(device, [](gfx::command_list &list) {
    gfx::push_debug_group(list, "quiet");
    gfx::pop_debug_group(list);
  });
  nx::vector<gfx::gpu_marker> out;
  CHECK(read_all(device, out) == 0);
  CHECK(gfx::gpu_markers_stats(device).frames == 0);
}

TEST_CASE("gpu markers: groups come back named, nested, in order and timed") {
  nx::gfx::test::OwnedDevice owned;
  if (!owned.init({.application_name = "nx gpu marker tests"}))
    SKIP("no usable device");
  gfx::device &device = owned;
  if (!has_timestamps(device))
    SKIP("no timestamps");
  gfx::set_gpu_markers(device, true);
  // The next frame is the first that times.
  const u64 began = nx::os::ticks_ns();
  run_and_read(device, [](gfx::command_list &list) {
    gfx::push_debug_group(list, "outer");
    gfx::push_debug_group(list, "inner");
    gfx::pop_debug_group(list);
    gfx::pop_debug_group(list);
    gfx::push_debug_group(list, "after");
    gfx::pop_debug_group(list);
  });
  const u64 ended = nx::os::ticks_ns();
  nx::vector<gfx::gpu_marker> out;
  REQUIRE(read_all(device, out) >= 3);
  usize at = out.size();
  for (usize i = 0; i < out.size(); ++i)
    if (out[i].text() == "outer") {
      at = i;
      break;
    }
  REQUIRE(at + 2 < out.size());
  const gfx::gpu_marker &outer = out[at];
  const gfx::gpu_marker &inner = out[at + 1];
  const gfx::gpu_marker &after = out[at + 2];
  CHECK(inner.text() == "inner");
  CHECK(after.text() == "after");
  CHECK(outer.depth == 0);
  CHECK(inner.depth == 1);
  CHECK(after.depth == 0);
  CHECK(outer.begin_ns <= inner.begin_ns);
  CHECK(inner.end_ns <= outer.end_ns);
  CHECK(outer.end_ns <= after.begin_ns);
  CHECK(outer.end_ns >= outer.begin_ns);

  // On the host's clock, within what the frames took and some slack for reading
  // the two clocks a moment apart.
  constexpr u64 SLACK_NS = 50'000'000;
  CHECK(outer.begin_ns + SLACK_NS >= began);
  CHECK(after.end_ns <= ended + SLACK_NS);
  gfx::set_gpu_markers(device, false);
}

TEST_CASE("gpu markers: a read takes what fits and leaves the rest") {
  nx::gfx::test::OwnedDevice owned;
  if (!owned.init({.application_name = "nx gpu marker tests"}))
    SKIP("no usable device");
  gfx::device &device = owned;
  if (!has_timestamps(device))
    SKIP("no timestamps");
  gfx::set_gpu_markers(device, true);
  run_and_read(device, [](gfx::command_list &list) {
    for (int i = 0; i < 5; ++i) {
      gfx::push_debug_group(list, "one of five");
      gfx::pop_debug_group(list);
    }
  });
  std::array<gfx::gpu_marker, 2> two;
  CHECK(gfx::read_gpu_markers(device, two) == 2);
  nx::vector<gfx::gpu_marker> rest;
  CHECK(read_all(device, rest) >= 3);
  gfx::set_gpu_markers(device, false);
}

TEST_CASE("gpu markers: a frame past its room counts what it could not time") {
  nx::gfx::test::OwnedDevice owned;
  if (!owned.init({.application_name = "nx gpu marker tests"}))
    SKIP("no usable device");
  gfx::device &device = owned;
  if (!has_timestamps(device))
    SKIP("no timestamps");
  gfx::set_gpu_markers(device, true);
  constexpr u32 EXTRA = 10;
  run_and_read(device, [](gfx::command_list &list) {
    for (u32 i = 0; i < 1024 + EXTRA; ++i) {
      gfx::push_debug_group(list, "many");
      gfx::pop_debug_group(list);
    }
  });
  CHECK(gfx::gpu_markers_stats(device).groups_dropped >= EXTRA);
  nx::vector<gfx::gpu_marker> out;
  CHECK(read_all(device, out) >= 1024);
  gfx::set_gpu_markers(device, false);
}

TEST_CASE("gpu markers: groups nested too deep are counted, and still pair") {
  nx::gfx::test::OwnedDevice owned;
  if (!owned.init({.application_name = "nx gpu marker tests"}))
    SKIP("no usable device");
  gfx::device &device = owned;
  if (!has_timestamps(device))
    SKIP("no timestamps");
  gfx::set_gpu_markers(device, true);
  constexpr int DEEP = 40;
  run_and_read(device, [](gfx::command_list &list) {
    for (int i = 0; i < DEEP; ++i)
      gfx::push_debug_group(list, "deep");
    for (int i = 0; i < DEEP; ++i)
      gfx::pop_debug_group(list);
    gfx::push_debug_group(list, "level again");
    gfx::pop_debug_group(list);
  });
  CHECK(gfx::gpu_markers_stats(device).groups_dropped >= 8);
  nx::vector<gfx::gpu_marker> out;
  REQUIRE(read_all(device, out) >= 33);
  bool found = false;
  for (const gfx::gpu_marker &marker : out)
    if (marker.text() == "level again") {
      CHECK(marker.depth == 0);
      found = true;
    }
  CHECK(found);
  gfx::set_gpu_markers(device, false);
}

TEST_CASE("gpu markers: a group left open is closed with its frame") {
  nx::gfx::test::OwnedDevice owned;
  if (!owned.init({.application_name = "nx gpu marker tests"}))
    SKIP("no usable device");
  gfx::device &device = owned;
  if (!has_timestamps(device))
    SKIP("no timestamps");
  gfx::set_gpu_markers(device, true);
  run_and_read(device, [](gfx::command_list &list) {
    gfx::push_debug_group(list, "never popped");
  });
  nx::vector<gfx::gpu_marker> out;
  REQUIRE(read_all(device, out) >= 1);
  CHECK(out[0].text() == "never popped");
  CHECK(out[0].end_ns >= out[0].begin_ns);
  gfx::set_gpu_markers(device, false);
}

TEST_CASE("gpu markers: a long name is cut, not overrun") {
  nx::gfx::test::OwnedDevice owned;
  if (!owned.init({.application_name = "nx gpu marker tests"}))
    SKIP("no usable device");
  gfx::device &device = owned;
  if (!has_timestamps(device))
    SKIP("no timestamps");
  gfx::set_gpu_markers(device, true);
  char long_name[200];
  std::memset(long_name, 'x', sizeof(long_name));
  run_and_read(device, [&](gfx::command_list &list) {
    gfx::push_debug_group(list, nx::string_view(long_name, sizeof(long_name)));
    gfx::pop_debug_group(list);
  });
  nx::vector<gfx::gpu_marker> out;
  REQUIRE(read_all(device, out) >= 1);
  CHECK(out[0].text().size() == gfx::GPU_MARKER_NAME_BYTES);
  gfx::set_gpu_markers(device, false);
}

TEST_CASE("gpu markers: nobody reading drops the oldest and counts them") {
  nx::gfx::test::OwnedDevice owned;
  if (!owned.init({.application_name = "nx gpu marker tests"}))
    SKIP("no usable device");
  gfx::device &device = owned;
  if (!has_timestamps(device))
    SKIP("no timestamps");
  gfx::set_gpu_markers(device, true);
  for (int frame = 0; frame < 16; ++frame)
    run_and_read(device, [](gfx::command_list &list) {
      for (int i = 0; i < 1024; ++i) {
        gfx::push_debug_group(list, "unread");
        gfx::pop_debug_group(list);
      }
    });
  CHECK(gfx::gpu_markers_stats(device).markers_dropped > 0);
  gfx::set_gpu_markers(device, false);
}

TEST_CASE("gpu markers: turned off, then on, it picks up again") {
  nx::gfx::test::OwnedDevice owned;
  if (!owned.init({.application_name = "nx gpu marker tests"}))
    SKIP("no usable device");
  gfx::device &device = owned;
  if (!has_timestamps(device))
    SKIP("no timestamps");
  gfx::set_gpu_markers(device, true);
  run_and_read(device, [](gfx::command_list &list) {
    gfx::push_debug_group(list, "first");
    gfx::pop_debug_group(list);
  });
  nx::vector<gfx::gpu_marker> out;
  CHECK(read_all(device, out) >= 1);

  gfx::set_gpu_markers(device, false);
  run_and_read(device, [](gfx::command_list &list) {
    gfx::push_debug_group(list, "while off");
    gfx::pop_debug_group(list);
  });
  // A frame that was recording as it turned off may still arrive, but none
  // after.
  (void)read_all(device, out);
  run_and_read(device, [](gfx::command_list &list) {
    gfx::push_debug_group(list, "still off");
    gfx::pop_debug_group(list);
  });
  CHECK(read_all(device, out) == 0);

  gfx::set_gpu_markers(device, true);
  run_and_read(device, [](gfx::command_list &list) {
    gfx::push_debug_group(list, "again");
    gfx::pop_debug_group(list);
  });
  REQUIRE(read_all(device, out) >= 1);
  bool again = false;
  for (const gfx::gpu_marker &marker : out)
    again = again || marker.text() == "again";
  CHECK(again);
  gfx::set_gpu_markers(device, false);
}

namespace {

/// The markers on their own, on the device's graphics queue, for what a
/// device's frames cannot be made to do.
struct DirectMarkers {
  gfx::device &device;
  VkDevice vk = gfx::vk::logical_device(device);
  VkQueue queue = static_cast<VkQueue>(gfx::native(device).queue);
  VkCommandPool pool = VK_NULL_HANDLE;
  nx::gfx::detail::VulkanGpuMarkers markers;

  DirectMarkers(gfx::device &owner, const bool inside_passes) : device(owner) {
    VkCommandPoolCreateInfo info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    info.queueFamilyIndex = gfx::vk::graphics_family(device);
    REQUIRE(vkCreateCommandPool(vk, &info, nullptr, &pool) == VK_SUCCESS);
    markers.init(vk, 1, gfx::caps(device).timestamp_period_ns, 64, 0, nullptr,
                 inside_passes);
    markers.set_enabled(true);
  }
  ~DirectMarkers() {
    vkQueueWaitIdle(queue);
    markers.shutdown();
    vkDestroyCommandPool(vk, pool, nullptr);
  }

  [[nodiscard]] VkCommandBuffer begin() {
    VkCommandBufferAllocateInfo alloc{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    REQUIRE(vkAllocateCommandBuffers(vk, &alloc, &cmd) == VK_SUCCESS);
    VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    REQUIRE(vkBeginCommandBuffer(cmd, &info) == VK_SUCCESS);
    return cmd;
  }

  void submit(const VkCommandBuffer cmd) {
    REQUIRE(vkEndCommandBuffer(cmd) == VK_SUCCESS);
    VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    info.commandBufferCount = 1;
    info.pCommandBuffers = &cmd;
    REQUIRE(vkQueueSubmit(queue, 1, &info, VK_NULL_HANDLE) == VK_SUCCESS);
    REQUIRE(vkQueueWaitIdle(queue) == VK_SUCCESS);
  }

  /// A frame timing nothing, which brings the slot round.
  [[nodiscard]] usize next_frame(std::span<gfx::gpu_marker> out) {
    const VkCommandBuffer cmd = begin();
    (void)markers.open_frame(0, cmd);
    submit(cmd);
    return markers.read(out);
  }
};

} // namespace

TEST_CASE("gpu markers: uncalibrated, a frame is placed by its submit") {
  nx::gfx::test::OwnedDevice owned;
  if (!owned.init({.application_name = "nx gpu marker tests"}))
    SKIP("no usable device");
  gfx::device &device = owned;
  if (!has_timestamps(device))
    SKIP("no timestamps");
  DirectMarkers direct(device, true);
  CHECK_FALSE(direct.markers.stats().calibrated);

  const VkCommandBuffer cmd = direct.begin();
  auto *const slot = direct.markers.open_frame(0, cmd);
  REQUIRE(slot != nullptr);
  direct.markers.push(*slot, cmd, "first", false);
  direct.markers.pop(*slot, cmd);
  const u64 before = nx::os::ticks_ns();
  direct.markers.frame_submitted(0);
  const u64 after = nx::os::ticks_ns();
  direct.submit(cmd);

  std::array<gfx::gpu_marker, 4> out;
  REQUIRE(direct.next_frame(out) == 1);
  CHECK(out[0].begin_ns >= before);
  CHECK(out[0].begin_ns <= after);
  CHECK(out[0].end_ns >= out[0].begin_ns);
}

TEST_CASE(
    "gpu markers: a frame never submitted gives nothing and costs nothing "
    "later") {
  nx::gfx::test::OwnedDevice owned;
  if (!owned.init({.application_name = "nx gpu marker tests"}))
    SKIP("no usable device");
  gfx::device &device = owned;
  if (!has_timestamps(device))
    SKIP("no timestamps");
  DirectMarkers direct(device, true);

  const VkCommandBuffer dropped = direct.begin();
  auto *const slot = direct.markers.open_frame(0, dropped);
  REQUIRE(slot != nullptr);
  direct.markers.push(*slot, dropped, "dropped", false);
  direct.markers.pop(*slot, dropped);
  REQUIRE(vkEndCommandBuffer(dropped) == VK_SUCCESS);

  const VkCommandBuffer cmd = direct.begin();
  auto *const again = direct.markers.open_frame(0, cmd);
  REQUIRE(again != nullptr);
  std::array<gfx::gpu_marker, 4> out;
  CHECK(direct.markers.read(out) == 0);
  CHECK(direct.markers.stats().frames == 0);
  direct.markers.push(*again, cmd, "kept", false);
  direct.markers.pop(*again, cmd);
  direct.markers.frame_submitted(0);
  direct.submit(cmd);

  REQUIRE(direct.next_frame(out) == 1);
  CHECK(out[0].text() == "kept");
}

TEST_CASE("gpu markers: where a pass cannot be timed its groups still pair") {
  nx::gfx::test::OwnedDevice owned;
  if (!owned.init({.application_name = "nx gpu marker tests"}))
    SKIP("no usable device");
  gfx::device &device = owned;
  if (!has_timestamps(device))
    SKIP("no timestamps");
  DirectMarkers direct(device, false);

  const VkCommandBuffer cmd = direct.begin();
  auto *const slot = direct.markers.open_frame(0, cmd);
  REQUIRE(slot != nullptr);
  direct.markers.push(*slot, cmd, "around", false);
  direct.markers.push(*slot, cmd, "in a pass", true);
  direct.markers.pop(*slot, cmd);
  direct.markers.pop(*slot, cmd);
  direct.markers.push(*slot, cmd, "after", false);
  direct.markers.pop(*slot, cmd);
  direct.markers.frame_submitted(0);
  direct.submit(cmd);

  std::array<gfx::gpu_marker, 4> out;
  REQUIRE(direct.next_frame(out) == 2);
  CHECK(out[0].text() == "around");
  CHECK(out[1].text() == "after");
  CHECK(out[1].depth == 0);
  CHECK(out[0].end_ns <= out[1].begin_ns);
}

TEST_CASE(
    "gpu markers: split lists are timed, in the order they are submitted") {
  nx::gfx::test::OwnedDevice owned;
  if (!owned.init({.application_name = "nx gpu marker tests"}))
    SKIP("no usable device");
  gfx::device &device = owned;
  if (!has_timestamps(device))
    SKIP("no timestamps");
  gfx::set_gpu_markers(device, true);
  run_and_read(device, [&](gfx::command_list &list) {
    gfx::push_debug_group(list, "before");
    gfx::pop_debug_group(list);
    const gfx::recording_split split = gfx::split_frame(device, list, 2);
    REQUIRE(split.count == 2);
    // The second list is recorded first and still runs second.
    gfx::command_list *const second = gfx::begin_recording(device, split, 1);
    REQUIRE(second != nullptr);
    gfx::command_list *const first = gfx::begin_recording(device, split, 0);
    REQUIRE(first != nullptr);
    gfx::push_debug_group(*second, "second");
    gfx::pop_debug_group(*second);
    gfx::push_debug_group(*first, "first");
    gfx::push_debug_group(*first, "left open");
    gfx::end_recording(device, *second);
    gfx::end_recording(device, *first);
    gfx::push_debug_group(list, "after");
    gfx::pop_debug_group(list);
  });
  nx::vector<gfx::gpu_marker> out;
  REQUIRE(read_all(device, out) == 5);
  CHECK(out[0].text() == "before");
  CHECK(out[1].text() == "first");
  CHECK(out[2].text() == "left open");
  CHECK(out[2].depth == 1);
  CHECK(out[3].text() == "second");
  CHECK(out[4].text() == "after");
  for (usize i = 1; i < out.size(); ++i) {
    CHECK(out[i - 1].begin_ns <= out[i].begin_ns);
    CHECK(out[i - 1].frame == out[i].frame);
  }
  CHECK(out[1].end_ns <= out[3].begin_ns);
  gfx::set_gpu_markers(device, false);
}

TEST_CASE("gpu markers: frames are told apart") {
  nx::gfx::test::OwnedDevice owned;
  if (!owned.init({.application_name = "nx gpu marker tests"}))
    SKIP("no usable device");
  gfx::device &device = owned;
  if (!has_timestamps(device))
    SKIP("no timestamps");
  gfx::set_gpu_markers(device, true);
  for (int frame = 0; frame < 2; ++frame)
    run_and_read(device, [](gfx::command_list &list) {
      gfx::push_debug_group(list, "one");
      gfx::push_debug_group(list, "two");
      gfx::pop_debug_group(list);
      gfx::pop_debug_group(list);
    });
  nx::vector<gfx::gpu_marker> out;
  REQUIRE(read_all(device, out) == 4);
  CHECK(out[0].frame == out[1].frame);
  CHECK(out[1].frame < out[2].frame);
  CHECK(out[2].frame == out[3].frame);
  gfx::set_gpu_markers(device, false);
}

TEST_CASE(
    "gpu markers: a copy on the copy queue is timed on the host's clock") {
  nx::gfx::test::OwnedDevice owned;
  if (!owned.init({.application_name = "nx gpu marker tests"}))
    SKIP("no usable device");
  gfx::device &device = owned;
  if (!has_timestamps(device))
    SKIP("no timestamps");
  if (gfx::vk::queue_families(device).size() < 2)
    SKIP("no copy queue of its own");
  gfx::set_gpu_markers(device, true);
  // A first frame on, so the clocks are matched before the copy.
  run_and_read(device, [](gfx::command_list &) {});
  const gfx::buffer_handle buffer =
      gfx::create_buffer(device, {.name = "marker copy",
                                  .size = 4096,
                                  .usage = gfx::buffer_usage::CopyDst |
                                           gfx::buffer_usage::Storage});
  REQUIRE(buffer.valid());
  const u8 bytes[4096] = {};
  const u64 began = nx::os::ticks_ns();
  const gfx::upload_ticket ticket =
      gfx::upload_buffer(device, buffer, 0, bytes);
  REQUIRE(ticket.ok());
  // Uploads go with the frame after them.
  run_and_read(device, [](gfx::command_list &) {});
  gfx::wait_idle(device);
  run_and_read(device, [](gfx::command_list &) {});
  const u64 ended = nx::os::ticks_ns();

  nx::vector<gfx::gpu_marker> out;
  (void)read_all(device, out);
  const gfx::gpu_marker *copy = nullptr;
  for (const gfx::gpu_marker &marker : out)
    if (marker.queue == gfx::gpu_marker_queue::Copy)
      copy = &marker;
  REQUIRE(copy != nullptr);
  CHECK_FALSE(copy->text().empty());
  CHECK(copy->frame == 0);
  constexpr u64 SLACK_NS = 50'000'000;
  CHECK(copy->begin_ns + SLACK_NS >= began);
  CHECK(copy->end_ns <= ended + SLACK_NS);
  gfx::destroy_buffer(device, buffer);
  gfx::set_gpu_markers(device, false);
}

TEST_CASE("gpu markers: copies come round again, and too many at once are "
          "counted") {
  nx::gfx::test::OwnedDevice owned;
  if (!owned.init({.application_name = "nx gpu marker tests"}))
    SKIP("no usable device");
  gfx::device &device = owned;
  if (!has_timestamps(device))
    SKIP("no timestamps");
  if (gfx::vk::queue_families(device).size() < 2)
    SKIP("no copy queue of its own");
  gfx::set_gpu_markers(device, true);
  run_and_read(device, [](gfx::command_list &) {});
  const gfx::buffer_handle buffer =
      gfx::create_buffer(device, {.name = "marker copies",
                                  .size = 16,
                                  .usage = gfx::buffer_usage::CopyDst |
                                           gfx::buffer_usage::Storage});
  REQUIRE(buffer.valid());
  const u8 bytes[16] = {};
  const auto copies = [&](const u32 count) {
    for (u32 i = 0; i < count; ++i)
      REQUIRE(gfx::upload_buffer(device, buffer, 0, bytes).ok());
    run_and_read(device, [](gfx::command_list &) {});
    gfx::wait_idle(device);
    run_and_read(device, [](gfx::command_list &) {});
  };
  const auto copies_read = [&] {
    nx::vector<gfx::gpu_marker> out;
    (void)read_all(device, out);
    usize seen = 0;
    for (const gfx::gpu_marker &marker : out)
      seen += marker.queue == gfx::gpu_marker_queue::Copy ? 1u : 0u;
    return seen;
  };

  (void)copies_read();
  usize seen = 0;
  for (int round = 0; round < 8; ++round) {
    copies(100);
    seen += copies_read();
  }
  CHECK(seen == 800);
  CHECK(gfx::gpu_markers_stats(device).groups_dropped == 0);

  copies(300);
  CHECK(copies_read() == 256);
  CHECK(gfx::gpu_markers_stats(device).groups_dropped == 44);

  gfx::destroy_buffer(device, buffer);
  gfx::set_gpu_markers(device, false);
}

TEST_CASE("gpu markers: shutting down with markers on and unread is fine") {
  nx::gfx::test::OwnedDevice owned;
  if (!owned.init({.application_name = "nx gpu marker tests"}))
    SKIP("no usable device");
  gfx::device &device = owned;
  if (!has_timestamps(device))
    SKIP("no timestamps");
  gfx::set_gpu_markers(device, true);
  run_and_read(device, [](gfx::command_list &list) {
    gfx::push_debug_group(list, "left behind");
    gfx::pop_debug_group(list);
  });
}

#endif
