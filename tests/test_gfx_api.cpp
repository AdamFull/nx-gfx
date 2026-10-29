#include "framework/nxtest.h"

#include "nx/gfx/commands.h"
#include "nx/gfx/device.h"
#include "nx/gfx/frame.h"
#include "nx/gfx/resources.h"
#include "nx/gfx/upload.h"

#include <cstring>
#include <span>
#include <vector>

// Only the nx/gfx headers: a consumer reaches the device without any of the
// backend's.

namespace {

namespace gfx = nx::gfx;

struct Device {
  gfx::device *dev = nullptr;

  Device() {
    auto made = gfx::create_device({.application_name = "nx gfx tests"});
    if (made)
      dev = made.value();
  }

  ~Device() { gfx::destroy_device(dev); }

  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
};

std::vector<u8> pattern(const u32 bytes, const u8 seed) {
  std::vector<u8> out(bytes);
  for (u32 i = 0; i < bytes; ++i)
    out[i] = nx::cast<u8>(i * 31 + seed);
  return out;
}

gfx::buffer_handle readback_buffer(gfx::device &dev, const u64 size) {
  return gfx::create_buffer(dev, {.name = "readback",
                                  .size = size,
                                  .usage = gfx::buffer_usage::CopyDst,
                                  .memory = gfx::memory_usage::Readback,
                                  .persistently_mapped = true});
}

gfx::buffer_handle source_buffer(gfx::device &dev,
                                 const std::vector<u8> &bytes) {
  const gfx::buffer_handle buffer = gfx::create_buffer(
      dev, {.name = "source",
            .size = bytes.size(),
            .usage = gfx::buffer_usage::CopySrc | gfx::buffer_usage::CopyDst});
  const gfx::upload_ticket ticket =
      gfx::upload_buffer(dev, buffer, 0, bytes, gfx::resource_state::CopySrc);
  gfx::wait_upload(dev, ticket);
  return buffer;
}

} // namespace

TEST_CASE("gfx: a device made through the free functions reports its caps") {
  Device fixture;
  if (fixture.dev == nullptr)
    SKIP("no usable device");
  gfx::device &dev = *fixture.dev;

  CHECK(!gfx::caps(dev).device_name.empty());
  CHECK(gfx::frames_in_flight(dev) > 0);
  CHECK(gfx::device_status(dev) == gfx::gpu_device_status::Ready);
}

#if defined(NX_RHI_VULKAN)
TEST_CASE("gfx: a device whose driver cannot load is refused, and says why") {
  const gfx::device_desc desc{
      .application_name = "nx gfx tests",
      .native_driver = {.resolve = [](void *, const char *) noexcept -> void * {
                          return nullptr;
                        },
                        .identity = "empty driver"}};
  auto made = gfx::create_device(desc);
  REQUIRE(!made);
  CHECK(made.error().report.failure == gfx::device_init_failure::Failed);
  CHECK(nx::string_view(made.error().message).find("empty driver") !=
        nx::string_view::npos);
}
#endif

TEST_CASE("gfx: buffer bytes go up and come back") {
  Device fixture;
  if (fixture.dev == nullptr)
    SKIP("no usable device");
  gfx::device &dev = *fixture.dev;

  const std::vector<u8> bytes = pattern(4096, 7);
  const gfx::buffer_handle buffer = source_buffer(dev, bytes);
  REQUIRE(buffer.valid());

  const gfx::readback_result read =
      gfx::read_buffer(dev, buffer, 0, bytes.size());
  gfx::wait_upload(dev, read.ticket);
  REQUIRE(gfx::upload_complete(dev, read.ticket));
  REQUIRE(read.size == bytes.size());
  CHECK(std::memcmp(read.data, bytes.data(), bytes.size()) == 0);
  CHECK(gfx::upload_stats(dev).total_submits > 0);

  gfx::destroy_buffer(dev, buffer);
}

TEST_CASE("gfx: uploads that overrun the staging ring wait for it and land") {
  Device fixture;
  if (fixture.dev == nullptr)
    SKIP("no usable device");
  gfx::device &dev = *fixture.dev;

  // Under half the ring each, so they stay on it, and four of them in one go
  // so later ones wait for the ring to come back.
  const u32 bytes = 12u * 1024u * 1024u;
  REQUIRE(gfx::upload_stats(dev).ring_capacity < 3u * bytes);
  std::vector<gfx::buffer_handle> buffers;
  std::vector<std::vector<u8>> patterns;
  gfx::upload_ticket last;
  for (u8 i = 0; i < 4; ++i) {
    patterns.push_back(pattern(bytes, nx::cast<u8>(i * 53 + 1)));
    buffers.push_back(gfx::create_buffer(
        dev,
        {.name = "ring overrun",
         .size = bytes,
         .usage = gfx::buffer_usage::CopySrc | gfx::buffer_usage::CopyDst}));
    REQUIRE(buffers.back().valid());
    last = gfx::upload_buffer(dev, buffers.back(), 0, patterns.back(),
                              gfx::resource_state::CopySrc);
    REQUIRE(last.valid());
  }
  gfx::wait_upload(dev, last);
  CHECK(gfx::upload_stats(dev).extra_buffers == 0u);

  for (usize i = 0; i < buffers.size(); ++i) {
    const gfx::readback_result read =
        gfx::read_buffer(dev, buffers[i], 0, bytes);
    gfx::wait_upload(dev, read.ticket);
    REQUIRE(read.size == bytes);
    CHECK(std::memcmp(read.data, patterns[i].data(), bytes) == 0);
    gfx::destroy_buffer(dev, buffers[i]);
  }
}

TEST_CASE("gfx: a headless frame's list copies one buffer into another") {
  Device fixture;
  if (fixture.dev == nullptr)
    SKIP("no usable device");
  gfx::device &dev = *fixture.dev;

  const std::vector<u8> bytes = pattern(1024, 3);
  const gfx::buffer_handle source = source_buffer(dev, bytes);
  const gfx::buffer_handle target = readback_buffer(dev, bytes.size());
  REQUIRE(source.valid());
  REQUIRE(target.valid());

  gfx::command_list *cmd = gfx::begin_headless_frame(dev);
  REQUIRE(cmd != nullptr);
  const gfx::buffer_barrier to_copy{.buffer = target,
                                    .to = gfx::resource_state::CopyDst};
  gfx::barrier(*cmd, nullptr, 0, &to_copy, 1);
  CHECK(gfx::copy_buffer(*cmd, source, 0, target, 0, bytes.size()));
  REQUIRE(gfx::end_headless_frame(dev));

  gfx::invalidate_buffer(dev, target);
  CHECK(std::memcmp(gfx::buffer_mapped(dev, target), bytes.data(),
                    bytes.size()) == 0);

  gfx::destroy_buffer(dev, target);
  gfx::destroy_buffer(dev, source);
}

TEST_CASE("gfx: each list of a split is its own, and a frame splits twice") {
  Device fixture;
  if (fixture.dev == nullptr)
    SKIP("no usable device");
  gfx::device &dev = *fixture.dev;
  if (!gfx::caps(dev).parallel_recording)
    SKIP("the device records on one thread");

  constexpr u32 PART = 256;
  constexpr u32 PARTS = 5;
  const std::vector<u8> bytes = pattern(PART * PARTS, 11);
  const gfx::buffer_handle source = source_buffer(dev, bytes);
  const gfx::buffer_handle target = readback_buffer(dev, bytes.size());
  REQUIRE(source.valid());
  REQUIRE(target.valid());

  gfx::command_list *frame = gfx::begin_headless_frame(dev);
  REQUIRE(frame != nullptr);
  const gfx::buffer_barrier to_copy{.buffer = target,
                                    .to = gfx::resource_state::CopyDst};
  gfx::barrier(*frame, nullptr, 0, &to_copy, 1);

  // Three lists, then two more in a second split: one part each.
  u32 part = 0;
  for (const u32 count : {3u, 2u}) {
    const gfx::recording_split split = gfx::split_frame(dev, *frame, count);
    REQUIRE(split.count == count);
    CHECK(gfx::begin_recording(dev, split, count) == nullptr);
    std::vector<gfx::command_list *> lists;
    for (u32 i = 0; i < count; ++i) {
      gfx::command_list *list = gfx::begin_recording(dev, split, i);
      REQUIRE(list != nullptr);
      CHECK(list != frame);
      for (gfx::command_list *other : lists)
        CHECK(list != other);
      lists.push_back(list);
    }
    for (gfx::command_list *list : lists) {
      CHECK(gfx::copy_buffer(*list, source, u64{part} * PART, target,
                             u64{part} * PART, PART));
      gfx::end_recording(dev, *list);
      ++part;
    }
  }
  REQUIRE(gfx::end_headless_frame(dev));

  gfx::invalidate_buffer(dev, target);
  CHECK(std::memcmp(gfx::buffer_mapped(dev, target), bytes.data(),
                    bytes.size()) == 0);

  gfx::destroy_buffer(dev, target);
  gfx::destroy_buffer(dev, source);
}
