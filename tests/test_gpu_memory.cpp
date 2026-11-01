#include "framework/nxtest.h"

#include "nx/gfx/gfx.h"
#include "owned_device.h"

#include <cstring>

#if defined(NX_RHI_VULKAN)

namespace gfx = nx::gfx;

namespace {

[[nodiscard]] gfx::buffer_handle some_buffer(gfx::device &device,
                                             const char *const name,
                                             const u64 size = 4096) {
  return gfx::create_buffer(
      device,
      {.name = name, .size = size, .usage = gfx::buffer_usage::Storage});
}

/// Enough frames that what was destroyed is really gone.
void settle(gfx::device &device) {
  gfx::wait_idle(device);
  for (int i = 0; i < 6; ++i) {
    REQUIRE(gfx::begin_headless_frame(device) != nullptr);
    REQUIRE(gfx::end_headless_frame(device));
  }
  gfx::wait_idle(device);
}

[[nodiscard]] nx::vector<gfx::gpu_resource_event>
read_events(gfx::device &device) {
  nx::vector<gfx::gpu_resource_event> out(gfx::GPU_RESOURCE_EVENTS_HELD);
  out.resize(gfx::read_resource_events(device, {out.data(), out.size()}));
  return out;
}

} // namespace

TEST_CASE("gpu memory: heaps have a budget and buffers and textures are "
          "counted while they live") {
  nx::gfx::test::OwnedDevice owned;
  if (!owned.init({.application_name = "nx gpu memory tests"}))
    SKIP("no usable device");
  gfx::device &device = owned;

  const gfx::gpu_memory_stats before = gfx::memory_stats(device);
  REQUIRE(before.heap_count > 0);
  REQUIRE(before.heap_count <= gfx::GPU_MOST_HEAPS);
  bool device_local = false;
  for (u32 i = 0; i < before.heap_count; ++i) {
    CHECK(before.heaps[i].size > 0);
    CHECK(before.heaps[i].budget > 0);
    device_local = device_local || before.heaps[i].device_local;
  }
  CHECK(device_local);

  constexpr u64 SIZE = 1024 * 1024;
  const gfx::buffer_handle buffer = some_buffer(device, "counted", SIZE);
  REQUIRE(buffer.valid());
  const gfx::texture_handle texture =
      gfx::create_texture(device, {.name = "counted texture",
                                   .format = gfx::format::RGBA8_UNORM,
                                   .width = 64,
                                   .height = 64,
                                   .usage = gfx::texture_usage::Sampled});
  REQUIRE(texture.valid());

  const gfx::gpu_memory_stats alive = gfx::memory_stats(device);
  CHECK(alive.buffers.count == before.buffers.count + 1);
  CHECK(alive.buffers.bytes >= before.buffers.bytes + SIZE);
  CHECK(alive.textures.count == before.textures.count + 1);
  CHECK(alive.textures.bytes >= before.textures.bytes + 64 * 64 * 4);

  gfx::destroy_buffer(device, buffer);
  gfx::destroy_texture(device, texture);
  settle(device);
  const gfx::gpu_memory_stats after = gfx::memory_stats(device);
  CHECK(after.buffers.count == before.buffers.count);
  CHECK(after.buffers.bytes == before.buffers.bytes);
  CHECK(after.textures.count == before.textures.count);
  CHECK(after.textures.bytes == before.textures.bytes);
}

TEST_CASE("gpu memory: a resource's creation and destruction pair up") {
  nx::gfx::test::OwnedDevice owned;
  if (!owned.init({.application_name = "nx gpu memory tests"}))
    SKIP("no usable device");
  gfx::device &device = owned;

  // Off to begin with.
  const gfx::buffer_handle unseen = some_buffer(device, "unseen");
  REQUIRE(unseen.valid());
  CHECK(read_events(device).empty());
  CHECK(gfx::resource_events_dropped(device) == 0);

  gfx::set_resource_events(device, true);
  const gfx::buffer_handle buffer = some_buffer(device, "seen");
  REQUIRE(buffer.valid());
  nx::vector<gfx::gpu_resource_event> events = read_events(device);
  REQUIRE(events.size() == 1);
  CHECK(events[0].created);
  CHECK(events[0].kind == gfx::gpu_resource_kind::Buffer);
  CHECK(events[0].text() == "seen");
  CHECK(events[0].bytes >= 4096);
  const u64 id = events[0].id;

  gfx::destroy_buffer(device, buffer);
  gfx::destroy_buffer(device, unseen);
  // Not yet: the GPU may still be using them.
  CHECK(read_events(device).empty());
  settle(device);
  events = read_events(device);
  REQUIRE(events.size() == 2);
  bool paired = false;
  bool without_a_creation = false;
  for (const gfx::gpu_resource_event &event : events) {
    CHECK_FALSE(event.created);
    paired = paired || (event.id == id && event.text() == "seen");
    without_a_creation =
        without_a_creation || (event.id != id && event.text() == "unseen");
  }
  CHECK(paired);
  CHECK(without_a_creation);

  gfx::set_resource_events(device, false);
  const gfx::buffer_handle after = some_buffer(device, "after");
  REQUIRE(after.valid());
  CHECK(read_events(device).empty());
  gfx::destroy_buffer(device, after);
}

TEST_CASE("gpu memory: textures and memory heaps have events too, placed "
          "textures with no bytes of their own") {
  nx::gfx::test::OwnedDevice owned;
  if (!owned.init({.application_name = "nx gpu memory tests"}))
    SKIP("no usable device");
  gfx::device &device = owned;
  gfx::set_resource_events(device, true);

  const gfx::texture_handle texture =
      gfx::create_texture(device, {.name = "evented texture",
                                   .format = gfx::format::RGBA8_UNORM,
                                   .width = 16,
                                   .height = 16,
                                   .usage = gfx::texture_usage::Sampled});
  REQUIRE(texture.valid());
  nx::vector<gfx::gpu_resource_event> events = read_events(device);
  REQUIRE(events.size() == 1);
  CHECK(events[0].kind == gfx::gpu_resource_kind::Texture);
  CHECK(events[0].bytes >= 16 * 16 * 4);

  gfx::destroy_texture(device, texture);
  settle(device);
  events = read_events(device);
  REQUIRE(events.size() == 1);
  CHECK_FALSE(events[0].created);
  CHECK(events[0].text() == "evented texture");
  CHECK(events[0].bytes >= 16 * 16 * 4);
  gfx::set_resource_events(device, false);
}

TEST_CASE("gpu memory: too many events unread are counted, and reads take "
          "what fits") {
  nx::gfx::test::OwnedDevice owned;
  if (!owned.init({.application_name = "nx gpu memory tests"}))
    SKIP("no usable device");
  gfx::device &device = owned;
  gfx::set_resource_events(device, true);
  CHECK(gfx::resource_events_dropped(device) == 0);

  nx::vector<gfx::buffer_handle> buffers;
  for (usize i = 0; i < gfx::GPU_RESOURCE_EVENTS_HELD + 5; ++i) {
    const gfx::buffer_handle buffer = some_buffer(device, "many", 16);
    REQUIRE(buffer.valid());
    buffers.push_back(buffer);
  }
  CHECK(gfx::resource_events_dropped(device) == 5);

  gfx::gpu_resource_event two[2];
  CHECK(gfx::read_resource_events(device, two) == 2);
  CHECK(read_events(device).size() == gfx::GPU_RESOURCE_EVENTS_HELD - 2);
  CHECK(read_events(device).empty());

  for (const gfx::buffer_handle buffer : buffers)
    gfx::destroy_buffer(device, buffer);
  gfx::set_resource_events(device, false);
}

TEST_CASE("gpu memory: a long name is cut, and a name may be empty") {
  nx::gfx::test::OwnedDevice owned;
  if (!owned.init({.application_name = "nx gpu memory tests"}))
    SKIP("no usable device");
  gfx::device &device = owned;
  gfx::set_resource_events(device, true);
  char long_name[200];
  std::memset(long_name, 'x', sizeof(long_name));
  long_name[199] = 0;
  const gfx::buffer_handle cut = some_buffer(device, long_name);
  const gfx::buffer_handle empty = some_buffer(device, "");
  REQUIRE(cut.valid());
  REQUIRE(empty.valid());
  const nx::vector<gfx::gpu_resource_event> events = read_events(device);
  REQUIRE(events.size() == 2);
  CHECK(events[0].text().size() == gfx::GPU_RESOURCE_NAME_BYTES);
  CHECK(events[1].text().empty());
  gfx::destroy_buffer(device, cut);
  gfx::destroy_buffer(device, empty);
  settle(device);
  gfx::set_resource_events(device, false);
}

TEST_CASE("gpu memory: uploads add up") {
  nx::gfx::test::OwnedDevice owned;
  if (!owned.init({.application_name = "nx gpu memory tests"}))
    SKIP("no usable device");
  gfx::device &device = owned;
  const gfx::buffer_handle buffer =
      gfx::create_buffer(device, {.name = "uploaded",
                                  .size = 4096,
                                  .usage = gfx::buffer_usage::CopyDst |
                                           gfx::buffer_usage::Storage});
  REQUIRE(buffer.valid());
  const u64 before = gfx::upload_stats(device).bytes_uploaded;
  const u8 bytes[1000] = {};
  REQUIRE(gfx::upload_buffer(device, buffer, 0, bytes).ok());
  REQUIRE(gfx::upload_buffer(device, buffer, 1000, bytes).ok());
  CHECK(gfx::upload_stats(device).bytes_uploaded == before + 2000);
  settle(device);
  gfx::destroy_buffer(device, buffer);
}

#endif
