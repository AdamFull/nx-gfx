#include "bvh_scenes.h"
#include "framework/nxtest.h"
#include "pipeline_test_utils.h"

#include "core/foundation/diagnostics/log.h"
#include "core/foundation/platform/filesystem.h"
#include "nx/gfx/gfx.h"
#include "owned_device.h"
#include "validation_errors.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <span>

namespace gfx = nx::gfx;
using state = gfx::resource_state;

namespace {

struct Hit {
  float t, u, v, instance;
};

struct Push {
  u64 scene = 0;
  u64 out = 0;
  u32 accept = 0;
  u32 pad = 0;
};

/// One triangle at z 0, and the same moved 1 along z.
constexpr float POSITIONS[9] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
constexpr float MOVED[9] = {0, 0, 1, 1, 0, 1, 0, 1, 1};
constexpr u32 INDICES[3] = {0, 1, 2};
constexpr u32 CUSTOM_INDEX = 7;

[[nodiscard]] u64 align_up(const u64 value, const u64 alignment) {
  return alignment <= 1 ? value
                        : (value + alignment - 1) / alignment * alignment;
}

template <typename T>
[[nodiscard]] std::span<const u8> bytes_of(const T &value) noexcept {
  return {reinterpret_cast<const u8 *>(&value), sizeof(value)};
}

/// Uploads @p bytes into @p buffer and leaves them for builds to read.
void upload_input(gfx::device &device, const gfx::buffer_handle buffer,
                  const std::span<const u8> bytes) {
  gfx::wait_upload(device,
                   gfx::upload_buffer(device, buffer, 0, bytes,
                                      state::AccelerationStructureInput));
}

[[nodiscard]] gfx::buffer_handle input_buffer(gfx::device &device,
                                              const nx::string_view name,
                                              const std::span<const u8> bytes) {
  const gfx::buffer_handle buffer = gfx::create_buffer(
      device,
      {.name = name,
       .size = bytes.size(),
       .usage = gfx::buffer_usage::AccelerationStructureInput |
                gfx::buffer_usage::DeviceAddress | gfx::buffer_usage::CopyDst});
  if (buffer.valid())
    upload_input(device, buffer, bytes);
  return buffer;
}

void transition(gfx::command_list &cmd, const gfx::buffer_handle buffer,
                const state from, const state to) {
  const gfx::buffer_barrier barrier{.buffer = buffer, .from = from, .to = to};
  gfx::barrier(cmd, nullptr, 0, &barrier, 1);
}

/// A one-triangle bottom level of @p flags under one instance moved 2 along
/// z, with the scratch every build and update shares: all made, none built.
struct Scene {
  gfx::device &device;
  gfx::acceleration_structure_flags flags;
  gfx::buffer_handle vertices;
  gfx::buffer_handle indices;
  gfx::buffer_handle instances;
  gfx::buffer_handle scratch;
  gfx::acceleration_structure_handle bottom;
  gfx::acceleration_structure_handle top;
  gfx::ray_triangles triangle;
  gfx::top_level_desc top_desc;
  u64 scratch_address = 0;

  Scene(gfx::device &dev,
        const gfx::acceleration_structure_flags bottom_flags =
            gfx::acceleration_structure_flag::PreferFastTrace,
        const gfx::ray_geometry_flags geometry = gfx::ray_geometry_flag::Opaque)
      : device(dev), flags(bottom_flags) {
    vertices = input_buffer(device, "ray vertices", bytes_of(POSITIONS));
    indices = input_buffer(device, "ray indices", bytes_of(INDICES));
    triangle = {.vertex_address = gfx::buffer_address(device, vertices),
                .vertex_count = 3,
                .index_address = gfx::buffer_address(device, indices),
                .triangle_count = 1,
                .flags = geometry};
    const gfx::acceleration_structure_sizes bottom_sizes =
        gfx::sizes(device, bottom_desc());
    bottom = gfx::create_acceleration_structure(
        device, {.name = "ray bottom",
                 .kind = gfx::acceleration_structure_kind::Bottom,
                 .size = bottom_sizes.structure,
                 .flags = flags});

    instances = instance_buffer(bottom);
    top_desc = {.instance_address = gfx::buffer_address(device, instances),
                .instance_count = 1};
    const gfx::acceleration_structure_sizes top_sizes =
        gfx::sizes(device, top_desc);
    top = gfx::create_acceleration_structure(
        device, {.name = "ray top",
                 .kind = gfx::acceleration_structure_kind::Top,
                 .size = top_sizes.structure});

    const u64 alignment =
        gfx::caps(device).acceleration_structure_scratch_alignment;
    const u64 most = nx::max(
        nx::max(bottom_sizes.build_scratch, bottom_sizes.update_scratch),
        top_sizes.build_scratch);
    scratch =
        gfx::create_buffer(device, {.name = "ray scratch",
                                    .size = most + alignment,
                                    .usage = gfx::buffer_usage::Storage |
                                             gfx::buffer_usage::DeviceAddress});
    scratch_address = align_up(gfx::buffer_address(device, scratch), alignment);
  }

  ~Scene() {
    gfx::destroy_acceleration_structure(device, top);
    gfx::destroy_acceleration_structure(device, bottom);
    gfx::destroy_buffer(device, scratch);
    gfx::destroy_buffer(device, instances);
    gfx::destroy_buffer(device, indices);
    gfx::destroy_buffer(device, vertices);
  }

  /// One instance of @p structure moved 2 along z.
  [[nodiscard]] gfx::buffer_handle
  instance_buffer(const gfx::acceleration_structure_handle structure) const {
    gfx::ray_instance placed = gfx::make_ray_instance(
        gfx::acceleration_structure_address(device, structure), CUSTOM_INDEX);
    placed.row2.w = 2.0f;
    return input_buffer(device, "ray instances", bytes_of(placed));
  }

  [[nodiscard]] gfx::bottom_level_desc bottom_desc() const {
    return {.geometries = {&triangle, 1}, .flags = flags};
  }

  [[nodiscard]] bool made() const {
    return vertices.valid() && indices.valid() && instances.valid() &&
           scratch.valid() && bottom.valid() && top.valid();
  }

  [[nodiscard]] gfx::buffer_handle
  storage(const gfx::acceleration_structure_handle structure) const {
    return gfx::acceleration_structure_buffer(device, structure);
  }

  /// Records both builds from nothing, each behind what it needs, and leaves
  /// both levels readable.
  [[nodiscard]] bool build(gfx::command_list &cmd) const {
    transition(cmd, scratch, state::Undefined,
               state::AccelerationStructureBuild);
    transition(cmd, storage(bottom), state::Undefined,
               state::AccelerationStructureBuild);
    if (!gfx::build_acceleration_structure(cmd, bottom, bottom_desc(),
                                           scratch_address))
      return false;
    transition(cmd, storage(bottom), state::AccelerationStructureBuild,
               state::AccelerationStructureRead);
    return build_top(cmd, state::Undefined);
  }

  /// Records building the top level again over what its instances name,
  /// its storage last in @p from.
  [[nodiscard]] bool build_top(gfx::command_list &cmd, const state from) const {
    transition(cmd, scratch, state::AccelerationStructureBuild,
               state::AccelerationStructureBuild);
    transition(cmd, storage(top), from, state::AccelerationStructureBuild);
    if (!gfx::build_acceleration_structure(cmd, top, top_desc, scratch_address))
      return false;
    transition(cmd, storage(top), state::AccelerationStructureBuild,
               state::AccelerationStructureRead);
    return true;
  }
};

/// The trace shader's pipeline and where it writes its two hits.
struct Tracer {
  gfx::device &device;
  gfx::shader_handle shader;
  gfx::pipeline_handle pipeline;
  gfx::buffer_handle out;

  explicit Tracer(gfx::device &dev) : device(dev) {
    const nx::string base = nx::string(NX_GFX_TEST_SHADER_DIR) + "/ray_trace";
    auto code = nx::fs::file_read(nx::fs::path_view(base + ".spv"));
    auto refl = nx::fs::file_read_text(nx::fs::path_view(base + ".refl.json"));
    if (!code.has_value() || !refl.has_value())
      return;
    shader = nx::gfx::test::create_shader(
        device,
        {.name = "ray trace", .code = code->data(), .code_size = code->size()},
        refl->view());
    if (!shader.valid())
      return;
    pipeline = nx::gfx::test::compile_pipeline(
        device,
        gfx::compute_pipeline_desc{
            .name = "ray trace",
            .compute = {.shader = shader, .entry_point = "trace_main"}});
    out = gfx::create_buffer(device, {.name = "ray hits",
                                      .size = 2 * sizeof(Hit),
                                      .usage = gfx::buffer_usage::Storage |
                                               gfx::buffer_usage::DeviceAddress,
                                      .memory = gfx::memory_usage::Readback,
                                      .persistently_mapped = true});
  }
  ~Tracer() {
    gfx::destroy_buffer(device, out);
    gfx::destroy_pipeline(device, pipeline);
    gfx::destroy_shader(device, shader);
  }

  [[nodiscard]] bool ready() const { return pipeline.valid() && out.valid(); }

  /// A shade that, where @p accept, counts every triangle not made opaque.
  void record(gfx::command_list &cmd,
              const gfx::acceleration_structure_handle top,
              const bool accept = true) const {
    std::memset(gfx::buffer_mapped(device, out), 0, 2 * sizeof(Hit));
    const Push push{.scene = gfx::acceleration_structure_address(device, top),
                    .out = gfx::buffer_address(device, out),
                    .accept = accept ? 1u : 0u};
    gfx::bind_pipeline(cmd, pipeline);
    gfx::push_constants(cmd, &push, sizeof(push));
    gfx::dispatch(cmd, 1);
  }

  /// After the frame that recorded it has finished.
  [[nodiscard]] std::array<Hit, 2> hits() const {
    std::array<Hit, 2> got{};
    std::memcpy(got.data(), gfx::buffer_mapped(device, out), sizeof(got));
    return got;
  }
};

/// A device with validation on, counting what the layer reports.
struct RayFixture {
  nx::gfx::test::OwnedDevice device;
  nx::gfx::test::ValidationErrors *errors = nullptr;
  bool ready = false;

  explicit RayFixture(const gfx::route_flags disabled = gfx::route::None) {
    errors = nx::allocate<nx::gfx::test::ValidationErrors>();
    nx::log::add_sink(errors);
    ready = device.init(
        {.application_name = "nx ray tests", .disable_routes = disabled});
  }
  ~RayFixture() {
    device.shutdown();
    nx::log::flush();
    nx::log::remove_sink(errors);
  }
  [[nodiscard]] u32 validation_errors() const {
    nx::log::flush();
    return errors->count();
  }
};

void check_hit(const Hit &hit, const float t) {
  CHECK(hit.t == t);
  CHECK(hit.u == 0.25f);
  CHECK(hit.v == 0.25f);
  CHECK(hit.instance == float(CUSTOM_INDEX));
}

} // namespace

TEST_CASE("gfx ray: a top level over an instanced bottom level is traced "
          "through its address") {
  RayFixture fixture;
  if (!fixture.ready)
    SKIP("no usable device");
  gfx::device &device = fixture.device;
  if (!gfx::caps(device).ray_queries)
    SKIP("the device reports no ray queries");
  CHECK(gfx::caps(device).acceleration_structure_scratch_alignment > 0u);
  CHECK(gfx::caps(device).max_ray_instances > 0u);
  {
    Tracer tracer(device);
    if (!tracer.ready())
      SKIP("test shaders are not built in this configuration");
    Scene scene(device);
    REQUIRE(scene.made());
    CHECK(gfx::acceleration_structure_address(device, scene.top) != 0u);
    CHECK(gfx::acceleration_structure_address(device, scene.bottom) !=
          gfx::acceleration_structure_address(device, scene.top));

    gfx::command_list *cmd = nullptr;
    REQUIRE(((cmd = gfx::begin_headless_frame(device)) != nullptr));
    REQUIRE(scene.build(*cmd));
    tracer.record(*cmd, scene.top);
    REQUIRE(gfx::end_headless_frame(device));
    gfx::wait_idle(device);

    const std::array<Hit, 2> got = tracer.hits();
    check_hit(got[0], 3.0f);
    CHECK(got[1].t == -1.0f);
  }
  gfx::wait_idle(device);
  CHECK(fixture.validation_errors() == 0u);
}

TEST_CASE("gfx ray: a triangle not made opaque counts only where the shade "
          "accepts it") {
  RayFixture fixture;
  if (!fixture.ready)
    SKIP("no usable device");
  gfx::device &device = fixture.device;
  if (!gfx::caps(device).ray_queries)
    SKIP("the device reports no ray queries");
  {
    Tracer tracer(device);
    if (!tracer.ready())
      SKIP("test shaders are not built in this configuration");
    Scene masked(device, gfx::acceleration_structure_flag::PreferFastTrace,
                 gfx::ray_geometry_flag::None);
    Scene opaque(device);
    REQUIRE(masked.made());
    REQUIRE(opaque.made());

    gfx::command_list *cmd = nullptr;
    REQUIRE(((cmd = gfx::begin_headless_frame(device)) != nullptr));
    REQUIRE(masked.build(*cmd));
    REQUIRE(opaque.build(*cmd));
    tracer.record(*cmd, masked.top, false);
    REQUIRE(gfx::end_headless_frame(device));
    gfx::wait_idle(device);
    CHECK(tracer.hits()[0].t == -1.0f);

    REQUIRE(((cmd = gfx::begin_headless_frame(device)) != nullptr));
    tracer.record(*cmd, masked.top, true);
    REQUIRE(gfx::end_headless_frame(device));
    gfx::wait_idle(device);
    check_hit(tracer.hits()[0], 3.0f);

    REQUIRE(((cmd = gfx::begin_headless_frame(device)) != nullptr));
    tracer.record(*cmd, opaque.top, false);
    REQUIRE(gfx::end_headless_frame(device));
    gfx::wait_idle(device);
    check_hit(tracer.hits()[0], 3.0f);
  }
  gfx::wait_idle(device);
  CHECK(fixture.validation_errors() == 0u);
}

TEST_CASE("gfx ray: an update in place moves what the ray hits") {
  RayFixture fixture;
  if (!fixture.ready)
    SKIP("no usable device");
  gfx::device &device = fixture.device;
  if (!gfx::caps(device).ray_queries)
    SKIP("the device reports no ray queries");
  {
    Tracer tracer(device);
    if (!tracer.ready())
      SKIP("test shaders are not built in this configuration");
    Scene scene(device, gfx::acceleration_structure_flag::PreferFastTrace |
                            gfx::acceleration_structure_flag::AllowUpdate);
    REQUIRE(scene.made());

    gfx::command_list *cmd = nullptr;
    REQUIRE(((cmd = gfx::begin_headless_frame(device)) != nullptr));
    REQUIRE(scene.build(*cmd));
    tracer.record(*cmd, scene.top);
    REQUIRE(gfx::end_headless_frame(device));
    gfx::wait_idle(device);
    check_hit(tracer.hits()[0], 3.0f);

    upload_input(device, scene.vertices, bytes_of(MOVED));
    REQUIRE(((cmd = gfx::begin_headless_frame(device)) != nullptr));
    transition(*cmd, scene.scratch, state::AccelerationStructureBuild,
               state::AccelerationStructureBuild);
    transition(*cmd, scene.storage(scene.bottom),
               state::AccelerationStructureRead,
               state::AccelerationStructureBuild);
    REQUIRE(gfx::update_acceleration_structure(
        *cmd, scene.bottom, scene.bottom_desc(), scene.scratch_address));
    transition(*cmd, scene.storage(scene.bottom),
               state::AccelerationStructureBuild,
               state::AccelerationStructureRead);
    REQUIRE(scene.build_top(*cmd, state::AccelerationStructureRead));
    tracer.record(*cmd, scene.top);
    REQUIRE(gfx::end_headless_frame(device));
    gfx::wait_idle(device);

    const std::array<Hit, 2> got = tracer.hits();
    check_hit(got[0], 4.0f);
    CHECK(got[1].t == -1.0f);
  }
  gfx::wait_idle(device);
  CHECK(fixture.validation_errors() == 0u);
}

TEST_CASE("gfx ray: a compacted copy is no larger and traces the same") {
  RayFixture fixture;
  if (!fixture.ready)
    SKIP("no usable device");
  gfx::device &device = fixture.device;
  if (!gfx::caps(device).ray_queries)
    SKIP("the device reports no ray queries");
  {
    Tracer tracer(device);
    if (!tracer.ready())
      SKIP("test shaders are not built in this configuration");
    constexpr gfx::acceleration_structure_flags FLAGS =
        gfx::acceleration_structure_flag::PreferFastTrace |
        gfx::acceleration_structure_flag::AllowCompaction;
    Scene scene(device, FLAGS);
    REQUIRE(scene.made());
    const gfx::query_pool_handle pool =
        gfx::create_compacted_size_pool(device, 1);
    REQUIRE(pool.valid());

    gfx::command_list *cmd = nullptr;
    REQUIRE(((cmd = gfx::begin_headless_frame(device)) != nullptr));
    gfx::reset_queries(*cmd, pool, 0, 1);
    REQUIRE(scene.build(*cmd));
    gfx::write_compacted_sizes(*cmd, {&scene.bottom, 1}, pool);
    REQUIRE(gfx::end_headless_frame(device));
    gfx::wait_idle(device);

    u64 compacted = 0;
    REQUIRE(gfx::read_compacted_sizes(device, pool, {&compacted, 1}));
    u64 timestamp = 0;
    CHECK_FALSE(gfx::read_timestamps(device, pool, {&timestamp, 1}));
    const u64 built = gfx::sizes(device, scene.bottom_desc()).structure;
    CHECK(compacted > 0u);
    CHECK(compacted <= built);

    const gfx::acceleration_structure_handle small =
        gfx::create_acceleration_structure(
            device,
            {.name = "ray compacted", .size = compacted, .flags = FLAGS});
    REQUIRE(small.valid());
    const gfx::buffer_handle first_instances = scene.instances;
    scene.instances = scene.instance_buffer(small);
    scene.top_desc.instance_address =
        gfx::buffer_address(device, scene.instances);

    REQUIRE(((cmd = gfx::begin_headless_frame(device)) != nullptr));
    transition(*cmd, scene.storage(small), state::Undefined,
               state::AccelerationStructureBuild);
    REQUIRE(gfx::copy_acceleration_structure(*cmd, scene.bottom, small, true));
    transition(*cmd, scene.storage(small), state::AccelerationStructureBuild,
               state::AccelerationStructureRead);
    REQUIRE(scene.build_top(*cmd, state::AccelerationStructureRead));
    tracer.record(*cmd, scene.top);
    gfx::destroy_acceleration_structure(device, scene.bottom);
    scene.bottom = small;
    REQUIRE(gfx::end_headless_frame(device));
    gfx::wait_idle(device);

    check_hit(tracer.hits()[0], 3.0f);
    gfx::destroy_buffer(device, first_instances);
    gfx::destroy_query_pool(device, pool);
  }
  gfx::wait_idle(device);
  CHECK(fixture.validation_errors() == 0u);
}

TEST_CASE("gfx ray: builds that cannot work are refused and record nothing") {
  RayFixture fixture;
  if (!fixture.ready)
    SKIP("no usable device");
  gfx::device &device = fixture.device;
  if (!gfx::caps(device).acceleration_structures)
    SKIP("the device has no acceleration structures");
  {
    Scene scene(device);
    REQUIRE(scene.made());
    const gfx::acceleration_structure_sizes none =
        gfx::sizes(device, gfx::bottom_level_desc{});
    CHECK(none.structure == 0u);
    CHECK(none.build_scratch == 0u);
    CHECK(gfx::sizes(device, gfx::top_level_desc{}).structure == 0u);
    CHECK_FALSE(
        gfx::create_acceleration_structure(device, {.size = 0}).valid());

    const gfx::acceleration_structure_handle small =
        gfx::create_acceleration_structure(
            device,
            {.name = "ray too small",
             .size = nx::max<u64>(
                 gfx::sizes(device, scene.bottom_desc()).structure / 4, 256)});
    REQUIRE(small.valid());
    const gfx::query_pool_handle timestamps =
        gfx::create_timestamp_pool(device, 1);

    gfx::command_list *cmd = nullptr;
    REQUIRE(((cmd = gfx::begin_headless_frame(device)) != nullptr));
    CHECK_FALSE(gfx::build_acceleration_structure(
        *cmd, scene.top, scene.bottom_desc(), scene.scratch_address));
    CHECK_FALSE(gfx::build_acceleration_structure(
        *cmd, scene.bottom, scene.top_desc, scene.scratch_address));
    CHECK_FALSE(gfx::build_acceleration_structure(
        *cmd, scene.bottom, scene.bottom_desc(), scene.scratch_address + 4));
    CHECK_FALSE(gfx::build_acceleration_structure(*cmd, scene.bottom,
                                                  scene.bottom_desc(), 0));
    CHECK_FALSE(gfx::build_acceleration_structure(
        *cmd, small, scene.bottom_desc(), scene.scratch_address));
    gfx::bottom_level_desc other_flags = scene.bottom_desc();
    other_flags.flags = gfx::acceleration_structure_flag::PreferFastBuild;
    CHECK_FALSE(gfx::build_acceleration_structure(
        *cmd, scene.bottom, other_flags, scene.scratch_address));
    CHECK_FALSE(gfx::update_acceleration_structure(
        *cmd, scene.bottom, scene.bottom_desc(), scene.scratch_address));
    gfx::top_level_desc unaligned = scene.top_desc;
    unaligned.instance_address += 4;
    CHECK_FALSE(gfx::build_acceleration_structure(*cmd, scene.top, unaligned,
                                                  scene.scratch_address));
    CHECK_FALSE(gfx::build_acceleration_structure(
        *cmd, gfx::acceleration_structure_handle{}, scene.bottom_desc(),
        scene.scratch_address));
    CHECK_FALSE(
        gfx::copy_acceleration_structure(*cmd, scene.bottom, small, true));
    CHECK_FALSE(
        gfx::copy_acceleration_structure(*cmd, scene.bottom, small, false));
    CHECK_FALSE(
        gfx::copy_acceleration_structure(*cmd, scene.bottom, scene.top, false));
    if (timestamps.valid())
      gfx::write_compacted_sizes(*cmd, {&scene.bottom, 1}, timestamps);
    REQUIRE(gfx::end_headless_frame(device));
    gfx::destroy_acceleration_structure(device, small);
    gfx::destroy_query_pool(device, timestamps);
  }
  gfx::wait_idle(device);
  CHECK(fixture.validation_errors() == 0u);
}

TEST_CASE("gfx ray: a structure destroyed under a hold outlives an idle GPU "
          "until the held work is released") {
  RayFixture fixture;
  if (!fixture.ready)
    SKIP("no usable device");
  gfx::device &device = fixture.device;
  if (!gfx::caps(device).acceleration_structures)
    SKIP("the device has no acceleration structures");

  const auto alive = [&](const gfx::acceleration_structure_handle structure) {
    return gfx::acceleration_structure_address(device, structure) != 0u &&
           gfx::acceleration_structure_buffer(device, structure).valid();
  };
  const gfx::acceleration_structure_handle held =
      gfx::create_acceleration_structure(device,
                                         {.name = "ray held", .size = 4096});
  REQUIRE(alive(held));
  const gfx::buffer_handle storage =
      gfx::acceleration_structure_buffer(device, held);

  gfx::hold_retirements(device, 2);
  gfx::destroy_acceleration_structure(device, held);
  gfx::wait_idle(device);
  CHECK(alive(held));
  gfx::release_retirements(device, 1);
  gfx::wait_idle(device);
  CHECK(alive(held));
  gfx::release_retirements(device, 2);
  gfx::wait_idle(device);
  CHECK_FALSE(alive(held));
  CHECK(gfx::buffer_address(device, storage) == 0u);

  const gfx::acceleration_structure_handle free =
      gfx::create_acceleration_structure(device,
                                         {.name = "ray free", .size = 4096});
  REQUIRE(alive(free));
  gfx::destroy_acceleration_structure(device, free);
  gfx::wait_idle(device);
  CHECK_FALSE(alive(free));
  CHECK(fixture.validation_errors() == 0u);
}

TEST_CASE("gfx ray: with the ray-query route disabled the device has none of "
          "it") {
  RayFixture fixture(gfx::route::RayQueries);
  if (!fixture.ready)
    SKIP("no usable device");
  gfx::device &device = fixture.device;
  const gfx::device_caps &caps = gfx::caps(device);
  CHECK_FALSE(caps.acceleration_structures);
  CHECK_FALSE(caps.ray_queries);
  CHECK(caps.acceleration_structure_scratch_alignment == 0u);
  CHECK(caps.max_ray_instances == 0u);
  CHECK_FALSE(
      gfx::create_acceleration_structure(device, {.size = 4096}).valid());
  CHECK_FALSE(
      gfx::create_buffer(
          device, {.name = "ray input",
                   .size = 64,
                   .usage = gfx::buffer_usage::AccelerationStructureInput |
                            gfx::buffer_usage::DeviceAddress})
          .valid());
  CHECK_FALSE(gfx::create_compacted_size_pool(device, 1).valid());
  const gfx::ray_triangles triangle{.vertex_count = 3, .triangle_count = 1};
  CHECK(gfx::sizes(device, gfx::bottom_level_desc{.geometries = {&triangle, 1}})
            .structure == 0u);
  CHECK(fixture.validation_errors() == 0u);
}

namespace {

constexpr u32 PARITY_RAYS = 4096;

/// What a ray met, from the GPU's two float4s or the CPU reference.
struct Met {
  bool hit = false;
  f32 t = -1.0f;
  f32 u = 0.0f;
  f32 v = 0.0f;
  bool front = false;
  u32 primitive = 0;
  u32 instance = 0;
  u32 instance_index = 0;
  u32 geometry = 0;
};

[[nodiscard]] Met met_of(const float4 &first, const float4 &second) {
  Met out;
  out.hit = first.x >= 0.0f;
  out.t = first.x;
  out.u = first.y;
  out.v = first.z;
  out.front = first.w > 0.5f;
  out.primitive = std::bit_cast<u32>(second.x);
  out.instance = std::bit_cast<u32>(second.y);
  out.instance_index = std::bit_cast<u32>(second.z);
  out.geometry = std::bit_cast<u32>(second.w);
  return out;
}

template <typename T>
[[nodiscard]] std::span<const u8> bytes_of_all(const nx::vector<T> &values) {
  return {reinterpret_cast<const u8 *>(values.data()),
          values.size() * sizeof(T)};
}

/// Two soups placed by 30 instances, three ways over the same data: CPU
/// blocks, the same blocks on the GPU for the software scene, and hardware
/// bottom levels under a top level.
struct ParityScene {
  struct Level {
    nx::gfx::test::Soup soup;
    gfx::buffer_handle positions;
    gfx::buffer_handle indices;
    gfx::ray_triangles triangles;
    gfx::acceleration_structure_handle bottom;
    nx::blob<u8> block;
    gfx::buffer_handle block_buffer;
  };

  gfx::device &device;
  Level levels[2];
  nx::vector<gfx::bvh_instance> software;
  nx::vector<NxRayInstance> hardware;
  gfx::buffer_handle hardware_instances;
  gfx::top_level_desc top_desc;
  gfx::acceleration_structure_handle top;
  nx::blob<u8> software_top;
  gfx::buffer_handle software_top_buffer;
  gfx::buffer_handle scratch;
  u64 scratch_address = 0;
  bool made = false;

  ParityScene(gfx::device &dev, const bool opaque) : device(dev) {
    u64 most_scratch = 0;
    for (u32 l = 0; l < 2; ++l) {
      Level &level = levels[l];
      level.soup = nx::gfx::test::make_soup(l == 0 ? 300 : 1500, 101 + l);
      level.positions = input_buffer(device, "parity positions",
                                     bytes_of_all(level.soup.positions));
      level.indices = input_buffer(device, "parity indices",
                                   bytes_of_all(level.soup.indices));
      level.triangles = {
          .vertex_address = gfx::buffer_address(device, level.positions),
          .vertex_count = nx::cast<u32>(level.soup.positions.size() / 3),
          .index_address = gfx::buffer_address(device, level.indices),
          .triangle_count = nx::cast<u32>(level.soup.indices.size() / 3),
          .flags = opaque
                       ? gfx::ray_geometry_flags(gfx::ray_geometry_flag::Opaque)
                       : gfx::ray_geometry_flags(gfx::ray_geometry_flag::None)};
      const gfx::bottom_level_desc desc{.geometries = {&level.triangles, 1}};
      const gfx::acceleration_structure_sizes sizes = gfx::sizes(device, desc);
      most_scratch = nx::max(most_scratch, sizes.build_scratch);
      level.bottom = gfx::create_acceleration_structure(
          device, {.name = "parity bottom", .size = sizes.structure});

      const gfx::bvh_geometry geometry{.positions = level.soup.positions,
                                       .indices = level.soup.indices,
                                       .opaque = opaque};
      auto built = gfx::build_bvh_here({&geometry, 1});
      if (!built.has_value())
        return;
      level.block = std::move(built.value().bytes);
      level.block_buffer = gfx::create_buffer(
          device, {.name = "parity block",
                   .size = level.block.size(),
                   .usage = gfx::buffer_usage::Storage |
                            gfx::buffer_usage::DeviceAddress |
                            gfx::buffer_usage::CopyDst});
      gfx::wait_upload(
          device, gfx::upload_buffer(device, level.block_buffer, 0,
                                     {level.block.data(), level.block.size()},
                                     state::ShaderReadOnly));
    }

    nx::gfx::test::Random random(103);
    for (u32 i = 0; i < 30; ++i) {
      const nx::gfx::test::Placed placed =
          nx::gfx::test::make_placement(random, 2, i);
      NxRayInstance instance{};
      instance.row0 = nx::gfx::test::row_of(placed.rows[0]);
      instance.row1 = nx::gfx::test::row_of(placed.rows[1]);
      instance.row2 = nx::gfx::test::row_of(placed.rows[2]);
      instance.custom_index_and_mask = placed.custom | (placed.mask << 24);
      instance.flags = 0;
      instance.bottom_level = gfx::acceleration_structure_address(
          device, levels[placed.soup].bottom);
      hardware.push_back(instance);

      const auto *const header = reinterpret_cast<const NxBvhHeader *>(
          levels[placed.soup].block.data());
      gfx::bvh_instance soft{.instance = instance,
                             .bounds = {header->lo_x, header->lo_y,
                                        header->lo_z, header->hi_x,
                                        header->hi_y, header->hi_z}};
      soft.instance.bottom_level =
          gfx::buffer_address(device, levels[placed.soup].block_buffer);
      software.push_back(soft);
    }
    hardware_instances =
        input_buffer(device, "parity instances", bytes_of_all(hardware));
    top_desc = {.instance_address =
                    gfx::buffer_address(device, hardware_instances),
                .instance_count = nx::cast<u32>(hardware.size())};
    const gfx::acceleration_structure_sizes top_sizes =
        gfx::sizes(device, top_desc);
    most_scratch = nx::max(most_scratch, top_sizes.build_scratch);
    top = gfx::create_acceleration_structure(
        device, {.name = "parity top",
                 .kind = gfx::acceleration_structure_kind::Top,
                 .size = top_sizes.structure});

    auto built_top = gfx::build_bvh_top_here(software);
    if (!built_top.has_value())
      return;
    software_top = std::move(built_top.value().bytes);
    software_top_buffer =
        gfx::create_buffer(device, {.name = "parity software top",
                                    .size = software_top.size(),
                                    .usage = gfx::buffer_usage::Storage |
                                             gfx::buffer_usage::DeviceAddress |
                                             gfx::buffer_usage::CopyDst});
    gfx::wait_upload(
        device, gfx::upload_buffer(device, software_top_buffer, 0,
                                   {software_top.data(), software_top.size()},
                                   state::ShaderReadOnly));

    const u64 alignment =
        gfx::caps(device).acceleration_structure_scratch_alignment;
    scratch =
        gfx::create_buffer(device, {.name = "parity scratch",
                                    .size = most_scratch + alignment,
                                    .usage = gfx::buffer_usage::Storage |
                                             gfx::buffer_usage::DeviceAddress});
    scratch_address = align_up(gfx::buffer_address(device, scratch), alignment);
    made = top.valid() && scratch.valid() && software_top_buffer.valid() &&
           levels[0].bottom.valid() && levels[1].bottom.valid();
  }

  ~ParityScene() {
    gfx::destroy_acceleration_structure(device, top);
    gfx::destroy_buffer(device, scratch);
    gfx::destroy_buffer(device, software_top_buffer);
    gfx::destroy_buffer(device, hardware_instances);
    for (Level &level : levels) {
      gfx::destroy_acceleration_structure(device, level.bottom);
      gfx::destroy_buffer(device, level.block_buffer);
      gfx::destroy_buffer(device, level.indices);
      gfx::destroy_buffer(device, level.positions);
    }
  }

  [[nodiscard]] bool build(gfx::command_list &cmd) const {
    transition(cmd, scratch, state::Undefined,
               state::AccelerationStructureBuild);
    for (const Level &level : levels) {
      const gfx::buffer_handle storage =
          gfx::acceleration_structure_buffer(device, level.bottom);
      transition(cmd, storage, state::Undefined,
                 state::AccelerationStructureBuild);
      if (!gfx::build_acceleration_structure(
              cmd, level.bottom, {.geometries = {&level.triangles, 1}},
              scratch_address))
        return false;
      transition(cmd, storage, state::AccelerationStructureBuild,
                 state::AccelerationStructureRead);
      transition(cmd, scratch, state::AccelerationStructureBuild,
                 state::AccelerationStructureBuild);
    }
    const gfx::buffer_handle top_storage =
        gfx::acceleration_structure_buffer(device, top);
    transition(cmd, top_storage, state::Undefined,
               state::AccelerationStructureBuild);
    if (!gfx::build_acceleration_structure(cmd, top, top_desc, scratch_address))
      return false;
    transition(cmd, top_storage, state::AccelerationStructureBuild,
               state::AccelerationStructureRead);
    return true;
  }

  /// The CPU block a software instance's bottom-level address names.
  [[nodiscard]] std::span<const u8> block_at(const u64 address) const {
    for (const Level &level : levels)
      if (gfx::buffer_address(device, level.block_buffer) == address)
        return {level.block.data(), level.block.size()};
    return {};
  }
};

struct ParityPush {
  u64 scene = 0;
  u64 rays = 0;
  u64 hits = 0;
  u32 count = 0;
  f32 accept_from = 0.0f;
  u32 mask = 0xFFu;
  u32 pad = 0;
};

/// One shader's pipelines for both entry points.
struct ParityShader {
  gfx::device &device;
  gfx::shader_handle shader;
  gfx::pipeline_handle main;
  gfx::pipeline_handle cull_back;

  ParityShader(gfx::device &dev, const nx::string_view name) : device(dev) {
    const nx::string base = nx::string(NX_GFX_TEST_SHADER_DIR) + "/" + name;
    auto code = nx::fs::file_read(nx::fs::path_view(base + ".spv"));
    auto refl = nx::fs::file_read_text(nx::fs::path_view(base + ".refl.json"));
    if (!code.has_value() || !refl.has_value())
      return;
    shader = nx::gfx::test::create_shader(
        device, {.name = name, .code = code->data(), .code_size = code->size()},
        refl->view());
    if (!shader.valid())
      return;
    main = nx::gfx::test::compile_pipeline(
        device,
        gfx::compute_pipeline_desc{
            .name = name,
            .compute = {.shader = shader, .entry_point = "trace_main"}});
    cull_back = nx::gfx::test::compile_pipeline(
        device,
        gfx::compute_pipeline_desc{
            .name = name,
            .compute = {.shader = shader, .entry_point = "trace_cull_back"}});
  }
  ~ParityShader() {
    gfx::destroy_pipeline(device, cull_back);
    gfx::destroy_pipeline(device, main);
    gfx::destroy_shader(device, shader);
  }
  [[nodiscard]] bool ready() const { return main.valid() && cull_back.valid(); }
};

/// Random rays through the scene, half aimed at its middle.
[[nodiscard]] nx::vector<float4> parity_rays() {
  nx::gfx::test::Random random(104);
  nx::vector<float4> rays;
  for (u32 r = 0; r < PARITY_RAYS; ++r) {
    f32 o[3];
    f32 d[3];
    for (u32 a = 0; a < 3; ++a) {
      o[a] = random.uniform(-30.0f, 30.0f);
      d[a] = r % 2 == 0 ? random.uniform(-12.0f, 12.0f) - o[a]
                        : random.uniform(-1.0f, 1.0f);
    }
    rays.push_back(float4{o[0], o[1], o[2], 0.0f});
    rays.push_back(float4{d[0], d[1], d[2], 1.0e30f});
  }
  return rays;
}

/// Runs @p pipeline over the rays through the scene at @p scene, building
/// the hardware levels first where @p build.
[[nodiscard]] nx::vector<Met>
run_parity(gfx::device &device, const ParityScene *const build,
           const gfx::pipeline_handle pipeline, const u64 scene,
           const gfx::buffer_handle rays, const gfx::buffer_handle hits,
           const f32 accept_from, const u32 mask = 0xFFu) {
  nx::vector<Met> out;
  std::memset(gfx::buffer_mapped(device, hits), 0,
              PARITY_RAYS * 2 * sizeof(float4));
  gfx::command_list *cmd = gfx::begin_headless_frame(device);
  if (cmd == nullptr)
    return out;
  if (build != nullptr && !build->build(*cmd))
    return out;
  const ParityPush push{.scene = scene,
                        .rays = gfx::buffer_address(device, rays),
                        .hits = gfx::buffer_address(device, hits),
                        .count = PARITY_RAYS,
                        .accept_from = accept_from,
                        .mask = mask};
  gfx::bind_pipeline(*cmd, pipeline);
  gfx::push_constants(*cmd, &push, sizeof(push));
  gfx::dispatch(*cmd, (PARITY_RAYS + 63) / 64);
  if (!gfx::end_headless_frame(device))
    return out;
  gfx::wait_idle(device);
  const auto *const read =
      static_cast<const float4 *>(gfx::buffer_mapped(device, hits));
  for (u32 r = 0; r < PARITY_RAYS; ++r)
    out.push_back(met_of(read[r * 2], read[r * 2 + 1]));
  return out;
}

struct Parity {
  u32 hits = 0;
  /// Rays one side hit and the other missed, or hit something else further
  /// than a tie, and hits that disagree on facing.
  u32 differ = 0;
  u32 facing = 0;
};

/// A ray differs where one side hit and the other missed, or both hit at
/// distances apart; a different triangle at the same distance is a tie.
[[nodiscard]] Parity compare_met(const nx::vector<Met> &a,
                                 const nx::vector<Met> &b) {
  Parity out;
  for (usize r = 0; r < a.size() && r < b.size(); ++r) {
    out.hits += a[r].hit ? 1 : 0;
    if (a[r].hit != b[r].hit) {
      ++out.differ;
      continue;
    }
    if (!a[r].hit)
      continue;
    if (std::fabs(a[r].t - b[r].t) > 1e-4f * std::max(1.0f, a[r].t)) {
      ++out.differ;
      continue;
    }
    const bool same = a[r].instance_index == b[r].instance_index &&
                      a[r].primitive == b[r].primitive;
    if (same && a[r].front != b[r].front)
      ++out.facing;
  }
  return out;
}

} // namespace

TEST_CASE("gfx ray: the software scene meets what the CPU reference and the "
          "hardware meet") {
  RayFixture fixture;
  if (!fixture.ready)
    SKIP("no usable device");
  gfx::device &device = fixture.device;
  const bool hardware = gfx::caps(device).ray_queries;
  {
    ParityShader software_shader(device, "ray_scene_software");
    if (!software_shader.ready())
      SKIP("test shaders are not built in this configuration");
    const nx::vector<float4> ray_data = parity_rays();
    const gfx::buffer_handle rays =
        gfx::create_buffer(device, {.name = "parity rays",
                                    .size = ray_data.size() * sizeof(float4),
                                    .usage = gfx::buffer_usage::Storage |
                                             gfx::buffer_usage::DeviceAddress |
                                             gfx::buffer_usage::CopyDst});
    gfx::wait_upload(device,
                     gfx::upload_buffer(device, rays, 0, bytes_of_all(ray_data),
                                        state::ShaderReadOnly));
    const gfx::buffer_handle hits =
        gfx::create_buffer(device, {.name = "parity hits",
                                    .size = PARITY_RAYS * 2 * sizeof(float4),
                                    .usage = gfx::buffer_usage::Storage |
                                             gfx::buffer_usage::DeviceAddress,
                                    .memory = gfx::memory_usage::Readback,
                                    .persistently_mapped = true});
    REQUIRE(rays.valid());
    REQUIRE(hits.valid());

    for (const bool opaque : {true, false}) {
      ParityScene scene(device, opaque);
      REQUIRE(scene.software_top.size() > 0);
      if (hardware)
        REQUIRE(scene.made);
      const u64 software_address =
          gfx::buffer_address(device, scene.software_top_buffer);
      const f32 accept_from = opaque ? 0.0f : 0.3f;
      const nx::vector<Met> soft =
          run_parity(device, nullptr, software_shader.main, software_address,
                     rays, hits, accept_from);
      REQUIRE(soft.size() == PARITY_RAYS);

      if (opaque) {
        nx::vector<Met> reference;
        for (u32 r = 0; r < PARITY_RAYS; ++r) {
          const float4 &o = ray_data[r * 2];
          const float4 &d = ray_data[r * 2 + 1];
          const gfx::bvh_ray ray{.origin = {o.x, o.y, o.z},
                                 .direction = {d.x, d.y, d.z},
                                 .tmin = o.w,
                                 .tmax = d.w};
          gfx::bvh_hit got;
          Met met;
          met.hit = gfx::trace_bvh_scene(
              {scene.software_top.data(), scene.software_top.size()},
              [&](const u64 address) { return scene.block_at(address); }, ray,
              0xFFu, got);
          met.t = met.hit ? got.t : -1.0f;
          met.primitive = got.primitive;
          met.instance = got.instance;
          met.instance_index = got.instance_index;
          met.front = soft[r].front;
          reference.push_back(met);
        }
        const Parity cpu = compare_met(soft, reference);
        std::printf("ray parity: software vs CPU: %u of %u hit, %u differ\n",
                    cpu.hits, PARITY_RAYS, cpu.differ);
        CHECK(cpu.hits > PARITY_RAYS / 4);
        CHECK(cpu.hits < PARITY_RAYS);
        CHECK(cpu.differ * 500 <= PARITY_RAYS);
      }
      if (!hardware)
        continue;
      ParityShader hardware_shader(device, "ray_scene");
      REQUIRE(hardware_shader.ready());
      const nx::vector<Met> hard =
          run_parity(device, &scene, hardware_shader.main,
                     gfx::acceleration_structure_address(device, scene.top),
                     rays, hits, accept_from);
      const Parity parity = compare_met(soft, hard);
      std::printf("ray parity: software vs hardware (%s): %u of %u hit, %u "
                  "differ, %u disagree on facing\n",
                  opaque ? "opaque" : "any-hit", parity.hits, PARITY_RAYS,
                  parity.differ, parity.facing);
      CHECK(parity.differ * 500 <= PARITY_RAYS);
      CHECK(parity.facing == 0u);

      if (opaque) {
        const nx::vector<Met> soft_culled =
            run_parity(device, nullptr, software_shader.cull_back,
                       software_address, rays, hits, 0.0f);
        const nx::vector<Met> hard_culled =
            run_parity(device, nullptr, hardware_shader.cull_back,
                       gfx::acceleration_structure_address(device, scene.top),
                       rays, hits, 0.0f);
        const Parity culled = compare_met(soft_culled, hard_culled);
        std::printf("ray parity: back faces culled: %u hit, %u differ\n",
                    culled.hits, culled.differ);
        CHECK(culled.differ * 500 <= PARITY_RAYS);
        CHECK(culled.hits < parity.hits);

        const nx::vector<Met> soft_masked =
            run_parity(device, nullptr, software_shader.main, software_address,
                       rays, hits, 0.0f, 0x1u);
        const nx::vector<Met> hard_masked =
            run_parity(device, nullptr, hardware_shader.main,
                       gfx::acceleration_structure_address(device, scene.top),
                       rays, hits, 0.0f, 0x1u);
        const Parity masked = compare_met(soft_masked, hard_masked);
        std::printf("ray parity: half the instances masked out: %u hit, %u "
                    "differ\n",
                    masked.hits, masked.differ);
        CHECK(masked.differ * 500 <= PARITY_RAYS);
        CHECK(masked.hits < parity.hits);
      }
    }
    gfx::destroy_buffer(device, hits);
    gfx::destroy_buffer(device, rays);
  }
  gfx::wait_idle(device);
  CHECK(fixture.validation_errors() == 0u);
}

namespace {

struct RefitPush {
  u64 block = 0;
  u64 geometries = 0;
  u32 first = 0;
  u32 count = 0;
};

/// @p soup's vertices moved: twisted about y and pushed by a wave, so boxes
/// grow, shrink and slide.
[[nodiscard]] nx::gfx::test::Soup moved(const nx::gfx::test::Soup &soup) {
  nx::gfx::test::Soup out = soup;
  for (usize v = 0; v * 3 < out.positions.size(); ++v) {
    const f32 x = soup.positions[v * 3];
    const f32 y = soup.positions[v * 3 + 1];
    const f32 z = soup.positions[v * 3 + 2];
    const f32 turn = 0.05f * y;
    out.positions[v * 3] =
        x * std::cos(turn) - z * std::sin(turn) + 0.8f * std::sin(0.7f * y);
    out.positions[v * 3 + 1] = y + 0.6f * std::cos(0.5f * x);
    out.positions[v * 3 + 2] = x * std::sin(turn) + z * std::cos(turn);
  }
  return out;
}

/// The refit's kernels and a GPU copy of a block with the geometries it
/// reads, run once and read back.
struct Refit {
  gfx::device &device;
  gfx::shader_handle shader;
  gfx::pipeline_handle triangles;
  gfx::pipeline_handle nodes;

  explicit Refit(gfx::device &dev) : device(dev) {
    const nx::string base = nx::string(NX_GFX_TEST_SHADER_DIR) + "/bvh_refit";
    auto code = nx::fs::file_read(nx::fs::path_view(base + ".spv"));
    auto refl = nx::fs::file_read_text(nx::fs::path_view(base + ".refl.json"));
    if (!code.has_value() || !refl.has_value())
      return;
    shader = nx::gfx::test::create_shader(
        device,
        {.name = "bvh refit", .code = code->data(), .code_size = code->size()},
        refl->view());
    if (!shader.valid())
      return;
    triangles = nx::gfx::test::compile_pipeline(
        device,
        gfx::compute_pipeline_desc{
            .name = "bvh refit triangles",
            .compute = {.shader = shader, .entry_point = "refit_triangles"}});
    nodes = nx::gfx::test::compile_pipeline(
        device,
        gfx::compute_pipeline_desc{
            .name = "bvh refit nodes",
            .compute = {.shader = shader, .entry_point = "refit_nodes"}});
  }
  ~Refit() {
    gfx::destroy_pipeline(device, nodes);
    gfx::destroy_pipeline(device, triangles);
    gfx::destroy_shader(device, shader);
  }
  [[nodiscard]] bool ready() const {
    return triangles.valid() && nodes.valid();
  }

  [[nodiscard]] gfx::buffer_handle
  upload(const nx::string_view name, const std::span<const u8> bytes,
         const gfx::buffer_usage_flags extra = gfx::buffer_usage::None) {
    const gfx::buffer_handle buffer = gfx::create_buffer(
        device, {.name = name,
                 .size = bytes.size(),
                 .usage = gfx::buffer_usage::Storage |
                          gfx::buffer_usage::DeviceAddress |
                          gfx::buffer_usage::CopyDst | extra});
    if (buffer.valid())
      gfx::wait_upload(device, gfx::upload_buffer(device, buffer, 0, bytes,
                                                  state::ShaderReadOnly));
    return buffer;
  }

  /// @p block refitted on the GPU to @p soups' vertices as they are.
  [[nodiscard]] nx::blob<u8>
  run(const nx::blob<u8> &block,
      const std::span<const nx::gfx::test::Soup> soups) {
    nx::vector<gfx::buffer_handle> buffers;
    nx::vector<NxBvhRefitGeometry> table;
    for (const nx::gfx::test::Soup &soup : soups) {
      const gfx::buffer_handle positions =
          upload("refit positions", bytes_of_all(soup.positions));
      const gfx::buffer_handle indices =
          upload("refit indices", bytes_of_all(soup.indices));
      buffers.push_back(positions);
      buffers.push_back(indices);
      table.push_back({gfx::buffer_address(device, positions),
                       gfx::buffer_address(device, indices)});
    }
    const gfx::buffer_handle geometries =
        upload("refit geometries", bytes_of_all(table));
    const gfx::buffer_handle copy =
        upload("refit block", {block.data(), block.size()},
               gfx::buffer_usage::CopySrc);
    const gfx::buffer_handle read =
        gfx::create_buffer(device, {.name = "refit read",
                                    .size = block.size(),
                                    .usage = gfx::buffer_usage::CopyDst,
                                    .memory = gfx::memory_usage::Readback,
                                    .persistently_mapped = true});
    buffers.push_back(geometries);
    buffers.push_back(copy);
    buffers.push_back(read);

    nx::blob<u8> out;
    const auto *const header =
        reinterpret_cast<const NxBvhHeader *>(block.data());
    const auto *const levels =
        reinterpret_cast<const u32 *>(block.data() + header->levels_offset);
    gfx::command_list *cmd = gfx::begin_headless_frame(device);
    if (cmd != nullptr) {
      transition(*cmd, copy, state::ShaderReadOnly, state::ShaderReadWrite);
      RefitPush push{.block = gfx::buffer_address(device, copy),
                     .geometries = gfx::buffer_address(device, geometries),
                     .first = 0,
                     .count = header->primitive_count};
      gfx::bind_pipeline(*cmd, triangles);
      gfx::push_constants(*cmd, &push, sizeof(push));
      gfx::dispatch(*cmd, (push.count + 63) / 64);
      gfx::bind_pipeline(*cmd, nodes);
      for (u32 level = header->level_count; level-- > 0;) {
        transition(*cmd, copy, state::ShaderReadWrite, state::ShaderReadWrite);
        push.first = levels[level];
        push.count = levels[level + 1] - levels[level];
        gfx::push_constants(*cmd, &push, sizeof(push));
        gfx::dispatch(*cmd, (push.count + 63) / 64);
      }
      transition(*cmd, copy, state::ShaderReadWrite, state::CopySrc);
      transition(*cmd, read, state::Undefined, state::CopyDst);
      if (gfx::copy_buffer(*cmd, copy, 0, read, 0, block.size()) &&
          gfx::end_headless_frame(device)) {
        gfx::wait_idle(device);
        out = nx::blob<u8>(block.size());
        std::memcpy(out.data(), gfx::buffer_mapped(device, read), block.size());
      }
    }
    for (const gfx::buffer_handle buffer : buffers)
      gfx::destroy_buffer(device, buffer);
    return out;
  }
};

} // namespace

TEST_CASE("gfx bvh refit: a block refitted on the GPU after its vertices move "
          "holds and traces the moved triangles") {
  RayFixture fixture;
  if (!fixture.ready)
    SKIP("no usable device");
  gfx::device &device = fixture.device;
  {
    Refit refit(device);
    if (!refit.ready())
      SKIP("test shaders are not built in this configuration");
    const nx::gfx::test::Soup soups[] = {nx::gfx::test::make_soup(2000, 111),
                                         nx::gfx::test::make_soup(1000, 112)};
    const gfx::bvh_geometry geometries[] = {
        nx::gfx::test::geometry_of(soups[0]),
        nx::gfx::test::geometry_of(soups[1])};
    auto built = gfx::build_bvh_here(geometries, {.keep_bounds = true});
    REQUIRE(built.has_value());
    const nx::blob<u8> &block = built.value().bytes;

    const nx::gfx::test::Soup shifted[] = {moved(soups[0]), moved(soups[1])};
    const gfx::bvh_geometry moved_geometries[] = {
        nx::gfx::test::geometry_of(shifted[0]),
        nx::gfx::test::geometry_of(shifted[1])};
    const nx::blob<u8> refitted = refit.run(block, shifted);
    REQUIRE(refitted.size() == block.size());
    REQUIRE(gfx::validate_bvh(nx::gfx::test::view(refitted)).has_value());
    const nx::gfx::test::Containment held =
        nx::gfx::test::containment(refitted, NX_BVH_MAX_LEAF);
    CHECK(held.escapes == 0u);

    const nx::gfx::test::Agreement fresh = nx::gfx::test::compare(
        moved_geometries, shifted[0], refitted, 3000, 113);
    const nx::gfx::test::Agreement stale = nx::gfx::test::compare(
        moved_geometries, shifted[0], block, 3000, 113, false);
    std::printf("bvh refit: refitted %u of %u rays hit, %u grazes, %u wrong; "
                "unrefitted %u wrong\n",
                fresh.hits, fresh.rays, fresh.grazes, fresh.wrong, stale.wrong);
    CHECK(fresh.wrong == 0u);
    CHECK(fresh.grazes * 1000 <= fresh.rays);
    CHECK(fresh.hits > fresh.rays / 4);
    CHECK(stale.wrong > fresh.rays / 10);

    const auto *const header =
        reinterpret_cast<const NxBvhHeader *>(refitted.data());
    const auto *const bounds = reinterpret_cast<const NxBvhBounds *>(
        refitted.data() + header->bounds_offset);
    CHECK(header->lo_x == bounds[0].lo_x);
    CHECK(header->hi_z == bounds[0].hi_z);
  }
  gfx::wait_idle(device);
  CHECK(fixture.validation_errors() == 0u);
}

TEST_CASE("gfx bvh refit: a block whose vertices did not move refits to the "
          "bytes it was built with") {
  RayFixture fixture;
  if (!fixture.ready)
    SKIP("no usable device");
  gfx::device &device = fixture.device;
  {
    Refit refit(device);
    if (!refit.ready())
      SKIP("test shaders are not built in this configuration");
    const nx::gfx::test::Soup soup = nx::gfx::test::make_soup(5000, 121);
    const gfx::bvh_geometry geometry = nx::gfx::test::geometry_of(soup);
    auto built = gfx::build_bvh_here({&geometry, 1}, {.keep_bounds = true});
    REQUIRE(built.has_value());
    const nx::blob<u8> &block = built.value().bytes;
    const nx::blob<u8> refitted = refit.run(block, {&soup, 1});
    REQUIRE(refitted.size() == block.size());
    usize differ = 0;
    for (usize i = 0; i < block.size(); ++i)
      differ += block.data()[i] != refitted.data()[i] ? 1 : 0;
    std::printf("bvh refit: an unmoved block differs in %zu of %zu bytes\n",
                differ, block.size());
    CHECK(differ == 0u);
  }
  gfx::wait_idle(device);
  CHECK(fixture.validation_errors() == 0u);
}
