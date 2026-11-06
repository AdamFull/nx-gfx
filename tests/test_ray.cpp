#include "framework/nxtest.h"
#include "pipeline_test_utils.h"

#include "core/foundation/diagnostics/log.h"
#include "core/foundation/platform/filesystem.h"
#include "nx/gfx/gfx.h"
#include "owned_device.h"
#include "validation_errors.h"

#include <array>
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
