#include "framework/nxtest.h"

#include "bvh_scenes.h"
#include "core/foundation/core/foundation.h"
#include "core/foundation/platform/os.h"
#include "core/foundation/threading/cancellation.h"
#include "core/foundation/threading/thread_pool.h"
#include "nx/gfx/bvh.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace gfx = nx::gfx;
using nx::gfx::test::Agreement;
using nx::gfx::test::brute_force;
using nx::gfx::test::compare;
using nx::gfx::test::Containment;
using nx::gfx::test::containment;
using nx::gfx::test::geometry_of;
using nx::gfx::test::make_placement;
using nx::gfx::test::make_ray;
using nx::gfx::test::make_soup;
using nx::gfx::test::Placed;
using nx::gfx::test::Random;
using nx::gfx::test::Reference;
using nx::gfx::test::row_of;
using nx::gfx::test::Soup;
using nx::gfx::test::view;

TEST_CASE("gfx bvh: the nearest hit is the one every triangle tested gives") {
  constexpr u32 SIZES[] = {1, 2, 7, 100, 5000, 100000};
  for (const u32 size : SIZES) {
    const Soup soup = make_soup(size, size);
    const gfx::bvh_geometry geometry = geometry_of(soup);
    auto built = gfx::build_bvh_here({&geometry, 1});
    REQUIRE(built.has_value());
    REQUIRE(gfx::validate_bvh(view(built.value().bytes)).has_value());
    CHECK(built.value().primitive_count == size);
    const u32 rays = size >= 100000 ? 200 : 2000;
    const Agreement agreement =
        compare({&geometry, 1}, soup, built.value().bytes, rays, size + 1);
    std::printf("bvh: %u triangles, %u nodes, %u levels: %u of %u rays hit, "
                "%u grazes, %u wrong\n",
                size, built.value().node_count, built.value().level_count,
                agreement.hits, agreement.rays, agreement.grazes,
                agreement.wrong);
    CHECK(agreement.wrong == 0u);
    CHECK(agreement.grazes * 1000 <= agreement.rays);
    CHECK(agreement.hits > 0u);
  }
}

TEST_CASE("gfx bvh: a hit names its geometry and its triangle within it") {
  const Soup first = make_soup(300, 11);
  const Soup second = make_soup(500, 12);
  const gfx::bvh_geometry geometries[] = {geometry_of(first),
                                          geometry_of(second, false)};
  auto built = gfx::build_bvh_here(geometries);
  REQUIRE(built.has_value());
  REQUIRE(gfx::validate_bvh(view(built.value().bytes)).has_value());
  CHECK(built.value().primitive_count == 800u);
  const Agreement agreement =
      compare(geometries, second, built.value().bytes, 2000, 13);
  CHECK(agreement.wrong == 0u);
  CHECK(agreement.grazes * 1000 <= agreement.rays);

  const auto *const header =
      reinterpret_cast<const NxBvhHeader *>(built.value().bytes.data());
  const auto *const triangles = reinterpret_cast<const NxBvhTriangle *>(
      built.value().bytes.data() + header->primitives_offset);
  u32 opaque[2] = {};
  for (u32 t = 0; t < header->primitive_count; ++t) {
    const u32 geometry = std::bit_cast<u32>(triangles[t].v1.w);
    REQUIRE(geometry < 2u);
    if ((std::bit_cast<u32>(triangles[t].v2.w) & NX_BVH_TRIANGLE_OPAQUE) != 0)
      ++opaque[geometry];
  }
  CHECK(opaque[0] == 300u);
  CHECK(opaque[1] == 0u);
}

TEST_CASE("gfx bvh: every decoded box holds what it stands for, and leaves "
          "hold at most their limit") {
  for (const u32 limit : {1u, 4u, 8u}) {
    const Soup soup = make_soup(20000, 21 + limit);
    const gfx::bvh_geometry geometry = geometry_of(soup);
    auto built = gfx::build_bvh_here({&geometry, 1},
                                     {.max_leaf = limit, .keep_bounds = true});
    REQUIRE(built.has_value());
    const nx::blob<u8> &bytes = built.value().bytes;
    REQUIRE(gfx::validate_bvh(view(bytes)).has_value());
    const Containment held = containment(bytes, limit);
    CHECK(held.escapes == 0u);
    CHECK(held.oversized == 0u);
  }
}

TEST_CASE("gfx bvh: the bytes are the same built here and on 1 or 8 workers") {
  const Soup soup = make_soup(200000, 31);
  const gfx::bvh_geometry geometry = geometry_of(soup);
  auto here = gfx::build_bvh_here({&geometry, 1});
  REQUIRE(here.has_value());
  for (const u32 workers : {1u, 8u}) {
    nx::thread_pool pool({.thread_count = workers, .name = "nx-test-bvh"});
    auto future = gfx::build_bvh(pool, {&geometry, 1});
    auto built = future.get();
    REQUIRE(built.has_value());
    const nx::blob<u8> &a = here.value().bytes;
    const nx::blob<u8> &b = built.value().bytes;
    REQUIRE(a.size() == b.size());
    CHECK(std::memcmp(a.data(), b.data(), a.size()) == 0);
  }
}

TEST_CASE("gfx bvh: a cancelled build reports cancelled") {
  const Soup soup = make_soup(200000, 41);
  const gfx::bvh_geometry geometry = geometry_of(soup);
  nx::thread_pool pool(
      {.thread_count = 4, .name = "nx-test-bvh", .collect_statistics = true});

  nx::cancellation_source before;
  before.request_stop();
  auto refused = gfx::build_bvh(pool, {&geometry, 1}, {}, before.token());
  refused.wait();
  CHECK(refused.cancelled());

  // Stopped once the build has begun its own tasks, so the builder, not the
  // pool's check before it starts, is what stops it.
  nx::cancellation_source during;
  const u64 critical_before =
      pool.statistics()
          .lanes[nx::task_priority_index(nx::task_priority::Critical)]
          .started;
  auto stopped = gfx::build_bvh(pool, {&geometry, 1}, {}, during.token());
  const u64 deadline = nx::os::ticks_ns() + 10'000'000'000ull;
  while (pool.statistics()
                 .lanes[nx::task_priority_index(nx::task_priority::Critical)]
                 .started == critical_before &&
         !stopped.ready() && nx::os::ticks_ns() < deadline) {
  }
  REQUIRE_FALSE(stopped.ready());
  during.request_stop();
  stopped.wait();
  CHECK(stopped.cancelled());

  auto kept = gfx::build_bvh(pool, {&geometry, 1});
  auto built = kept.get();
  CHECK(built.has_value());
}

TEST_CASE("gfx bvh: input it cannot build is refused") {
  const auto refused = [](std::span<const gfx::bvh_geometry> geometries,
                          const gfx::bvh_options options = {}) {
    auto built = gfx::build_bvh_here(geometries, options);
    REQUIRE_FALSE(built.has_value());
    return built.error();
  };
  const f32 positions[9] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
  const u32 indices[3] = {0, 1, 2};
  CHECK(refused({}) == gfx::bvh_error::Empty);
  const gfx::bvh_geometry none{};
  CHECK(refused({&none, 1}) == gfx::bvh_error::Empty);
  const gfx::bvh_geometry two_indices{.positions = positions,
                                      .indices = {indices, 2}};
  CHECK(refused({&two_indices, 1}) == gfx::bvh_error::NotTriangles);
  const gfx::bvh_geometry eight_floats{.positions = {positions, 8},
                                       .indices = indices};
  CHECK(refused({&eight_floats, 1}) == gfx::bvh_error::NotTriangles);
  const u32 past[3] = {0, 1, 3};
  const gfx::bvh_geometry out_of_range{.positions = positions, .indices = past};
  CHECK(refused({&out_of_range, 1}) == gfx::bvh_error::IndexOutOfRange);
  for (const f32 bad : {std::numeric_limits<f32>::quiet_NaN(),
                        std::numeric_limits<f32>::infinity()}) {
    f32 broken[9];
    std::memcpy(broken, positions, sizeof(broken));
    broken[4] = bad;
    const gfx::bvh_geometry non_finite{.positions = broken, .indices = indices};
    CHECK(refused({&non_finite, 1}) == gfx::bvh_error::NonFinite);
  }
  const gfx::bvh_geometry good{.positions = positions, .indices = indices};
  CHECK(refused({&good, 1}, {.max_leaf = 0}) == gfx::bvh_error::BadOptions);
  CHECK(refused({&good, 1}, {.max_leaf = 9}) == gfx::bvh_error::BadOptions);
  CHECK(refused({&good, 1}, {.bins = 1}) == gfx::bvh_error::BadOptions);
  CHECK(refused({&good, 1}, {.bins = 65}) == gfx::bvh_error::BadOptions);
  CHECK(gfx::build_bvh_here({&good, 1}).has_value());
}

TEST_CASE("gfx bvh: malformed blocks are refused") {
  const Soup soup = make_soup(3000, 51);
  const gfx::bvh_geometry geometry = geometry_of(soup);
  auto built = gfx::build_bvh_here({&geometry, 1}, {.keep_bounds = true});
  REQUIRE(built.has_value());
  const nx::blob<u8> &good = built.value().bytes;
  REQUIRE(gfx::validate_bvh(view(good)).has_value());
  const auto *const header = reinterpret_cast<const NxBvhHeader *>(good.data());
  REQUIRE(header->node_count > 2u);

  // An aligned copy of the block, changed by @p change, then validated.
  const auto check = [&](auto change, const usize size_change = 0) {
    nx::blob<u8> copy(good.size() + size_change);
    std::memcpy(copy.data(), good.data(), std::min(good.size(), copy.size()));
    change(copy.data());
    auto result = gfx::validate_bvh({copy.data(), copy.size()});
    REQUIRE_FALSE(result.has_value());
    return result.error();
  };
  const auto node_at = [&](u8 *const base, const u32 n) {
    return reinterpret_cast<NxBvhNode *>(base + header->nodes_offset +
                                         usize{n} * sizeof(NxBvhNode));
  };
  const auto none = [](u8 *) {};

  CHECK(gfx::validate_bvh({good.data(), 63}).error() ==
        gfx::bvh_error::Truncated);
  CHECK(gfx::validate_bvh({good.data(), good.size() - 1}).error() ==
        gfx::bvh_error::Truncated);
  CHECK(check(none, 16) == gfx::bvh_error::BadLayout);
  CHECK(check([](u8 *b) { reinterpret_cast<NxBvhHeader *>(b)->magic ^= 1u; }) ==
        gfx::bvh_error::BadMagic);
  CHECK(check([](u8 *b) {
          reinterpret_cast<NxBvhHeader *>(b)->version = NX_BVH_VERSION + 1u;
        }) == gfx::bvh_error::BadVersion);
  CHECK(check([](u8 *b) {
          reinterpret_cast<NxBvhHeader *>(b)->node_count += 1u;
        }) == gfx::bvh_error::BadLayout);
  CHECK(check([&](u8 *b) { node_at(b, 0)->children.x = 0u; }) ==
        gfx::bvh_error::BadLayout);
  CHECK(check([&](u8 *b) { node_at(b, 0)->meta &= 0x00FFFFFFu; }) ==
        gfx::bvh_error::BadLayout);
  CHECK(check([&](u8 *b) { node_at(b, 1)->meta &= 0xFFFFFF00u; }) ==
        gfx::bvh_error::BadLayout);
  CHECK(check([&](u8 *b) { node_at(b, 1)->parent = 5u; }) ==
        gfx::bvh_error::BadLayout);
  // A leaf past the triangles, and a leaf naming triangles another holds.
  const u32 last = header->node_count - 1;
  CHECK(check([&](u8 *b) {
          node_at(b, last)->children.x =
              NX_BVH_LEAF | (header->primitive_count - 1u);
          node_at(b, last)->children.x |= 1u << NX_BVH_LEAF_COUNT_SHIFT;
        }) == gfx::bvh_error::BadLayout);
  CHECK(check([&](u8 *b) {
          NxBvhNode *const leaf = node_at(b, last);
          leaf->children.y = leaf->children.x;
          leaf->meta = (leaf->meta & 0x00FFFFFFu) | (2u << 24);
        }) == gfx::bvh_error::BadLayout);
  CHECK(check([&](u8 *b) {
          u32 *const levels =
              reinterpret_cast<u32 *>(b + header->levels_offset);
          levels[1] = 2u;
        }) == gfx::bvh_error::BadLayout);
  CHECK(check([&](u8 *b) {
          auto *const triangles =
              reinterpret_cast<NxBvhTriangle *>(b + header->primitives_offset);
          triangles[7].v1.y = std::numeric_limits<f32>::quiet_NaN();
        }) == gfx::bvh_error::BadLayout);

  nx::blob<u8> shifted(good.size() + 16);
  std::memcpy(shifted.data() + 4, good.data(), good.size());
  CHECK(gfx::validate_bvh({shifted.data() + 4, good.size()}).error() ==
        gfx::bvh_error::BadLayout);

  auto plain = gfx::build_bvh_here({&geometry, 1});
  REQUIRE(plain.has_value());
  const usize kept = good.size() - plain.value().bytes.size() -
                     usize{header->node_count} * sizeof(NxBvhBounds);
  CHECK(kept < 16u);
  CHECK(header->bounds_offset % 16 == 0u);
}

TEST_CASE("gfx bvh: a million triangles build, and the time is reported") {
  const Soup soup = make_soup(1000000, 61);
  const gfx::bvh_geometry geometry = geometry_of(soup);
  nx::thread_pool pool({.name = "nx-test-bvh"});
  auto built = gfx::build_bvh(pool, {&geometry, 1}).get();
  REQUIRE(built.has_value());
  REQUIRE(gfx::validate_bvh(view(built.value().bytes)).has_value());
  std::printf("bvh: 1M triangles on %u workers in %.1f ms: %u nodes, %u "
              "levels, %.1f MiB\n",
              pool.worker_count(), nx::cast<f64>(built.value().build_ns) / 1e6,
              built.value().node_count, built.value().level_count,
              nx::cast<f64>(built.value().bytes.size()) / (1024.0 * 1024.0));
  const Agreement agreement =
      compare({&geometry, 1}, soup, built.value().bytes, 20, 62);
  CHECK(agreement.wrong == 0u);
  CHECK(agreement.hits > 0u);
}

namespace {

[[nodiscard]] gfx::bvh_instance instance_of(const Placed &placed,
                                            const nx::blob<u8> &bottom) {
  const auto *const header =
      reinterpret_cast<const NxBvhHeader *>(bottom.data());
  gfx::bvh_instance out{};
  out.instance.row0 = row_of(placed.rows[0]);
  out.instance.row1 = row_of(placed.rows[1]);
  out.instance.row2 = row_of(placed.rows[2]);
  out.instance.custom_index_and_mask = placed.custom | (placed.mask << 24);
  out.instance.flags = 0;
  out.instance.bottom_level = placed.soup;
  out.bounds = {header->lo_x, header->lo_y, header->lo_z,
                header->hi_x, header->hi_y, header->hi_z};
  return out;
}

struct SceneReference {
  Reference hit;
  u32 custom = 0;
  u32 index = 0;
};

/// Every instance's every triangle, placed by the rows the top level was
/// given (the float ones) and tested in double.
[[nodiscard]] SceneReference
brute_force_scene(std::span<const Soup> soups,
                  std::span<const gfx::bvh_instance> instances,
                  const gfx::bvh_ray &ray, const u32 mask) {
  SceneReference best;
  best.hit.t = ray.tmax;
  for (u32 i = 0; i < instances.size(); ++i) {
    const gfx::bvh_instance &instance = instances[i];
    if (((instance.instance.custom_index_and_mask >> 24) & mask) == 0)
      continue;
    const Soup &soup = soups[instance.instance.bottom_level];
    const float4 rows[3] = {instance.instance.row0, instance.instance.row1,
                            instance.instance.row2};
    Soup placed;
    placed.indices = soup.indices;
    for (usize v = 0; v * 3 < soup.positions.size(); ++v)
      for (u32 r = 0; r < 3; ++r)
        placed.positions.push_back(nx::cast<f32>(
            f64{rows[r].x} * soup.positions[v * 3] +
            f64{rows[r].y} * soup.positions[v * 3 + 1] +
            f64{rows[r].z} * soup.positions[v * 3 + 2] + f64{rows[r].w}));
    const gfx::bvh_geometry geometry = geometry_of(placed);
    Reference found = brute_force({&geometry, 1}, ray);
    if (!found.hit)
      continue;
    if (found.t < best.hit.t || !best.hit.hit) {
      const f64 previous = best.hit.hit ? best.hit.t : found.second;
      found.second = std::min(found.second, previous);
      best.hit = found;
      best.custom = instance.instance.custom_index_and_mask & 0x00FFFFFFu;
      best.index = i;
    } else {
      best.hit.second = std::min(best.hit.second, found.t);
    }
  }
  return best;
}

} // namespace

TEST_CASE("gfx bvh: the nearest hit through a top level is the one every "
          "placed triangle gives") {
  const Soup soups[] = {make_soup(500, 71), make_soup(2000, 72)};
  nx::vector<nx::blob<u8>> bottoms;
  for (const Soup &soup : soups) {
    const gfx::bvh_geometry geometry = geometry_of(soup);
    auto built = gfx::build_bvh_here({&geometry, 1});
    REQUIRE(built.has_value());
    bottoms.push_back(std::move(built.value().bytes));
  }
  Random random(73);
  nx::vector<gfx::bvh_instance> instances;
  for (u32 i = 0; i < 40; ++i)
    instances.push_back(
        instance_of(make_placement(random, 2, i), bottoms[i % 2]));
  auto top = gfx::build_bvh_top_here(instances);
  REQUIRE(top.has_value());
  const nx::blob<u8> &top_bytes = top.value().bytes;
  REQUIRE(gfx::validate_bvh(view(top_bytes)).has_value());
  CHECK((reinterpret_cast<const NxBvhHeader *>(top_bytes.data())->flags &
         NX_BVH_TOP) != 0u);

  const auto bottom_of = [&](const u64 bottom) -> std::span<const u8> {
    return bottom < bottoms.size() ? view(bottoms[nx::cast<usize>(bottom)])
                                   : std::span<const u8>{};
  };
  for (const u32 mask : {0xFFu, 0x1u}) {
    Agreement agreement;
    Random rays(74 + mask);
    for (u32 r = 0; r < 600; ++r) {
      gfx::bvh_ray ray;
      for (u32 a = 0; a < 3; ++a) {
        ray.origin[a] = rays.uniform(-30.0f, 30.0f);
        ray.direction[a] = r % 2 == 0
                               ? rays.uniform(-10.0f, 10.0f) - ray.origin[a]
                               : rays.uniform(-1.0f, 1.0f);
      }
      const SceneReference expected =
          brute_force_scene(soups, instances, ray, mask);
      gfx::bvh_hit got;
      const bool hit =
          gfx::trace_bvh_scene(view(top_bytes), bottom_of, ray, mask, got);
      ++agreement.rays;
      agreement.hits += expected.hit.hit ? 1 : 0;
      const Reference &e = expected.hit;
      const bool close_call =
          e.edge < 1e-5 || (e.second - e.t) < 1e-5 * std::max(1.0, e.t);
      bool agree = hit == e.hit;
      if (agree && hit) {
        const f64 off = std::fabs(got.t - e.t);
        const f64 scale = 1e-5 * std::max(1.0, e.t);
        const bool same = got.primitive == e.primitive &&
                          got.instance == expected.custom &&
                          got.instance_index == expected.index;
        agree = (off <= 10.0 * scale || off * e.cosine <= scale) &&
                (same || close_call);
      }
      if (agree)
        continue;
      if (close_call) {
        ++agreement.grazes;
        continue;
      }
      ++agreement.wrong;
      std::printf("bvh scene: ray %u mask %u: expected %s t %.9g instance %u "
                  "triangle %u, got %s t %.9g instance %u triangle %u\n",
                  r, mask, e.hit ? "hit" : "miss", e.t, expected.custom,
                  e.primitive, hit ? "hit" : "miss", got.t, got.instance,
                  got.primitive);
    }
    std::printf("bvh scene: mask %u: %u of %u rays hit, %u grazes, %u wrong\n",
                mask, agreement.hits, agreement.rays, agreement.grazes,
                agreement.wrong);
    CHECK(agreement.wrong == 0u);
    CHECK(agreement.grazes * 100 <= agreement.rays);
    CHECK(agreement.hits > 50u);
    CHECK(agreement.hits < agreement.rays);
  }
}

TEST_CASE("gfx bvh: a top level's bytes are the same built here and on 8 "
          "workers") {
  const Soup soup = make_soup(200, 81);
  const gfx::bvh_geometry geometry = geometry_of(soup);
  auto bottom = gfx::build_bvh_here({&geometry, 1});
  REQUIRE(bottom.has_value());
  Random random(82);
  nx::vector<gfx::bvh_instance> instances;
  for (u32 i = 0; i < 50000; ++i)
    instances.push_back(
        instance_of(make_placement(random, 1, i), bottom.value().bytes));
  auto here = gfx::build_bvh_top_here(instances);
  REQUIRE(here.has_value());
  REQUIRE(gfx::validate_bvh(view(here.value().bytes)).has_value());
  nx::thread_pool pool({.thread_count = 8, .name = "nx-test-bvh"});
  auto built = gfx::build_bvh_top(pool, instances).get();
  REQUIRE(built.has_value());
  REQUIRE(here.value().bytes.size() == built.value().bytes.size());
  CHECK(std::memcmp(here.value().bytes.data(), built.value().bytes.data(),
                    here.value().bytes.size()) == 0);
}

TEST_CASE("gfx bvh: instances a top level cannot hold are refused, and so "
          "are malformed top levels") {
  const Soup soup = make_soup(50, 91);
  const gfx::bvh_geometry geometry = geometry_of(soup);
  auto bottom = gfx::build_bvh_here({&geometry, 1});
  REQUIRE(bottom.has_value());
  Random random(92);
  const gfx::bvh_instance good =
      instance_of(make_placement(random, 1, 0), bottom.value().bytes);
  const auto refused = [](const gfx::bvh_instance &instance) {
    auto built = gfx::build_bvh_top_here({&instance, 1});
    REQUIRE_FALSE(built.has_value());
    return built.error();
  };
  CHECK(gfx::build_bvh_top_here({&good, 1}).has_value());
  CHECK(gfx::build_bvh_top_here({}).error() == gfx::bvh_error::Empty);
  gfx::bvh_instance singular = good;
  singular.instance.row1 = float4{0.0f, 0.0f, 0.0f, 1.0f};
  CHECK(refused(singular) == gfx::bvh_error::BadInstance);
  gfx::bvh_instance not_finite = good;
  not_finite.instance.row0.w = std::numeric_limits<f32>::infinity();
  CHECK(refused(not_finite) == gfx::bvh_error::BadInstance);
  gfx::bvh_instance inside_out = good;
  std::swap(inside_out.bounds.lo_x, inside_out.bounds.hi_x);
  CHECK(refused(inside_out) == gfx::bvh_error::BadInstance);
  gfx::bvh_instance low_flags = good;
  low_flags.instance.flags = 1;
  CHECK(refused(low_flags) == gfx::bvh_error::BadInstance);

  nx::vector<gfx::bvh_instance> many;
  for (u32 i = 0; i < 64; ++i)
    many.push_back(
        instance_of(make_placement(random, 1, i), bottom.value().bytes));
  auto top = gfx::build_bvh_top_here(many);
  REQUIRE(top.has_value());
  const nx::blob<u8> &bytes = top.value().bytes;
  const auto *const header =
      reinterpret_cast<const NxBvhHeader *>(bytes.data());
  const auto check = [&](auto change) {
    nx::blob<u8> copy(bytes.size());
    std::memcpy(copy.data(), bytes.data(), bytes.size());
    change(copy.data());
    auto result = gfx::validate_bvh({copy.data(), copy.size()});
    REQUIRE_FALSE(result.has_value());
    return result.error();
  };
  CHECK(check([](u8 *b) {
          reinterpret_cast<NxBvhHeader *>(b)->flags &= ~NX_BVH_TOP;
        }) == gfx::bvh_error::BadLayout);
  CHECK(check([&](u8 *b) {
          reinterpret_cast<NxBvhInstance *>(b + header->primitives_offset)[3]
              .row2.y = std::numeric_limits<f32>::quiet_NaN();
        }) == gfx::bvh_error::BadLayout);
}
