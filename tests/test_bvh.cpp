#include "framework/nxtest.h"

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

namespace {

/// A small, fixed random sequence, so every run sees the same soup.
struct Random {
  u64 state;
  explicit Random(const u64 seed) : state(seed * 0x9E3779B97F4A7C15ull + 1) {}
  [[nodiscard]] u64 next() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  }
  /// In [lo, hi).
  [[nodiscard]] f32 uniform(const f32 lo, const f32 hi) {
    return lo +
           (hi - lo) * nx::cast<f32>(next() >> 40) / nx::cast<f32>(1ull << 24);
  }
};

struct Soup {
  nx::vector<f32> positions;
  nx::vector<u32> indices;
};

/// @p count triangles of random shape about random points in a cube 20 on a
/// side, each about as large as the spacing between them.
[[nodiscard]] Soup make_soup(const u32 count, const u64 seed) {
  Random random(seed);
  Soup soup;
  const f32 size = 20.0f / std::cbrt(nx::cast<f32>(count)) + 0.05f;
  for (u32 t = 0; t < count; ++t) {
    const f32 cx = random.uniform(-10.0f, 10.0f);
    const f32 cy = random.uniform(-10.0f, 10.0f);
    const f32 cz = random.uniform(-10.0f, 10.0f);
    for (u32 c = 0; c < 3; ++c) {
      soup.positions.push_back(cx + random.uniform(-size, size));
      soup.positions.push_back(cy + random.uniform(-size, size));
      soup.positions.push_back(cz + random.uniform(-size, size));
      soup.indices.push_back(t * 3 + c);
    }
  }
  return soup;
}

struct Reference {
  bool hit = false;
  f64 t = 0.0;
  u32 primitive = 0;
  u32 geometry = 0;
  /// The nearest a hit came to an edge, in barycentrics, and the next
  /// nearest distance any other triangle was met at.
  f64 edge = 1.0;
  f64 second = std::numeric_limits<f64>::infinity();
  /// How squarely the ray met the nearest triangle: the cosine between the
  /// ray and its normal. A grazing ray's distance is ill-conditioned in
  /// float, so distances are compared along the normal.
  f64 cosine = 1.0;
};

/// Every triangle tested, in double (Moller and Trumbore).
[[nodiscard]] Reference
brute_force(std::span<const gfx::bvh_geometry> geometries,
            const gfx::bvh_ray &ray) {
  Reference best;
  best.t = ray.tmax;
  const f64 o[3] = {ray.origin[0], ray.origin[1], ray.origin[2]};
  const f64 d[3] = {ray.direction[0], ray.direction[1], ray.direction[2]};
  for (u32 g = 0; g < geometries.size(); ++g) {
    const gfx::bvh_geometry &geometry = geometries[g];
    for (u32 t = 0; t * 3 < geometry.indices.size(); ++t) {
      f64 p[3][3];
      for (u32 c = 0; c < 3; ++c)
        for (u32 a = 0; a < 3; ++a)
          p[c][a] =
              geometry.positions[usize{geometry.indices[t * 3 + c]} * 3 + a];
      const f64 e1[3] = {p[1][0] - p[0][0], p[1][1] - p[0][1],
                         p[1][2] - p[0][2]};
      const f64 e2[3] = {p[2][0] - p[0][0], p[2][1] - p[0][1],
                         p[2][2] - p[0][2]};
      const f64 h[3] = {d[1] * e2[2] - d[2] * e2[1],
                        d[2] * e2[0] - d[0] * e2[2],
                        d[0] * e2[1] - d[1] * e2[0]};
      const f64 det = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2];
      if (std::fabs(det) < 1e-30)
        continue;
      const f64 inv = 1.0 / det;
      const f64 s[3] = {o[0] - p[0][0], o[1] - p[0][1], o[2] - p[0][2]};
      const f64 u = inv * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);
      const f64 q[3] = {s[1] * e1[2] - s[2] * e1[1],
                        s[2] * e1[0] - s[0] * e1[2],
                        s[0] * e1[1] - s[1] * e1[0]};
      const f64 v = inv * (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]);
      if (u < 0.0 || v < 0.0 || u + v > 1.0)
        continue;
      const f64 hit_t = inv * (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]);
      if (hit_t <= ray.tmin)
        continue;
      if (hit_t < best.t) {
        best.second = best.hit ? best.t : best.second;
        best.hit = true;
        best.t = hit_t;
        best.primitive = t;
        best.geometry = g;
        best.edge = std::min({u, v, 1.0 - u - v});
        const f64 n[3] = {e1[1] * e2[2] - e1[2] * e2[1],
                          e1[2] * e2[0] - e1[0] * e2[2],
                          e1[0] * e2[1] - e1[1] * e2[0]};
        const f64 length = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        best.cosine = std::fabs(d[0] * n[0] + d[1] * n[1] + d[2] * n[2]) /
                      (length > 0.0 ? length : 1.0);
      } else if (hit_t < best.second) {
        best.second = hit_t;
      }
    }
  }
  return best;
}

/// Half the rays aimed through a random triangle, half random.
[[nodiscard]] gfx::bvh_ray make_ray(Random &random, const Soup &soup) {
  gfx::bvh_ray ray;
  for (u32 a = 0; a < 3; ++a)
    ray.origin[a] = random.uniform(-14.0f, 14.0f);
  f32 target[3];
  if ((random.next() & 1) != 0) {
    const usize t = random.next() % (soup.indices.size() / 3);
    for (u32 a = 0; a < 3; ++a) {
      target[a] = 0.0f;
      for (u32 c = 0; c < 3; ++c)
        target[a] +=
            soup.positions[usize{soup.indices[t * 3 + c]} * 3 + a] / 3.0f;
    }
  } else {
    for (u32 a = 0; a < 3; ++a)
      target[a] = random.uniform(-10.0f, 10.0f);
  }
  f32 length = 0.0f;
  for (u32 a = 0; a < 3; ++a) {
    ray.direction[a] = target[a] - ray.origin[a];
    length += ray.direction[a] * ray.direction[a];
  }
  length = std::sqrt(length);
  for (u32 a = 0; a < 3; ++a)
    ray.direction[a] /= length;
  return ray;
}

struct Agreement {
  u32 rays = 0;
  u32 hits = 0;
  /// Disagreements where the reference grazed an edge or met two triangles
  /// at once, which float and double may settle differently.
  u32 grazes = 0;
  u32 wrong = 0;
};

/// Traces @p rays through a block of @p geometries and against every
/// triangle, and counts how they agree.
[[nodiscard]] Agreement compare(std::span<const gfx::bvh_geometry> geometries,
                                const Soup &aim, const nx::blob<u8> &bvh,
                                const u32 rays, const u64 seed) {
  Agreement out;
  Random random(seed);
  for (u32 r = 0; r < rays; ++r) {
    const gfx::bvh_ray ray = make_ray(random, aim);
    const Reference expected = brute_force(geometries, ray);
    gfx::bvh_hit got;
    const bool hit = gfx::trace_bvh({bvh.data(), bvh.size()}, ray, got);
    ++out.rays;
    out.hits += expected.hit ? 1 : 0;
    const bool close_call =
        expected.edge < 1e-5 ||
        (expected.second - expected.t) < 1e-5 * std::max(1.0, expected.t);
    bool agree = hit == expected.hit;
    if (agree && hit) {
      const bool same = got.primitive == expected.primitive &&
                        got.geometry == expected.geometry;
      const f64 off = std::fabs(got.t - expected.t);
      const f64 scale = 1e-5 * std::max(1.0, expected.t);
      agree = (off <= 10.0 * scale || off * expected.cosine <= scale) &&
              (same || close_call);
    }
    if (agree)
      continue;
    if (close_call) {
      ++out.grazes;
      continue;
    }
    ++out.wrong;
    std::printf("bvh: ray %u from (%.9g %.9g %.9g) along (%.9g %.9g %.9g): "
                "expected %s t %.9g triangle %u/%u edge %.3g, got %s t %.9g "
                "triangle %u/%u\n",
                r, ray.origin[0], ray.origin[1], ray.origin[2],
                ray.direction[0], ray.direction[1], ray.direction[2],
                expected.hit ? "hit" : "miss", expected.t, expected.geometry,
                expected.primitive, expected.edge, hit ? "hit" : "miss", got.t,
                got.geometry, got.primitive);
  }
  return out;
}

[[nodiscard]] gfx::bvh_geometry geometry_of(const Soup &soup,
                                            const bool opaque = true) {
  return {
      .positions = soup.positions, .indices = soup.indices, .opaque = opaque};
}

[[nodiscard]] std::span<const u8> view(const nx::blob<u8> &bytes) {
  return {bytes.data(), bytes.size()};
}

} // namespace

TEST_CASE("gfx bvh: the nearest hit is the one every triangle tested gives") {
  constexpr u32 SIZES[] = {1, 2, 7, 100, 5000, 100000};
  for (const u32 size : SIZES) {
    const Soup soup = make_soup(size, size);
    const gfx::bvh_geometry geometry = geometry_of(soup);
    auto built = gfx::build_bvh_here({&geometry, 1});
    REQUIRE(built.has_value());
    REQUIRE(gfx::validate_bvh(view(built.value().bytes)).has_value());
    CHECK(built.value().triangle_count == size);
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
  CHECK(built.value().triangle_count == 800u);
  const Agreement agreement =
      compare(geometries, second, built.value().bytes, 2000, 13);
  CHECK(agreement.wrong == 0u);
  CHECK(agreement.grazes * 1000 <= agreement.rays);

  const auto *const header =
      reinterpret_cast<const NxBvhHeader *>(built.value().bytes.data());
  const auto *const triangles = reinterpret_cast<const NxBvhTriangle *>(
      built.value().bytes.data() + header->triangles_offset);
  u32 opaque[2] = {};
  for (u32 t = 0; t < header->triangle_count; ++t) {
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
    auto built = gfx::build_bvh_here(
        {&geometry, 1}, {.max_leaf_triangles = limit, .keep_bounds = true});
    REQUIRE(built.has_value());
    const nx::blob<u8> &bytes = built.value().bytes;
    REQUIRE(gfx::validate_bvh(view(bytes)).has_value());
    const auto *const header =
        reinterpret_cast<const NxBvhHeader *>(bytes.data());
    const auto *const nodes = reinterpret_cast<const NxBvhNode *>(
        bytes.data() + header->nodes_offset);
    const auto *const triangles = reinterpret_cast<const NxBvhTriangle *>(
        bytes.data() + header->triangles_offset);
    const auto *const bounds = reinterpret_cast<const NxBvhBounds *>(
        bytes.data() + header->bounds_offset);
    u32 escapes = 0;
    u32 oversized = 0;
    for (u32 n = 0; n < header->node_count; ++n) {
      const NxBvhNode &node = nodes[n];
      const f32 origin[3] = {node.origin_x, node.origin_y, node.origin_z};
      const u32 lo[3] = {node.lo_x, node.lo_y, node.lo_z};
      const u32 hi[3] = {node.hi_x, node.hi_y, node.hi_z};
      const u32 words[4] = {node.children.x, node.children.y, node.children.z,
                            node.children.w};
      for (u32 c = 0; c < nx_bvh_child_count(node.meta); ++c) {
        f32 want_lo[3] = {std::numeric_limits<f32>::infinity(),
                          std::numeric_limits<f32>::infinity(),
                          std::numeric_limits<f32>::infinity()};
        f32 want_hi[3] = {-want_lo[0], -want_lo[1], -want_lo[2]};
        if (nx_bvh_is_leaf(words[c])) {
          if (nx_bvh_leaf_count(words[c]) > limit)
            ++oversized;
          for (u32 t = nx_bvh_leaf_first(words[c]);
               t < nx_bvh_leaf_first(words[c]) + nx_bvh_leaf_count(words[c]);
               ++t)
            for (const float4 &v :
                 {triangles[t].v0, triangles[t].v1, triangles[t].v2}) {
              const f32 p[3] = {v.x, v.y, v.z};
              for (u32 a = 0; a < 3; ++a) {
                want_lo[a] = std::min(want_lo[a], p[a]);
                want_hi[a] = std::max(want_hi[a], p[a]);
              }
            }
        } else {
          const NxBvhBounds &child = bounds[words[c]];
          const f32 clo[3] = {child.lo_x, child.lo_y, child.lo_z};
          const f32 chi[3] = {child.hi_x, child.hi_y, child.hi_z};
          for (u32 a = 0; a < 3; ++a) {
            want_lo[a] = clo[a];
            want_hi[a] = chi[a];
          }
        }
        for (u32 a = 0; a < 3; ++a) {
          const f32 step = nx_bvh_step(node.meta, a);
          const f32 got_lo =
              origin[a] + nx::cast<f32>(nx_bvh_byte(lo[a], c)) * step;
          const f32 got_hi =
              origin[a] + nx::cast<f32>(nx_bvh_byte(hi[a], c)) * step;
          if (got_lo > want_lo[a] || got_hi < want_hi[a])
            ++escapes;
        }
      }
    }
    CHECK(escapes == 0u);
    CHECK(oversized == 0u);
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
  CHECK(refused({&good, 1}, {.max_leaf_triangles = 0}) ==
        gfx::bvh_error::BadOptions);
  CHECK(refused({&good, 1}, {.max_leaf_triangles = 9}) ==
        gfx::bvh_error::BadOptions);
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
          reinterpret_cast<NxBvhHeader *>(b)->version = 2u;
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
              NX_BVH_LEAF | (header->triangle_count - 1u);
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
              reinterpret_cast<NxBvhTriangle *>(b + header->triangles_offset);
          triangles[7].v1.y = std::numeric_limits<f32>::quiet_NaN();
        }) == gfx::bvh_error::BadLayout);

  nx::blob<u8> shifted(good.size() + 16);
  std::memcpy(shifted.data() + 4, good.data(), good.size());
  CHECK(gfx::validate_bvh({shifted.data() + 4, good.size()}).error() ==
        gfx::bvh_error::BadLayout);

  auto plain = gfx::build_bvh_here({&geometry, 1});
  REQUIRE(plain.has_value());
  CHECK(plain.value().bytes.size() + header->node_count * sizeof(NxBvhBounds) ==
        good.size());
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
