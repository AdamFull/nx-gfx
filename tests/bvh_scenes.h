#pragma once

// Random triangle soups and instance placements the BVH and ray-tracing tests
// share, the same from run to run.

#include "core/foundation/core/foundation.h"
#include "nx/gfx/bvh.h"
#include "nx/gfx/shaders/interop.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <span>

#include <cmath>

namespace nx::gfx::test {

namespace gfx = nx::gfx;

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
[[nodiscard]] inline Soup make_soup(const u32 count, const u64 seed) {
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

/// An instance of soup `soup`, placed by `rows` (3x4, its space to the
/// world's).
struct Placed {
  f64 rows[3][4];
  u32 soup;
  u32 custom;
  u32 mask;
};

/// A random rotation, scale (some mirrored, some uneven) and offset.
[[nodiscard]] inline Placed make_placement(Random &random, const u32 soups,
                                           const u32 index) {
  Placed out{};
  f64 q[4] = {random.uniform(-1, 1), random.uniform(-1, 1),
              random.uniform(-1, 1), random.uniform(-1, 1)};
  const f64 length =
      std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  for (f64 &c : q)
    c /= length;
  const f64 x = q[0], y = q[1], z = q[2], w = q[3];
  const f64 r[3][3] = {
      {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
      {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
      {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}};
  f64 scale[3];
  const f64 uniform = random.uniform(0.3f, 1.5f);
  for (u32 a = 0; a < 3; ++a)
    scale[a] = index % 3 == 1 ? random.uniform(0.3f, 1.5f) : uniform;
  if (index % 5 == 2)
    scale[index % 3] = -scale[index % 3];
  for (u32 i = 0; i < 3; ++i) {
    for (u32 j = 0; j < 3; ++j)
      out.rows[i][j] = r[i][j] * scale[j];
    out.rows[i][3] = random.uniform(-12.0f, 12.0f);
  }
  out.soup = index % soups;
  out.custom = 100 + index;
  out.mask = index % 2 == 0 ? 0x1u : 0x2u;
  return out;
}

[[nodiscard]] inline float4 row_of(const f64 *const row) {
  return float4{nx::cast<f32>(row[0]), nx::cast<f32>(row[1]),
                nx::cast<f32>(row[2]), nx::cast<f32>(row[3])};
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
[[nodiscard]] inline Reference
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
[[nodiscard]] inline gfx::bvh_ray make_ray(Random &random, const Soup &soup) {
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
[[nodiscard]] inline Agreement
compare(std::span<const gfx::bvh_geometry> geometries, const Soup &aim,
        const nx::blob<u8> &bvh, const u32 rays, const u64 seed,
        const bool report = true) {
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
    if (!report)
      continue;
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

[[nodiscard]] inline gfx::bvh_geometry geometry_of(const Soup &soup,
                                                   const bool opaque = true) {
  return {
      .positions = soup.positions, .indices = soup.indices, .opaque = opaque};
}

[[nodiscard]] inline std::span<const u8> view(const nx::blob<u8> &bytes) {
  return {bytes.data(), bytes.size()};
}

/// How a block's decoded boxes hold what they stand for: the child boxes that
/// let a corner or an exact box out, and the leaves over @p limit.
struct Containment {
  u32 escapes = 0;
  u32 oversized = 0;
};

[[nodiscard]] inline Containment containment(const nx::blob<u8> &bytes,
                                             const u32 limit) {
  const auto *const header =
      reinterpret_cast<const NxBvhHeader *>(bytes.data());
  const auto *const nodes =
      reinterpret_cast<const NxBvhNode *>(bytes.data() + header->nodes_offset);
  const auto *const triangles = reinterpret_cast<const NxBvhTriangle *>(
      bytes.data() + header->primitives_offset);
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
             t < nx_bvh_leaf_first(words[c]) + nx_bvh_leaf_count(words[c]); ++t)
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
  return {escapes, oversized};
}

} // namespace nx::gfx::test
