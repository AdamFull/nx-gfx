#pragma once

// Random triangle soups and instance placements the BVH and ray-tracing tests
// share, the same from run to run.

#include "core/foundation/core/foundation.h"
#include "nx/gfx/shaders/interop.h"

#include <cmath>

namespace nx::gfx::test {

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

} // namespace nx::gfx::test
