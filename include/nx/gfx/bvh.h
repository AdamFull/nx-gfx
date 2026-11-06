#pragma once

#include "core/foundation/containers/blob.h"
#include "core/foundation/core/foundation.h"
#include "core/foundation/core/result.h"
#include "core/foundation/strings/utf8_string_view.h"
#include "core/foundation/threading/cancellation.h"
#include "core/foundation/threading/thread_pool.h"
#include "nx/gfx/shaders/bvh_layout.h"

#include <span>

namespace nx::gfx {

/// The software ray-tracing route's bottom level: a 4-wide BVH over
/// triangles in one block (nx/gfx/shaders/bvh_layout.h) that a shader walks
/// from its address where the device has no ray queries. Building one is CPU
/// work, done once for cooked geometry and on fiber jobs for the rest.

/// Triangles over positions of three f32 each, indexed three a triangle.
struct bvh_geometry {
  std::span<const f32> positions;
  std::span<const u32> indices;
  /// No candidate is reported for its triangles: every hit counts.
  bool opaque = true;
};

struct bvh_options {
  /// 1 to NX_BVH_MAX_LEAF.
  u32 max_leaf_triangles = 4;
  /// Binned SAH: the bins an axis is split over, 2 to 64.
  u32 bins = 16;
  /// Keeps each node's exact bounds in the block, which refitting needs.
  bool keep_bounds = false;
};

enum class bvh_error : u8 {
  /// No triangles.
  Empty,
  /// An index or position count not a multiple of three.
  NotTriangles,
  IndexOutOfRange,
  /// A position not finite.
  NonFinite,
  /// More triangles or bytes than the block can name.
  TooLarge,
  BadOptions,
  // What validate_bvh finds.
  Truncated,
  BadMagic,
  BadVersion,
  BadLayout,
};

[[nodiscard]] nx::string_view bvh_error_name(bvh_error error) noexcept;

struct bvh_blob {
  nx::blob<u8> bytes;
  u32 node_count = 0;
  u32 triangle_count = 0;
  u32 level_count = 0;
  /// How long the build took.
  u64 build_ns = 0;
};

/// Builds on the calling thread alone. The bytes depend on the input and
/// options only, never on the thread or the workers that built them.
[[nodiscard]] nx::result<bvh_blob, bvh_error>
build_bvh_here(std::span<const bvh_geometry> geometries,
               const bvh_options &options = {});

/// Builds on @p pool's workers, the same bytes as build_bvh_here. The caller
/// keeps @p geometries, and what they point at, alive until the future is
/// ready; a cancelled build's future reports cancelled.
[[nodiscard]] nx::future<nx::result<bvh_blob, bvh_error>>
build_bvh(nx::thread_pool &pool, std::span<const bvh_geometry> geometries,
          const bvh_options &options = {}, nx::cancellation_token token = {},
          nx::task_priority priority = nx::task_priority::Normal);

/// Checks a block from anywhere, a file or a cook, before it is trusted: its
/// header, every offset and count within @p bytes, nodes after their parents
/// and in their levels, leaves within the triangles, finite corners. The
/// header on success.
[[nodiscard]] nx::result<const NxBvhHeader *, bvh_error>
validate_bvh(std::span<const u8> bytes) noexcept;

struct bvh_ray {
  f32 origin[3] = {0, 0, 0};
  f32 direction[3] = {0, 0, 1};
  f32 tmin = 0.0f;
  f32 tmax = 3.4e38f;
};

struct bvh_hit {
  f32 t = 0.0f;
  /// The barycentrics of the triangle's second and third corners.
  f32 u = 0.0f;
  f32 v = 0.0f;
  u32 primitive = 0;
  u32 geometry = 0;
};

/// The nearest triangle a ray meets in a validated block, every triangle
/// opaque: the CPU reference the shader traversal is tested against.
/// Watertight: a ray through an edge two triangles share meets one of them.
[[nodiscard]] bool trace_bvh(std::span<const u8> bvh, const bvh_ray &ray,
                             bvh_hit &out) noexcept;

} // namespace nx::gfx
