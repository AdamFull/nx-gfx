#pragma once

#include "core/foundation/containers/blob.h"
#include "core/foundation/core/callable.h"
#include "core/foundation/core/foundation.h"
#include "core/foundation/core/result.h"
#include "core/foundation/strings/utf8_string_view.h"
#include "core/foundation/threading/cancellation.h"
#include "core/foundation/threading/thread_pool.h"
#include "nx/gfx/shaders/bvh_layout.h"
#include "nx/gfx/shaders/ray_layout.h"

#include <span>

namespace nx::gfx {

/// The software ray-tracing route's levels: a 4-wide BVH in one block
/// (nx/gfx/shaders/bvh_layout.h) over triangles (a bottom level) or over
/// instances of bottom levels (a top level), which a shader walks from the
/// top level's address where the device has no ray queries. Building one is
/// CPU work, done once for cooked geometry and on fiber jobs for the rest.

/// Triangles over positions of three f32 each, indexed three a triangle.
struct bvh_geometry {
  std::span<const f32> positions;
  std::span<const u32> indices;
  /// No candidate is reported for its triangles: every hit counts.
  bool opaque = true;
};

/// One instance of a top level: its record as hardware takes it (its
/// bottom_level what the top level will name the bottom level by), and that
/// bottom level's bounds in its own space (its NxBvhHeader's).
struct bvh_instance {
  NxRayInstance instance;
  NxBvhBounds bounds;
};

struct bvh_options {
  /// The primitives a leaf holds at most, 1 to NX_BVH_MAX_LEAF.
  u32 max_leaf = 4;
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
  /// An instance whose transform has no inverse or whose bounds are no box.
  BadInstance,
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
  u32 primitive_count = 0;
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

/// A top level over @p instances, here or on @p pool's workers, as
/// build_bvh_here and build_bvh build a bottom one, except that each leaf
/// holds one instance whatever bvh_options::max_leaf says: a shader enters
/// one instance at a time.
[[nodiscard]] nx::result<bvh_blob, bvh_error>
build_bvh_top_here(std::span<const bvh_instance> instances,
                   const bvh_options &options = {});
[[nodiscard]] nx::future<nx::result<bvh_blob, bvh_error>>
build_bvh_top(nx::thread_pool &pool, std::span<const bvh_instance> instances,
              const bvh_options &options = {},
              nx::cancellation_token token = {},
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
  /// Through a top level: the instance's custom index and its place in the
  /// top level's input.
  u32 instance = 0;
  u32 instance_index = 0;
};

/// The nearest triangle a ray meets in a validated block, every triangle
/// opaque: the CPU reference the shader traversal is tested against.
/// Watertight: a ray through an edge two triangles share meets one of them.
[[nodiscard]] bool trace_bvh(std::span<const u8> bvh, const bvh_ray &ray,
                             bvh_hit &out) noexcept;

/// The nearest triangle a ray meets through a validated top level, skipping
/// instances whose mask shares no bit with @p mask; @p bottom_of gives the
/// validated bottom level an instance's bottom_level names.
[[nodiscard]] bool
trace_bvh_scene(std::span<const u8> top,
                nx::function_ref<std::span<const u8>(u64)> bottom_of,
                const bvh_ray &ray, u32 mask, bvh_hit &out) noexcept;

} // namespace nx::gfx
