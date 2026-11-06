#pragma once

#include "nx/gfx/types.h"

#include <span>

namespace nx::gfx {

/// Acceleration structures, where device_caps::acceleration_structures says
/// the device has them; elsewhere every call answers invalid, zero or false.
/// A shader traces one through its address (acceleration_structure_address).
///
/// The structure lives in a buffer it owns (acceleration_structure_buffer):
/// barriers and render-graph imports name that buffer. Builds read their
/// inputs in resource_state::AccelerationStructureInput, write the structure
/// and the scratch in AccelerationStructureBuild, and a shader, or a top
/// level built over a bottom one, reads it in AccelerationStructureRead. The
/// caller places those barriers; none is recorded for it.

enum class acceleration_structure_kind : u8 { Bottom, Top };

enum class acceleration_structure_flag : u8 {
  None = 0,
  PreferFastTrace = 1 << 0,
  PreferFastBuild = 1 << 1,
  AllowUpdate = 1 << 2,
  AllowCompaction = 1 << 3,
};
NX_ENUM_FLAGS_DEFINE(acceleration_structure_flag, acceleration_structure_flags)

enum class ray_geometry_flag : u8 {
  None = 0,
  /// No candidate is reported for its triangles: every hit commits.
  Opaque = 1 << 0,
  NoDuplicateAnyHit = 1 << 1,
};
NX_ENUM_FLAGS_DEFINE(ray_geometry_flag, ray_geometry_flags)

/// One geometry of a bottom level: triangles over positions of three f32
/// each, `vertex_stride` bytes apart, indexed by `indices` from
/// `index_address`, or in order when that is 0. `transform_address`, where
/// not 0, holds 12 f32: rows of a 3x4 matrix applied to the positions.
/// Addresses are buffer_address() values of buffers made with
/// buffer_usage::AccelerationStructureInput and DeviceAddress.
struct ray_triangles {
  u64 vertex_address = 0;
  u32 vertex_stride = 12;
  u32 vertex_count = 0;
  u64 index_address = 0;
  index_type indices = index_type::U32;
  u32 triangle_count = 0;
  u64 transform_address = 0;
  ray_geometry_flags flags = ray_geometry_flag::Opaque;
};

struct bottom_level_desc {
  std::span<const ray_triangles> geometries;
  acceleration_structure_flags flags =
      acceleration_structure_flag::PreferFastTrace;
};

/// `instance_count` ray_instance records at `instance_address`, 16 aligned.
struct top_level_desc {
  u64 instance_address = 0;
  u32 instance_count = 0;
  acceleration_structure_flags flags =
      acceleration_structure_flag::PreferFastTrace;
};

/// Bit for bit what ray_instance::flags holds.
enum class ray_instance_flag : u8 {
  None = 0,
  TriangleCullDisable = 1 << 0,
  TriangleFlipFacing = 1 << 1,
  ForceOpaque = 1 << 2,
  ForceNoOpaque = 1 << 3,
};
NX_ENUM_FLAGS_DEFINE(ray_instance_flag, ray_instance_flags)

/// One instance of a top level, as the GPU reads it: the rows of its 3x4
/// transform from the bottom level's space, its custom index (the low 24
/// bits; a shader's CommittedInstanceID) and mask (the top 8; a ray skips an
/// instance whose mask shares no bit with its own), 24 bits the library
/// leaves zero and its ray_instance_flags in the top 8, and the address of
/// its bottom level.
struct ray_instance {
  f32 transform[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  u32 custom_index_and_mask = 0xFF000000u;
  u32 flags = 0;
  u64 bottom_level = 0;
};
static_assert(sizeof(ray_instance) == 64);

[[nodiscard]] constexpr ray_instance
make_ray_instance(const u64 bottom_level, const u32 custom_index,
                  const u8 mask = 0xFF,
                  const ray_instance_flags flags = ray_instance_flag::None) {
  ray_instance out;
  out.custom_index_and_mask =
      (custom_index & 0x00FFFFFFu) | (nx::cast<u32>(mask) << 24);
  out.flags = nx::cast<u32>(flags.value()) << 24;
  out.bottom_level = bottom_level;
  return out;
}

/// What a structure built from a description needs: its own bytes, and the
/// scratch a build (or an update, for one AllowUpdate) uses, which the
/// caller provides at an address aligned to
/// device_caps::acceleration_structure_scratch_alignment.
struct acceleration_structure_sizes {
  u64 structure = 0;
  u64 build_scratch = 0;
  u64 update_scratch = 0;
};

struct acceleration_structure_desc {
  nx::string_view name = "acceleration structure";
  acceleration_structure_kind kind = acceleration_structure_kind::Bottom;
  /// acceleration_structure_sizes::structure, at least.
  u64 size = 0;
  /// What every build, update and copy into it names: a build of other
  /// flags is refused.
  acceleration_structure_flags flags =
      acceleration_structure_flag::PreferFastTrace;
};

// Any thread. Counts and flags matter; addresses do not.
[[nodiscard]] acceleration_structure_sizes sizes(const device &dev,
                                                 const bottom_level_desc &desc);
[[nodiscard]] acceleration_structure_sizes sizes(const device &dev,
                                                 const top_level_desc &desc);

// Any thread. Destruction is deferred past the work submitted, and past
// retirement holds, as a buffer's is.
[[nodiscard]] acceleration_structure_handle
create_acceleration_structure(device &dev,
                              const acceleration_structure_desc &desc);
void destroy_acceleration_structure(
    device &dev, acceleration_structure_handle structure) noexcept;
[[nodiscard]] u64 acceleration_structure_address(
    const device &dev, acceleration_structure_handle structure) noexcept;
/// The buffer the structure lives in, for barriers and imports; the
/// structure owns it, and nothing else may destroy it.
[[nodiscard]] buffer_handle
acceleration_structure_buffer(const device &dev,
                              acceleration_structure_handle structure) noexcept;

/// Records building @p target from @p desc, its scratch at
/// @p scratch_address. False, logged and nothing recorded, when the target
/// is of the other kind, too small or made with other flags, or the scratch
/// is misaligned.
[[nodiscard]] bool build_acceleration_structure(
    command_list &cmd, acceleration_structure_handle target,
    const bottom_level_desc &desc, u64 scratch_address);
[[nodiscard]] bool
build_acceleration_structure(command_list &cmd,
                             acceleration_structure_handle target,
                             const top_level_desc &desc, u64 scratch_address);

/// Records updating @p target in place, a structure made AllowUpdate and
/// built already: @p desc names what it was built from, the same counts and
/// flags, with new positions, transforms or instances. Its scratch is
/// acceleration_structure_sizes::update_scratch. Cheaper than a build, and
/// traces slower the further the geometry moved; rebuild now and then. A top
/// level over an updated bottom one is built or updated after it.
[[nodiscard]] bool update_acceleration_structure(
    command_list &cmd, acceleration_structure_handle target,
    const bottom_level_desc &desc, u64 scratch_address);
[[nodiscard]] bool
update_acceleration_structure(command_list &cmd,
                              acceleration_structure_handle target,
                              const top_level_desc &desc, u64 scratch_address);

// Compaction: a structure made AllowCompaction is built, its compacted size
// written into a pool and read back frames later, a structure of that size
// made, the built one copied into it compacted, and the first destroyed.

/// A pool of @p count compacted sizes; destroy_query_pool frees it. Reset
/// what a frame writes (reset_queries) before writing it.
[[nodiscard]] query_pool_handle create_compacted_size_pool(device &dev,
                                                           u32 count);
/// After their builds, in AccelerationStructureRead.
void write_compacted_sizes(
    command_list &cmd,
    std::span<const acceleration_structure_handle> structures,
    query_pool_handle pool, u32 first = 0);
/// Without waiting: false until the GPU has written every size asked for.
[[nodiscard]] bool read_compacted_sizes(const device &dev,
                                        query_pool_handle pool,
                                        std::span<u64> out);
/// Copies @p source (AccelerationStructureRead) into @p target
/// (AccelerationStructureBuild), of the same kind and flags: compacted, for
/// a target made with the size read back, or else whole.
[[nodiscard]] bool
copy_acceleration_structure(command_list &cmd,
                            acceleration_structure_handle source,
                            acceleration_structure_handle target, bool compact);

} // namespace nx::gfx
