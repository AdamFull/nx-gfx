#pragma once

// What C++ and Slang share for ray tracing (nx/gfx/ray.h, nxg_ray.slang):
// the record of one top-level instance as the GPU reads it, and the flags a
// trace and an instance take. Slang reads it through the nxg_ray_layout
// module, which nxg_ray re-exports.

#include "nx/gfx/shaders/interop.h"

// One instance of a top level: the rows of its 3x4 transform from the bottom
// level's space, its custom index (the low 24 bits; NxRayHit::instance) and
// mask (the top 8; a ray skips an instance whose mask shares no bit with its
// own), 24 bits left zero and its NX_RAY_INSTANCE_ flags in the top 8, and
// the address of its bottom level. Laid out as Vulkan's instance record.
struct NxRayInstance {
  float4 row0;
  float4 row1;
  float4 row2;
  uint custom_index_and_mask;
  uint flags;
  uint64_t bottom_level;
};
NX_SHARED_SIZE(NxRayInstance, 64);

// An instance's flags, in the top byte of NxRayInstance::flags.
NX_CONST(uint, NX_RAY_INSTANCE_CULL_DISABLE, 0x1u);
NX_CONST(uint, NX_RAY_INSTANCE_FLIP_FACING, 0x2u);
NX_CONST(uint, NX_RAY_INSTANCE_FORCE_OPAQUE, 0x4u);
NX_CONST(uint, NX_RAY_INSTANCE_FORCE_NO_OPAQUE, 0x8u);

// A trace's flags (nx_trace<Flags>), the values ray queries take: every
// triangle opaque, the first hit committed ends the trace, and back- or
// front-facing triangles skipped.
NX_CONST(uint, NX_RAY_NONE, 0x0u);
NX_CONST(uint, NX_RAY_OPAQUE, 0x1u);
NX_CONST(uint, NX_RAY_FIRST_HIT, 0x4u);
NX_CONST(uint, NX_RAY_CULL_BACK, 0x10u);
NX_CONST(uint, NX_RAY_CULL_FRONT, 0x20u);
