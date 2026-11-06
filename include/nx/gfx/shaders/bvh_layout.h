#pragma once

// The software ray-tracing route's bottom level (nx/gfx/bvh.h), as C++ writes
// it and shaders read it: one block starting at its address, a header, then
// 4-wide nodes in level order, the triangles in leaf order, the level table
// and, where kept, each node's exact bounds. Slang reads it through the
// nxg_ray_layout module.

#include "nx/gfx/shaders/interop.h"

NX_CONST(uint, NX_BVH_MAGIC, 0x4842584Eu);
NX_CONST(uint, NX_BVH_VERSION, 1u);
NX_CONST(uint, NX_BVH_WIDTH, 4u);
// What a leaf names at most, and in which bits of a child word.
NX_CONST(uint, NX_BVH_MAX_LEAF, 8u);
NX_CONST(uint, NX_BVH_LEAF, 0x80000000u);
NX_CONST(uint, NX_BVH_LEAF_FIRST_MASK, 0x0FFFFFFFu);
NX_CONST(uint, NX_BVH_LEAF_COUNT_SHIFT, 28u);
// A node index where none is: the root's parent.
NX_CONST(uint, NX_BVH_NONE, 0xFFFFFFFFu);
// NxBvhHeader::flags: the block ends with every node's exact bounds.
NX_CONST(uint, NX_BVH_HAS_BOUNDS, 0x1u);
// NxBvhTriangle flags (v2.w): no candidate is reported for it.
NX_CONST(uint, NX_BVH_TRIANGLE_OPAQUE, 0x1u);

// Byte offsets from the block's start; nodes follow the header directly. The
// bounds are those of the whole level, the root's.
struct NxBvhHeader {
  uint magic;
  uint version;
  uint node_count;
  uint triangle_count;
  uint level_count;
  uint flags;
  uint nodes_offset;
  uint triangles_offset;
  uint levels_offset;
  uint bounds_offset;
  float lo_x;
  float lo_y;
  float lo_z;
  float hi_x;
  float hi_y;
  float hi_z;
};
NX_SHARED_SIZE(NxBvhHeader, 64);

// Up to four children, each a box quantized to 8 bits an axis on a grid from
// `origin` in steps of a power of two an axis. `meta` holds each axis's step
// as a float's biased exponent in bytes 0 to 2 and the child count in byte 3;
// lo_* and hi_* a byte of each child's box per axis, child 0 lowest. A child
// word is a node index, always past this node's, or NX_BVH_LEAF with the
// first triangle and the count less one. `parent` is NX_BVH_NONE at the root.
struct NxBvhNode {
  float origin_x;
  float origin_y;
  float origin_z;
  uint meta;
  uint lo_x;
  uint lo_y;
  uint lo_z;
  uint hi_x;
  uint hi_y;
  uint hi_z;
  uint parent;
  uint reserved;
  uint4 children;
};
NX_SHARED_SIZE(NxBvhNode, 64);

// A triangle's corners. v0.w is its index within its geometry, v1.w its
// geometry, v2.w its NX_BVH_TRIANGLE_ flags, each a uint's bits.
struct NxBvhTriangle {
  float4 v0;
  float4 v1;
  float4 v2;
};
NX_SHARED_SIZE(NxBvhTriangle, 48);

struct NxBvhBounds {
  float lo_x;
  float lo_y;
  float lo_z;
  float hi_x;
  float hi_y;
  float hi_z;
};
NX_SHARED_SIZE(NxBvhBounds, 24);

NX_SHARED_FN uint nx_bvh_child_count(uint meta) { return meta >> 24u; }
// The step of @p axis's grid: 2 to the power its exponent byte less 127.
NX_SHARED_FN float nx_bvh_step(uint meta, uint axis) {
  return NX_ASFLOAT(((meta >> (axis * 8u)) & 0xFFu) << 23u);
}
// Child @p child's byte of a packed lo_* or hi_* word.
NX_SHARED_FN uint nx_bvh_byte(uint packed, uint child) {
  return (packed >> (child * 8u)) & 0xFFu;
}
NX_SHARED_FN bool nx_bvh_is_leaf(uint child) {
  return (child & NX_BVH_LEAF) != 0u;
}
NX_SHARED_FN uint nx_bvh_leaf_first(uint child) {
  return child & NX_BVH_LEAF_FIRST_MASK;
}
NX_SHARED_FN uint nx_bvh_leaf_count(uint child) {
  return ((child >> NX_BVH_LEAF_COUNT_SHIFT) & 0x7u) + 1u;
}
