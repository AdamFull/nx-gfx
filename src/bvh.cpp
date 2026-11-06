#include "nx/gfx/bvh.h"

#include "core/foundation/platform/os.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>

namespace nx::gfx {
namespace {

constexpr f32 INF = std::numeric_limits<f32>::infinity();
/// A subtree over more primitives than this is built by a task of its own.
constexpr u32 PARALLEL_PRIMITIVES = 8192;
/// Past this depth a node splits at its median, which bounds the depth by
/// it plus log2 of the primitives left.
constexpr u32 MEDIAN_DEPTH = 40;
/// Primitives a leaf word can name.
constexpr usize MAX_PRIMITIVES = usize{NX_BVH_LEAF_FIRST_MASK} + 1;
/// Instances a top level holds: each keeps its place in the input in the low
/// 24 bits of its flags word, as Vulkan caps a top level.
constexpr usize MAX_INSTANCES = usize{NX_BVH_INSTANCE_INDEX_MASK} + 1;
/// The deepest a block validate_bvh accepts may be, which bounds a
/// traversal's stack.
constexpr u32 MAX_LEVELS = 96;

struct Box {
  f32 lo[3] = {INF, INF, INF};
  f32 hi[3] = {-INF, -INF, -INF};

  void grow(const Box &other) noexcept {
    for (u32 a = 0; a < 3; ++a) {
      lo[a] = std::min(lo[a], other.lo[a]);
      hi[a] = std::max(hi[a], other.hi[a]);
    }
  }
  void grow(const f32 *const point) noexcept {
    for (u32 a = 0; a < 3; ++a) {
      lo[a] = std::min(lo[a], point[a]);
      hi[a] = std::max(hi[a], point[a]);
    }
  }
  [[nodiscard]] bool empty() const noexcept { return lo[0] > hi[0]; }
  [[nodiscard]] f32 area() const noexcept {
    if (empty())
      return 0.0f;
    const f32 x = hi[0] - lo[0];
    const f32 y = hi[1] - lo[1];
    const f32 z = hi[2] - lo[2];
    return x * y + y * z + z * x;
  }
};

/// A node of the binary tree. Over [begin, begin + count) of the order;
/// a leaf where left_count is 0, else its left child is at its slot + 1 and
/// its right at its slot + 2 * left_count, so slots depend on the ranges
/// alone and subtrees built at once write apart.
struct Binary {
  Box box;
  u32 begin = 0;
  u32 count = 0;
  u32 left_count = 0;
};

/// What every level's build shares: the primitives' boxes, the tree over
/// them and the workers it is built on. Each kind of level fills the boxes
/// and writes its own records.
struct Tree {
  bvh_options options;
  nx::thread_pool *pool = nullptr;
  nx::cancellation_token token;
  std::atomic<bool> stopped{false};

  nx::vector<Box> boxes;
  nx::vector<f32> centroids;
  nx::vector<u32> order;
  nx::vector<Binary> nodes;

  [[nodiscard]] bool stop() noexcept {
    if (token.stop_requested())
      stopped.store(true, std::memory_order_relaxed);
    return stopped.load(std::memory_order_relaxed);
  }

  /// Runs body(begin, end) over [0, count) in chunks, on the workers where
  /// there are any.
  template <typename F> void chunks(const usize count, F &&body) {
    constexpr usize GRAIN = 16384;
    if (pool == nullptr || count <= GRAIN) {
      body(usize{0}, count);
      return;
    }
    const usize parts = (count + GRAIN - 1) / GRAIN;
    pool->parallel_for(0, parts, 1, [&](const usize part) {
      body(part * GRAIN, std::min(count, (part + 1) * GRAIN));
    });
  }

  [[nodiscard]] std::optional<bvh_error> check_options() const noexcept {
    if (options.max_leaf == 0 || options.max_leaf > NX_BVH_MAX_LEAF ||
        options.bins < 2 || options.bins > 64)
      return bvh_error::BadOptions;
    return std::nullopt;
  }

  /// Room for @p count primitives, their centroids from their boxes once
  /// those are filled.
  void reserve(const usize count) {
    boxes.resize(count);
    centroids.resize(count * 3);
    order.resize(count);
    nodes.resize(count * 2 - 1);
  }
  void centre(const usize begin, const usize end) noexcept {
    for (usize i = begin; i < end; ++i) {
      for (u32 a = 0; a < 3; ++a)
        centroids[i * 3 + a] = 0.5f * boxes[i].lo[a] + 0.5f * boxes[i].hi[a];
      order[i] = nx::cast<u32>(i);
    }
  }

  void build(u32 slot, u32 begin, u32 end, u32 depth);
  [[nodiscard]] u32 split(u32 begin, u32 end, const Box &centroid_box, f32 area,
                          u32 depth);
  [[nodiscard]] u32 median(u32 begin, u32 end, u32 axis);
};

u32 Tree::median(const u32 begin, const u32 end, const u32 axis) {
  const u32 mid = begin + (end - begin) / 2;
  std::nth_element(order.begin() + begin, order.begin() + mid,
                   order.begin() + end, [&](const u32 a, const u32 b) {
                     const f32 ca = centroids[usize{a} * 3 + axis];
                     const f32 cb = centroids[usize{b} * 3 + axis];
                     return ca < cb || (ca == cb && a < b);
                   });
  return mid;
}

/// Where [begin, end) splits, its order partitioned so; 0 for a leaf.
u32 Tree::split(const u32 begin, const u32 end, const Box &centroid_box,
                const f32 area, const u32 depth) {
  const u32 count = end - begin;
  const u32 max_leaf = options.max_leaf;
  u32 widest = 0;
  for (u32 a = 1; a < 3; ++a)
    if (centroid_box.hi[a] - centroid_box.lo[a] >
        centroid_box.hi[widest] - centroid_box.lo[widest])
      widest = a;
  const f32 widest_extent = centroid_box.hi[widest] - centroid_box.lo[widest];
  if (!(widest_extent > 0.0f)) {
    if (count <= max_leaf)
      return 0;
    return begin + count / 2;
  }
  if (depth >= MEDIAN_DEPTH)
    return median(begin, end, widest);

  constexpr u32 MOST_BINS = 64;
  const u32 bins = options.bins;
  f32 best_cost = INF;
  u32 best_axis = 0;
  u32 best_bin = 0;
  f32 scales[3] = {};
  for (u32 axis = 0; axis < 3; ++axis) {
    const f32 extent = centroid_box.hi[axis] - centroid_box.lo[axis];
    if (!(extent > 0.0f))
      continue;
    const f32 scale = nx::cast<f32>(bins) / extent;
    scales[axis] = scale;
    Box bin_boxes[MOST_BINS];
    u32 bin_counts[MOST_BINS] = {};
    for (u32 i = begin; i < end; ++i) {
      const u32 t = order[i];
      const u32 bin =
          std::min(bins - 1, nx::cast<u32>((centroids[usize{t} * 3 + axis] -
                                            centroid_box.lo[axis]) *
                                           scale));
      bin_boxes[bin].grow(boxes[t]);
      ++bin_counts[bin];
    }
    f32 right_area[MOST_BINS];
    u32 right_count[MOST_BINS];
    Box right;
    u32 right_total = 0;
    for (u32 b = bins; b-- > 1;) {
      right.grow(bin_boxes[b]);
      right_total += bin_counts[b];
      right_area[b] = right.area();
      right_count[b] = right_total;
    }
    Box left;
    u32 left_total = 0;
    for (u32 b = 0; b + 1 < bins; ++b) {
      left.grow(bin_boxes[b]);
      left_total += bin_counts[b];
      if (left_total == 0 || right_count[b + 1] == 0)
        continue;
      const f32 cost =
          area > 0.0f
              ? 1.0f + (left.area() * nx::cast<f32>(left_total) +
                        right_area[b + 1] * nx::cast<f32>(right_count[b + 1])) /
                           area
              : 1.0f + nx::cast<f32>(count);
      if (cost < best_cost) {
        best_cost = cost;
        best_axis = axis;
        best_bin = b;
      }
    }
  }
  if (count <= max_leaf && nx::cast<f32>(count) <= best_cost)
    return 0;
  if (best_cost == INF)
    return median(begin, end, widest);

  const f32 lo = centroid_box.lo[best_axis];
  const f32 scale = scales[best_axis];
  const auto mid = std::partition(
      order.begin() + begin, order.begin() + end, [&](const u32 t) {
        return std::min(
                   bins - 1,
                   nx::cast<u32>((centroids[usize{t} * 3 + best_axis] - lo) *
                                 scale)) <= best_bin;
      });
  const u32 at = nx::cast<u32>(mid - order.begin());
  if (at == begin || at == end)
    return median(begin, end, widest);
  return at;
}

void Tree::build(const u32 slot, const u32 begin, const u32 end,
                 const u32 depth) {
  if (stop())
    return;
  Binary &node = nodes[slot];
  node.begin = begin;
  node.count = end - begin;
  node.left_count = 0;
  Box box;
  Box centroid_box;
  for (u32 i = begin; i < end; ++i) {
    box.grow(boxes[order[i]]);
    centroid_box.grow(&centroids[usize{order[i]} * 3]);
  }
  node.box = box;
  if (node.count == 1)
    return;
  const u32 mid = split(begin, end, centroid_box, box.area(), depth);
  if (mid == 0)
    return;
  node.left_count = mid - begin;
  const u32 left = slot + 1;
  const u32 right = slot + 2 * node.left_count;
  if (pool != nullptr && node.count > PARALLEL_PRIMITIVES) {
    nx::task_group group;
    pool->submit(group, nx::task_priority::Critical,
                 [this, left, begin, mid, depth] {
                   build(left, begin, mid, depth + 1);
                 });
    build(right, mid, end, depth + 1);
    group.wait();
    return;
  }
  build(left, begin, mid, depth + 1);
  build(right, mid, end, depth + 1);
}

/// A 4-wide node being laid out: the binary nodes its children stand for.
struct Wide {
  u32 children[NX_BVH_WIDTH] = {};
  u32 count = 0;
  u32 slot = 0;
  u32 parent = NX_BVH_NONE;
  u32 level = 0;
};

[[nodiscard]] u32 left_of(const u32 slot) noexcept { return slot + 1; }
[[nodiscard]] u32 right_of(const u32 slot, const Binary &node) noexcept {
  return slot + 2 * node.left_count;
}

/// The children a 4-wide node over binary node @p slot takes: its two, then
/// the inner one of largest area split, until there are four or only leaves.
void expand(const nx::vector<Binary> &nodes, const u32 slot, Wide &wide) {
  const Binary &node = nodes[slot];
  wide.slot = slot;
  if (node.left_count == 0) {
    wide.children[0] = slot;
    wide.count = 1;
    return;
  }
  wide.children[0] = left_of(slot);
  wide.children[1] = right_of(slot, node);
  wide.count = 2;
  while (wide.count < NX_BVH_WIDTH) {
    u32 pick = NX_BVH_NONE;
    f32 largest = -1.0f;
    for (u32 i = 0; i < wide.count; ++i) {
      const Binary &child = nodes[wide.children[i]];
      if (child.left_count != 0 && child.box.area() > largest) {
        largest = child.box.area();
        pick = i;
      }
    }
    if (pick == NX_BVH_NONE)
      return;
    const u32 split = wide.children[pick];
    wide.children[pick] = left_of(split);
    wide.children[wide.count++] = right_of(split, nodes[split]);
  }
}

/// What shaders compute for a grid point; exact whether or not the compiler
/// fuses it, as q * step is exact.
[[nodiscard]] f32 decode(const f32 origin, const u32 q, const f32 step) {
  return origin + nx::cast<f32>(q) * step;
}

[[nodiscard]] f32 step_of(const u32 biased) noexcept {
  return std::bit_cast<f32>(biased << 23);
}

/// Quantizes @p children's boxes on a grid from @p box's low corner, each
/// axis's step the smallest power of two that holds them all: every decoded
/// box holds its child's.
void quantize(const Box &box, const Box *const children, const u32 count,
              NxBvhNode &out) {
  u32 meta = count << 24;
  u32 lo_words[3] = {};
  u32 hi_words[3] = {};
  for (u32 axis = 0; axis < 3; ++axis) {
    const f32 origin = box.lo[axis];
    const f64 extent = f64{box.hi[axis]} - f64{origin};
    i32 biased = 1;
    if (extent > 0.0) {
      const i32 e = nx::cast<i32>(std::ceil(std::log2(extent / 255.0)));
      biased = std::clamp(e + 127, 1, 254);
    }
    for (;; ++biased) {
      const f32 step = step_of(nx::cast<u32>(biased));
      bool fits = true;
      u32 lo_word = 0;
      u32 hi_word = 0;
      for (u32 c = 0; c < count && fits; ++c) {
        const f32 lo = children[c].lo[axis];
        const f32 hi = children[c].hi[axis];
        f64 guess = std::floor((f64{lo} - f64{origin}) / f64{step});
        u32 qlo = nx::cast<u32>(std::clamp(guess, 0.0, 255.0));
        while (qlo > 0 && decode(origin, qlo, step) > lo)
          --qlo;
        guess = std::ceil((f64{hi} - f64{origin}) / f64{step});
        if (guess > 255.0) {
          fits = false;
          break;
        }
        u32 qhi = std::max(qlo, nx::cast<u32>(std::max(guess, 0.0)));
        while (qhi <= 255 && decode(origin, qhi, step) < hi)
          ++qhi;
        if (qhi > 255) {
          fits = false;
          break;
        }
        lo_word |= qlo << (c * 8);
        hi_word |= qhi << (c * 8);
      }
      if (fits || biased >= 254) {
        meta |= nx::cast<u32>(biased) << (axis * 8);
        lo_words[axis] = lo_word;
        hi_words[axis] = hi_word;
        break;
      }
    }
  }
  out.origin_x = box.lo[0];
  out.origin_y = box.lo[1];
  out.origin_z = box.lo[2];
  out.meta = meta;
  out.lo_x = lo_words[0];
  out.lo_y = lo_words[1];
  out.lo_z = lo_words[2];
  out.hi_x = hi_words[0];
  out.hi_y = hi_words[1];
  out.hi_z = hi_words[2];
}

/// The block over @p tree: header, nodes in level order, then a record of
/// @p record_size bytes per primitive in leaf order, which
/// write(index in the order, primitive, destination) fills.
template <typename Write>
[[nodiscard]] nx::result<bvh_blob, bvh_error>
lay_out(Tree &tree, const u32 record_size, const u32 kind_flags,
        Write &&write) {
  const nx::vector<Binary> &nodes = tree.nodes;
  nx::vector<Wide> wide;
  wide.reserve(tree.order.size() / 2 + 1);
  wide.emplace_back();
  expand(nodes, 0, wide[0]);

  nx::vector<NxBvhNode> out_nodes;
  out_nodes.reserve(wide.capacity());
  nx::vector<u32> level_starts{0};
  for (usize w = 0; w < wide.size(); ++w) {
    const Wide current = wide[w];
    if (current.level + 1 > level_starts.size())
      level_starts.push_back(nx::cast<u32>(w));
    NxBvhNode node{};
    node.parent = current.parent;
    node.reserved = 0;
    u32 words[NX_BVH_WIDTH] = {NX_BVH_NONE, NX_BVH_NONE, NX_BVH_NONE,
                               NX_BVH_NONE};
    Box child_boxes[NX_BVH_WIDTH];
    for (u32 c = 0; c < current.count; ++c) {
      const u32 slot = current.children[c];
      const Binary &child = nodes[slot];
      child_boxes[c] = child.box;
      if (child.left_count == 0) {
        words[c] = NX_BVH_LEAF |
                   ((child.count - 1) << NX_BVH_LEAF_COUNT_SHIFT) | child.begin;
        continue;
      }
      if (wide.size() + 1 > (usize{1} << 31))
        return nx::err(bvh_error::TooLarge);
      Wide next;
      expand(nodes, slot, next);
      next.parent = nx::cast<u32>(w);
      next.level = current.level + 1;
      words[c] = nx::cast<u32>(wide.size());
      wide.push_back(next);
    }
    quantize(nodes[current.slot].box, child_boxes, current.count, node);
    node.children = uint4{words[0], words[1], words[2], words[3]};
    out_nodes.push_back(node);
  }
  level_starts.push_back(nx::cast<u32>(wide.size()));
  const u32 level_count = nx::cast<u32>(level_starts.size() - 1);
  if (level_count > MAX_LEVELS)
    return nx::err(bvh_error::TooLarge);

  const usize primitives = tree.order.size();
  const u64 nodes_offset = sizeof(NxBvhHeader);
  const u64 primitives_offset =
      nodes_offset + u64{out_nodes.size()} * sizeof(NxBvhNode);
  const u64 levels_offset = primitives_offset + u64{primitives} * record_size;
  const u64 levels_end = levels_offset + u64{level_starts.size()} * sizeof(u32);
  const bool keep = tree.options.keep_bounds;
  const u64 bounds_offset = keep ? levels_end : 0;
  const u64 size =
      keep ? levels_end + u64{out_nodes.size()} * sizeof(NxBvhBounds)
           : levels_end;
  if (size > u64{std::numeric_limits<u32>::max()})
    return nx::err(bvh_error::TooLarge);

  bvh_blob blob;
  blob.bytes = nx::blob<u8>(nx::cast<usize>(size));
  u8 *const base = blob.bytes.data();
  const Box &root = nodes[0].box;
  NxBvhHeader header{};
  header.magic = NX_BVH_MAGIC;
  header.version = NX_BVH_VERSION;
  header.node_count = nx::cast<u32>(out_nodes.size());
  header.primitive_count = nx::cast<u32>(primitives);
  header.level_count = level_count;
  header.flags = kind_flags | (keep ? NX_BVH_HAS_BOUNDS : 0u);
  header.nodes_offset = nx::cast<u32>(nodes_offset);
  header.primitives_offset = nx::cast<u32>(primitives_offset);
  header.levels_offset = nx::cast<u32>(levels_offset);
  header.bounds_offset = nx::cast<u32>(bounds_offset);
  header.lo_x = root.lo[0];
  header.lo_y = root.lo[1];
  header.lo_z = root.lo[2];
  header.hi_x = root.hi[0];
  header.hi_y = root.hi[1];
  header.hi_z = root.hi[2];
  std::memcpy(base, &header, sizeof(header));
  std::memcpy(base + nodes_offset, out_nodes.data(),
              out_nodes.size() * sizeof(NxBvhNode));
  u8 *const records = base + primitives_offset;
  tree.chunks(primitives, [&](const usize begin, const usize end) {
    for (usize i = begin; i < end; ++i)
      write(tree.order[i], records + i * record_size);
  });
  std::memcpy(base + levels_offset, level_starts.data(),
              level_starts.size() * sizeof(u32));
  if (keep) {
    for (usize w = 0; w < wide.size(); ++w) {
      const Box &box = nodes[wide[w].slot].box;
      const NxBvhBounds bounds{box.lo[0], box.lo[1], box.lo[2],
                               box.hi[0], box.hi[1], box.hi[2]};
      std::memcpy(base + bounds_offset + w * sizeof(NxBvhBounds), &bounds,
                  sizeof(bounds));
    }
  }
  blob.node_count = header.node_count;
  blob.primitive_count = header.primitive_count;
  blob.level_count = level_count;
  return blob;
}

/// Builds the tree over boxes the caller has filled, then lays it out.
template <typename Write>
[[nodiscard]] nx::result<bvh_blob, bvh_error>
finish(Tree &tree, const u64 started, const u32 record_size,
       const u32 kind_flags, Write &&write) {
  tree.build(0, 0, nx::cast<u32>(tree.order.size()), 0);
  // Only a cancelled build stops, and its caller throws rather than
  // answering this.
  if (tree.stop())
    return nx::err(bvh_error::Empty);
  auto laid = lay_out(tree, record_size, kind_flags, write);
  if (laid.has_value())
    laid.value().build_ns = nx::os::ticks_ns() - started;
  return laid;
}

/// A triangle of the input: its geometry and its index within it.
struct Source {
  u32 geometry = 0;
  u32 primitive = 0;
};

[[nodiscard]] const f32 *corner(const std::span<const bvh_geometry> geometries,
                                const Source source,
                                const u32 corner) noexcept {
  const bvh_geometry &geometry = geometries[source.geometry];
  return geometry.positions.data() +
         usize{geometry.indices[usize{source.primitive} * 3 + corner]} * 3;
}

[[nodiscard]] nx::result<bvh_blob, bvh_error>
build_triangles(Tree &tree, const std::span<const bvh_geometry> geometries) {
  const u64 started = nx::os::ticks_ns();
  if (const auto refused = tree.check_options())
    return nx::err(*refused);
  usize total = 0;
  for (const bvh_geometry &geometry : geometries) {
    if (geometry.indices.size() % 3 != 0 || geometry.positions.size() % 3 != 0)
      return nx::err(bvh_error::NotTriangles);
    total += geometry.indices.size() / 3;
  }
  if (total == 0)
    return nx::err(bvh_error::Empty);
  if (total > MAX_PRIMITIVES)
    return nx::err(bvh_error::TooLarge);

  nx::vector<Source> sources;
  sources.reserve(total);
  for (u32 g = 0; g < geometries.size(); ++g) {
    const bvh_geometry &geometry = geometries[g];
    const usize vertices = geometry.positions.size() / 3;
    for (const u32 index : geometry.indices)
      if (index >= vertices)
        return nx::err(bvh_error::IndexOutOfRange);
    const u32 triangles = nx::cast<u32>(geometry.indices.size() / 3);
    for (u32 t = 0; t < triangles; ++t)
      sources.push_back({.geometry = g, .primitive = t});
  }

  tree.reserve(total);
  std::atomic<bool> finite{true};
  tree.chunks(total, [&](const usize begin, const usize end) {
    for (usize i = begin; i < end; ++i) {
      Box box;
      for (u32 c = 0; c < 3; ++c) {
        const f32 *const p = corner(geometries, sources[i], c);
        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) ||
            !std::isfinite(p[2]))
          finite.store(false, std::memory_order_relaxed);
        box.grow(p);
      }
      tree.boxes[i] = box;
    }
    tree.centre(begin, end);
  });
  if (!finite.load(std::memory_order_relaxed))
    return nx::err(bvh_error::NonFinite);

  return finish(
      tree, started, sizeof(NxBvhTriangle), 0u,
      [&](const u32 primitive, u8 *const out) {
        const Source source = sources[primitive];
        const f32 *const a = corner(geometries, source, 0);
        const f32 *const b = corner(geometries, source, 1);
        const f32 *const c = corner(geometries, source, 2);
        const u32 flags =
            geometries[source.geometry].opaque ? NX_BVH_TRIANGLE_OPAQUE : 0u;
        NxBvhTriangle triangle;
        triangle.v0 =
            float4{a[0], a[1], a[2], std::bit_cast<f32>(source.primitive)};
        triangle.v1 =
            float4{b[0], b[1], b[2], std::bit_cast<f32>(source.geometry)};
        triangle.v2 = float4{c[0], c[1], c[2], std::bit_cast<f32>(flags)};
        std::memcpy(out, &triangle, sizeof(triangle));
      });
}

/// Rows of a 3x4 transform, in double.
struct Rows {
  f64 m[3][4] = {};
};

[[nodiscard]] Rows rows_of(const NxRayInstance &instance) noexcept {
  const float4 r[3] = {instance.row0, instance.row1, instance.row2};
  Rows out;
  for (u32 i = 0; i < 3; ++i) {
    out.m[i][0] = r[i].x;
    out.m[i][1] = r[i].y;
    out.m[i][2] = r[i].z;
    out.m[i][3] = r[i].w;
  }
  return out;
}

/// The inverse of an affine transform, or nothing where it has none.
[[nodiscard]] std::optional<Rows> inverse(const Rows &a) noexcept {
  const auto &m = a.m;
  const f64 c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
  const f64 c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
  const f64 c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
  const f64 det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
  if (!std::isfinite(det) || std::fabs(det) < 1e-30)
    return std::nullopt;
  const f64 inv = 1.0 / det;
  Rows out;
  auto &r = out.m;
  r[0][0] = c00 * inv;
  r[1][0] = c01 * inv;
  r[2][0] = c02 * inv;
  r[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv;
  r[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv;
  r[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv;
  r[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv;
  r[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv;
  r[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv;
  for (u32 i = 0; i < 3; ++i)
    r[i][3] = -(r[i][0] * m[0][3] + r[i][1] * m[1][3] + r[i][2] * m[2][3]);
  for (const auto &row : r)
    for (const f64 value : row)
      if (!std::isfinite(value) || std::fabs(value) > 3.0e38)
        return std::nullopt;
  return out;
}

[[nodiscard]] nx::result<bvh_blob, bvh_error>
build_instances(Tree &tree, const std::span<const bvh_instance> instances) {
  const u64 started = nx::os::ticks_ns();
  if (const auto refused = tree.check_options())
    return nx::err(*refused);
  // A shader enters one instance at a time, so a top level's leaves hold one.
  tree.options.max_leaf = 1;
  if (instances.empty())
    return nx::err(bvh_error::Empty);
  if (instances.size() > MAX_INSTANCES)
    return nx::err(bvh_error::TooLarge);

  const usize count = instances.size();
  nx::vector<NxBvhInstance> records(count);
  tree.reserve(count);
  std::atomic<bool> valid{true};
  tree.chunks(count, [&](const usize begin, const usize end) {
    for (usize i = begin; i < end; ++i) {
      const bvh_instance &instance = instances[i];
      const NxBvhBounds &b = instance.bounds;
      const Rows rows = rows_of(instance.instance);
      const std::optional<Rows> inverted = inverse(rows);
      const f64 lo[3] = {b.lo_x, b.lo_y, b.lo_z};
      const f64 hi[3] = {b.hi_x, b.hi_y, b.hi_z};
      if (!inverted || !(lo[0] <= hi[0] && lo[1] <= hi[1] && lo[2] <= hi[2]) ||
          !std::isfinite(lo[0] + lo[1] + lo[2] + hi[0] + hi[1] + hi[2]) ||
          (instance.instance.flags & 0x00FFFFFFu) != 0) {
        valid.store(false, std::memory_order_relaxed);
        continue;
      }
      // The box's eight corners placed, held by a box rounded outward.
      Box box;
      for (u32 corner = 0; corner < 8; ++corner) {
        const f64 p[3] = {(corner & 1) ? hi[0] : lo[0],
                          (corner & 2) ? hi[1] : lo[1],
                          (corner & 4) ? hi[2] : lo[2]};
        for (u32 r = 0; r < 3; ++r) {
          const f64 placed = rows.m[r][0] * p[0] + rows.m[r][1] * p[1] +
                             rows.m[r][2] * p[2] + rows.m[r][3];
          const f32 down = std::nextafter(nx::cast<f32>(placed), -INF);
          const f32 up = std::nextafter(nx::cast<f32>(placed), INF);
          box.lo[r] = std::min(box.lo[r], down);
          box.hi[r] = std::max(box.hi[r], up);
        }
      }
      if (!std::isfinite(box.lo[0] + box.lo[1] + box.lo[2] + box.hi[0] +
                         box.hi[1] + box.hi[2])) {
        valid.store(false, std::memory_order_relaxed);
        continue;
      }
      tree.boxes[i] = box;
      const auto &inv = inverted->m;
      NxBvhInstance &record = records[i];
      record.row0 = float4{nx::cast<f32>(inv[0][0]), nx::cast<f32>(inv[0][1]),
                           nx::cast<f32>(inv[0][2]), nx::cast<f32>(inv[0][3])};
      record.row1 = float4{nx::cast<f32>(inv[1][0]), nx::cast<f32>(inv[1][1]),
                           nx::cast<f32>(inv[1][2]), nx::cast<f32>(inv[1][3])};
      record.row2 = float4{nx::cast<f32>(inv[2][0]), nx::cast<f32>(inv[2][1]),
                           nx::cast<f32>(inv[2][2]), nx::cast<f32>(inv[2][3])};
      record.bottom_level = instance.instance.bottom_level;
      record.custom_index_and_mask = instance.instance.custom_index_and_mask;
      record.flags = instance.instance.flags | nx::cast<u32>(i);
    }
    tree.centre(begin, end);
  });
  if (!valid.load(std::memory_order_relaxed))
    return nx::err(bvh_error::BadInstance);

  return finish(tree, started, sizeof(NxBvhInstance), NX_BVH_TOP,
                [&](const u32 primitive, u8 *const out) {
                  std::memcpy(out, &records[primitive], sizeof(NxBvhInstance));
                });
}

/// Runs @p build on a fresh tree, on @p pool's workers, and keeps a cancelled
/// build from answering.
template <typename Input, typename Build>
[[nodiscard]] nx::future<nx::result<bvh_blob, bvh_error>>
on_pool(nx::thread_pool &pool, const std::span<const Input> input,
        const bvh_options &options, nx::cancellation_token token,
        const nx::task_priority priority, Build build) {
  return pool.async(
      priority, std::move(token),
      [&pool, input, options, build](const nx::cancellation_token cancel)
          -> nx::result<bvh_blob, bvh_error> {
        Tree tree;
        tree.options = options;
        tree.pool = &pool;
        tree.token = cancel;
        auto built = build(tree, input);
        cancel.throw_if_requested();
        return built;
      });
}

} // namespace

nx::string_view bvh_error_name(const bvh_error error) noexcept {
  switch (error) {
  case bvh_error::Empty:
    return "nothing to build";
  case bvh_error::NotTriangles:
    return "an index or position count not a multiple of three";
  case bvh_error::IndexOutOfRange:
    return "an index past the positions";
  case bvh_error::NonFinite:
    return "a position not finite";
  case bvh_error::BadInstance:
    return "an instance whose transform has no inverse or whose bounds are "
           "not a box";
  case bvh_error::TooLarge:
    return "more than the block can name";
  case bvh_error::BadOptions:
    return "options out of range";
  case bvh_error::Truncated:
    return "truncated";
  case bvh_error::BadMagic:
    return "not a BVH block";
  case bvh_error::BadVersion:
    return "a BVH block of another version";
  case bvh_error::BadLayout:
    return "a malformed BVH block";
  }
  return "unknown";
}

nx::result<bvh_blob, bvh_error>
build_bvh_here(const std::span<const bvh_geometry> geometries,
               const bvh_options &options) {
  Tree tree;
  tree.options = options;
  return build_triangles(tree, geometries);
}

nx::future<nx::result<bvh_blob, bvh_error>>
build_bvh(nx::thread_pool &pool, const std::span<const bvh_geometry> geometries,
          const bvh_options &options, nx::cancellation_token token,
          const nx::task_priority priority) {
  return on_pool(pool, geometries, options, std::move(token), priority,
                 [](Tree &tree, std::span<const bvh_geometry> input) {
                   return build_triangles(tree, input);
                 });
}

nx::result<bvh_blob, bvh_error>
build_bvh_top_here(const std::span<const bvh_instance> instances,
                   const bvh_options &options) {
  Tree tree;
  tree.options = options;
  return build_instances(tree, instances);
}

nx::future<nx::result<bvh_blob, bvh_error>>
build_bvh_top(nx::thread_pool &pool,
              const std::span<const bvh_instance> instances,
              const bvh_options &options, nx::cancellation_token token,
              const nx::task_priority priority) {
  return on_pool(pool, instances, options, std::move(token), priority,
                 [](Tree &tree, std::span<const bvh_instance> input) {
                   return build_instances(tree, input);
                 });
}

namespace {

[[nodiscard]] bool finite(const float4 &v) noexcept {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

[[nodiscard]] bool finite4(const float4 &v) noexcept {
  return finite(v) && std::isfinite(v.w);
}

} // namespace

nx::result<const NxBvhHeader *, bvh_error>
validate_bvh(const std::span<const u8> bytes) noexcept {
  if (bytes.size() < sizeof(NxBvhHeader))
    return nx::err(bvh_error::Truncated);
  if (reinterpret_cast<uintptr_t>(bytes.data()) % 16 != 0)
    return nx::err(bvh_error::BadLayout);
  const auto *const header =
      reinterpret_cast<const NxBvhHeader *>(bytes.data());
  if (header->magic != NX_BVH_MAGIC)
    return nx::err(bvh_error::BadMagic);
  if (header->version != NX_BVH_VERSION)
    return nx::err(bvh_error::BadVersion);
  const u64 nodes = header->node_count;
  const u64 primitives = header->primitive_count;
  const u64 levels = header->level_count;
  if (nodes == 0 || primitives == 0 || primitives > MAX_PRIMITIVES ||
      levels == 0 || levels > MAX_LEVELS || levels > nodes ||
      (header->flags & ~(NX_BVH_HAS_BOUNDS | NX_BVH_TOP)) != 0)
    return nx::err(bvh_error::BadLayout);
  const bool top = (header->flags & NX_BVH_TOP) != 0;
  const u64 record_size = top ? sizeof(NxBvhInstance) : sizeof(NxBvhTriangle);
  const u64 primitives_offset = sizeof(NxBvhHeader) + nodes * sizeof(NxBvhNode);
  const u64 levels_offset = primitives_offset + primitives * record_size;
  const u64 levels_end = levels_offset + (levels + 1) * sizeof(u32);
  const bool kept = (header->flags & NX_BVH_HAS_BOUNDS) != 0;
  const u64 end = kept ? levels_end + nodes * sizeof(NxBvhBounds) : levels_end;
  if (header->nodes_offset != sizeof(NxBvhHeader) ||
      header->primitives_offset != primitives_offset ||
      header->levels_offset != levels_offset ||
      header->bounds_offset != (kept ? levels_end : 0))
    return nx::err(bvh_error::BadLayout);
  if (bytes.size() < end)
    return nx::err(bvh_error::Truncated);
  if (bytes.size() != end)
    return nx::err(bvh_error::BadLayout);

  const auto *const node =
      reinterpret_cast<const NxBvhNode *>(bytes.data() + sizeof(NxBvhHeader));
  u32 level_start[MAX_LEVELS + 1];
  std::memcpy(level_start, bytes.data() + levels_offset,
              nx::cast<usize>(levels + 1) * sizeof(u32));
  if (level_start[0] != 0 || level_start[1] != 1 ||
      level_start[levels] != nodes)
    return nx::err(bvh_error::BadLayout);
  for (u64 l = 0; l < levels; ++l)
    if (level_start[l] >= level_start[l + 1])
      return nx::err(bvh_error::BadLayout);

  nx::vector<u64> covered(nx::cast<usize>((primitives + 63) / 64), 0);
  const auto seen = [&](const u64 p) {
    return (covered[nx::cast<usize>(p / 64)] >> (p % 64)) & 1u;
  };
  u64 level = 0;
  for (u32 n = 0; n < nodes; ++n) {
    while (n >= level_start[level + 1])
      ++level;
    const NxBvhNode &current = node[n];
    const u32 count = nx_bvh_child_count(current.meta);
    if (count == 0 || count > NX_BVH_WIDTH)
      return nx::err(bvh_error::BadLayout);
    if ((n == 0) != (current.parent == NX_BVH_NONE))
      return nx::err(bvh_error::BadLayout);
    for (u32 axis = 0; axis < 3; ++axis) {
      const u32 biased = (current.meta >> (axis * 8)) & 0xFFu;
      if (biased == 0 || biased == 255)
        return nx::err(bvh_error::BadLayout);
    }
    if (!std::isfinite(current.origin_x) || !std::isfinite(current.origin_y) ||
        !std::isfinite(current.origin_z))
      return nx::err(bvh_error::BadLayout);
    const u32 words[NX_BVH_WIDTH] = {current.children.x, current.children.y,
                                     current.children.z, current.children.w};
    const u32 lo[3] = {current.lo_x, current.lo_y, current.lo_z};
    const u32 hi[3] = {current.hi_x, current.hi_y, current.hi_z};
    for (u32 c = 0; c < NX_BVH_WIDTH; ++c) {
      const u32 word = words[c];
      if (c >= count) {
        if (word != NX_BVH_NONE)
          return nx::err(bvh_error::BadLayout);
        continue;
      }
      for (u32 axis = 0; axis < 3; ++axis)
        if (nx_bvh_byte(lo[axis], c) > nx_bvh_byte(hi[axis], c))
          return nx::err(bvh_error::BadLayout);
      if (nx_bvh_is_leaf(word)) {
        const u64 first = nx_bvh_leaf_first(word);
        const u64 leaf = nx_bvh_leaf_count(word);
        if (first + leaf > primitives)
          return nx::err(bvh_error::BadLayout);
        for (u64 p = first; p < first + leaf; ++p) {
          if (seen(p))
            return nx::err(bvh_error::BadLayout);
          covered[nx::cast<usize>(p / 64)] |= u64{1} << (p % 64);
        }
        continue;
      }
      if (word <= n || word >= nodes || word < level_start[level + 1] ||
          (level + 2 <= levels && word >= level_start[level + 2]) ||
          (level + 1 >= levels) || node[word].parent != n)
        return nx::err(bvh_error::BadLayout);
    }
  }
  const u8 *const records = bytes.data() + primitives_offset;
  for (u64 p = 0; p < primitives; ++p) {
    if (!seen(p))
      return nx::err(bvh_error::BadLayout);
    if (top) {
      const auto &instance =
          reinterpret_cast<const NxBvhInstance *>(records)[p];
      if (!finite4(instance.row0) || !finite4(instance.row1) ||
          !finite4(instance.row2))
        return nx::err(bvh_error::BadLayout);
      continue;
    }
    const auto &triangle = reinterpret_cast<const NxBvhTriangle *>(records)[p];
    if (!finite(triangle.v0) || !finite(triangle.v1) || !finite(triangle.v2) ||
        (std::bit_cast<u32>(triangle.v2.w) & ~NX_BVH_TRIANGLE_OPAQUE) != 0)
      return nx::err(bvh_error::BadLayout);
  }
  return header;
}

namespace {

/// A ray as traversal uses it, its direction's reciprocal made once.
struct Walk {
  f32 origin[3];
  f32 direction[3];
  f32 inverse[3];
  f32 tmin;
};

[[nodiscard]] Walk walk_of(const f32 *const origin, const f32 *const direction,
                           const f32 tmin) noexcept {
  Walk out;
  for (u32 a = 0; a < 3; ++a) {
    out.origin[a] = origin[a];
    out.direction[a] = direction[a];
    out.inverse[a] = 1.0f / direction[a];
  }
  out.tmin = tmin;
  return out;
}

/// Woop, Benthin and Wald's watertight test: the ray's space sheared so it
/// runs along +z, the edge functions evaluated in double where float gives 0.
[[nodiscard]] bool hit_triangle(const Walk &ray, const NxBvhTriangle &tri,
                                const f32 best, bvh_hit &out) noexcept {
  const f32 *const d = ray.direction;
  u32 kz = 0;
  if (std::fabs(d[1]) > std::fabs(d[kz]))
    kz = 1;
  if (std::fabs(d[2]) > std::fabs(d[kz]))
    kz = 2;
  u32 kx = (kz + 1) % 3;
  u32 ky = (kx + 1) % 3;
  if (d[kz] < 0.0f)
    std::swap(kx, ky);
  const f32 sx = d[kx] / d[kz];
  const f32 sy = d[ky] / d[kz];
  const f32 sz = 1.0f / d[kz];
  const auto at = [&](const float4 &v, const u32 k) {
    return (k == 0 ? v.x : k == 1 ? v.y : v.z) - ray.origin[k];
  };
  const f32 ax = at(tri.v0, kx) - sx * at(tri.v0, kz);
  const f32 ay = at(tri.v0, ky) - sy * at(tri.v0, kz);
  const f32 bx = at(tri.v1, kx) - sx * at(tri.v1, kz);
  const f32 by = at(tri.v1, ky) - sy * at(tri.v1, kz);
  const f32 cx = at(tri.v2, kx) - sx * at(tri.v2, kz);
  const f32 cy = at(tri.v2, ky) - sy * at(tri.v2, kz);
  f32 u = cx * by - cy * bx;
  f32 v = ax * cy - ay * cx;
  f32 w = bx * ay - by * ax;
  if (u == 0.0f || v == 0.0f || w == 0.0f) {
    u = nx::cast<f32>(f64{cx} * f64{by} - f64{cy} * f64{bx});
    v = nx::cast<f32>(f64{ax} * f64{cy} - f64{ay} * f64{cx});
    w = nx::cast<f32>(f64{bx} * f64{ay} - f64{by} * f64{ax});
  }
  if ((u < 0.0f || v < 0.0f || w < 0.0f) && (u > 0.0f || v > 0.0f || w > 0.0f))
    return false;
  const f32 det = u + v + w;
  if (det == 0.0f)
    return false;
  const f32 az = sz * at(tri.v0, kz);
  const f32 bz = sz * at(tri.v1, kz);
  const f32 cz = sz * at(tri.v2, kz);
  const f32 t = (u * az + v * bz + w * cz) / det;
  if (!(t > ray.tmin) || !(t < best))
    return false;
  out.t = t;
  out.u = v / det;
  out.v = w / det;
  out.primitive = std::bit_cast<u32>(tri.v0.w);
  out.geometry = std::bit_cast<u32>(tri.v1.w);
  return true;
}

/// Walks @p block's nodes for @p ray up to @p best, calling
/// leaf(first, count) for each leaf whose box the ray meets; leaf may lower
/// best.
template <typename Leaf>
void walk(const u8 *const block, const Walk &ray, f32 &best, Leaf &&leaf) {
  const auto *const header = reinterpret_cast<const NxBvhHeader *>(block);
  const auto *const nodes =
      reinterpret_cast<const NxBvhNode *>(block + header->nodes_offset);
  // Rounding in the slab test can lose a box a ray grazes; widen its far
  // distance by a few ulps (Ize).
  constexpr f32 WIDEN = 1.0f + 6.0f * std::numeric_limits<f32>::epsilon();
  u32 stack[NX_BVH_WIDTH * MAX_LEVELS];
  u32 depth = 0;
  stack[depth++] = 0;
  while (depth > 0) {
    const NxBvhNode &node = nodes[stack[--depth]];
    const u32 count = nx_bvh_child_count(node.meta);
    const f32 origin[3] = {node.origin_x, node.origin_y, node.origin_z};
    const u32 lo[3] = {node.lo_x, node.lo_y, node.lo_z};
    const u32 hi[3] = {node.hi_x, node.hi_y, node.hi_z};
    const u32 words[NX_BVH_WIDTH] = {node.children.x, node.children.y,
                                     node.children.z, node.children.w};
    for (u32 c = 0; c < count; ++c) {
      f32 near_t = ray.tmin;
      f32 far_t = best;
      for (u32 axis = 0; axis < 3; ++axis) {
        const f32 step = nx_bvh_step(node.meta, axis);
        const f32 low = decode(origin[axis], nx_bvh_byte(lo[axis], c), step);
        const f32 high = decode(origin[axis], nx_bvh_byte(hi[axis], c), step);
        const f32 t0 = (low - ray.origin[axis]) * ray.inverse[axis];
        const f32 t1 = (high - ray.origin[axis]) * ray.inverse[axis];
        near_t = std::fmax(near_t, std::fmin(t0, t1));
        far_t = std::fmin(far_t, std::fmax(t0, t1) * WIDEN);
      }
      if (near_t > far_t)
        continue;
      const u32 word = words[c];
      if (!nx_bvh_is_leaf(word)) {
        if (depth < std::size(stack))
          stack[depth++] = word;
        continue;
      }
      leaf(nx_bvh_leaf_first(word), nx_bvh_leaf_count(word));
    }
  }
}

/// Tests leaf triangles [first, first + count) of @p block.
void test_triangles(const u8 *const block, const Walk &ray, f32 &best,
                    const u32 first, const u32 count, bool &found,
                    bvh_hit &out) noexcept {
  const auto *const header = reinterpret_cast<const NxBvhHeader *>(block);
  const auto *const triangles = reinterpret_cast<const NxBvhTriangle *>(
      block + header->primitives_offset);
  for (u32 t = first; t < first + count; ++t)
    if (hit_triangle(ray, triangles[t], best, out)) {
      best = out.t;
      found = true;
    }
}

[[nodiscard]] f32 apply(const float4 &row, const f32 *const v,
                        const f32 w) noexcept {
  return row.x * v[0] + row.y * v[1] + row.z * v[2] + row.w * w;
}

} // namespace

bool trace_bvh(const std::span<const u8> bvh, const bvh_ray &ray,
               bvh_hit &out) noexcept {
  if (bvh.size() < sizeof(NxBvhHeader))
    return false;
  const Walk walked = walk_of(ray.origin, ray.direction, ray.tmin);
  f32 best = ray.tmax;
  bool found = false;
  walk(bvh.data(), walked, best, [&](const u32 first, const u32 count) {
    test_triangles(bvh.data(), walked, best, first, count, found, out);
  });
  return found;
}

bool trace_bvh_scene(const std::span<const u8> top,
                     const nx::function_ref<std::span<const u8>(u64)> bottom_of,
                     const bvh_ray &ray, const u32 mask,
                     bvh_hit &out) noexcept {
  if (top.size() < sizeof(NxBvhHeader))
    return false;
  const auto *const header = reinterpret_cast<const NxBvhHeader *>(top.data());
  const auto *const instances = reinterpret_cast<const NxBvhInstance *>(
      top.data() + header->primitives_offset);
  const Walk world = walk_of(ray.origin, ray.direction, ray.tmin);
  f32 best = ray.tmax;
  bool found = false;
  walk(top.data(), world, best, [&](const u32 first, const u32 count) {
    for (u32 i = first; i < first + count; ++i) {
      const NxBvhInstance &instance = instances[i];
      if (((instance.custom_index_and_mask >> 24) & mask) == 0)
        continue;
      const std::span<const u8> bottom = bottom_of(instance.bottom_level);
      if (bottom.size() < sizeof(NxBvhHeader))
        continue;
      // Into the instance's space, the direction left unnormalized so a
      // distance along it is the same in both.
      const f32 origin[3] = {apply(instance.row0, ray.origin, 1.0f),
                             apply(instance.row1, ray.origin, 1.0f),
                             apply(instance.row2, ray.origin, 1.0f)};
      const f32 direction[3] = {apply(instance.row0, ray.direction, 0.0f),
                                apply(instance.row1, ray.direction, 0.0f),
                                apply(instance.row2, ray.direction, 0.0f)};
      const Walk local = walk_of(origin, direction, ray.tmin);
      bool hit = false;
      walk(bottom.data(), local, best,
           [&](const u32 leaf_first, const u32 leaf_count) {
             test_triangles(bottom.data(), local, best, leaf_first, leaf_count,
                            hit, out);
           });
      if (hit) {
        found = true;
        out.instance = instance.custom_index_and_mask & 0x00FFFFFFu;
        out.instance_index = instance.flags & NX_BVH_INSTANCE_INDEX_MASK;
      }
    }
  });
  return found;
}

} // namespace nx::gfx
