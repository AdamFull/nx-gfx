// The software BVH's GPU half beyond tracing (nx/gfx/bvh.h): refitting a
// bottom level whose vertices moved and whose triangles did not. A consumer
// writes the entry points, one thread a triangle for
// nx_bvh_refit_triangle, then one dispatch a level, deepest first, one thread
// a node for nx_bvh_refit_node, a barrier on the block between each. The
// block must have been built with bvh_options::keep_bounds: a quantized box
// cannot be refitted from quantized children, so each node's exact box is
// read and rewritten.
module nxg_bvh;

__exported import nxg_ray_layout;
import nxg_ptr;

/// The parts of a block a refit reads and writes.
struct NxBvhParts {
    uint64_t header;
    NxPtr<NxBvhNode> nodes;
    NxPtr<NxBvhTriangle> triangles;
    NxPtr<NxBvhBounds> bounds;
};

NxBvhParts nx_bvh_parts(uint64_t block) {
    NxPtr<NxBvhHeader> header;
    header.value = block;
    const NxBvhHeader read = header.load(0u);
    NxBvhParts parts;
    parts.header = block;
    parts.nodes.value = block + uint64_t(read.nodes_offset);
    parts.triangles.value = block + uint64_t(read.primitives_offset);
    parts.bounds.value = block + uint64_t(read.bounds_offset);
    return parts;
}

/// Rewrites triangle record @p record's corners from where its geometry's
/// vertices are now (@p geometries, by the geometry the record names),
/// keeping what else the record holds.
public void nx_bvh_refit_triangle(uint64_t block,
                                  NxPtr<NxBvhRefitGeometry> geometries,
                                  uint record) {
    const NxBvhParts parts = nx_bvh_parts(block);
    NxBvhTriangle triangle = parts.triangles.load(record);
    const NxBvhRefitGeometry geometry =
        geometries.load(asuint(triangle.v1.w));
    NxPtr<uint> indices;
    indices.value = geometry.indices;
    NxPtr<float> positions;
    positions.value = geometry.positions;
    const uint first = asuint(triangle.v0.w) * 3u;
    float3 corners[3];
    for (uint c = 0u; c < 3u; ++c) {
        const uint vertex = indices.load(first + c) * 3u;
        corners[c] = float3(positions.load(vertex), positions.load(vertex + 1u),
                            positions.load(vertex + 2u));
    }
    triangle.v0.xyz = corners[0];
    triangle.v1.xyz = corners[1];
    triangle.v2.xyz = corners[2];
    parts.triangles.store(record, triangle);
}

/// What a shader decodes a grid point to: exact whether or not it is fused,
/// as q * step is exact, so the check below is the decode itself.
float nx_bvh_decode(float origin, uint q, float step) {
    return origin + float(q) * step;
}

/// Re-boxes node @p node from its children as they are now: a leaf's from
/// its triangles' corners, a node's from the exact box its own refit wrote.
/// Its children's boxes are quantized again on the grid of its new box, the
/// smallest power-of-two step an axis that holds them all, each decoded box
/// checked to hold its child; its exact box is written, and the root's goes
/// into the header too.
public void nx_bvh_refit_node(uint64_t block, uint node_index) {
    const NxBvhParts parts = nx_bvh_parts(block);
    NxBvhNode node = parts.nodes.load(node_index);
    const uint count = nx_bvh_child_count(node.meta);
    const uint words[4] = {node.children.x, node.children.y, node.children.z,
                           node.children.w};
    float3 child_lo[4];
    float3 child_hi[4];
    float3 lo = float3(3.4e38);
    float3 hi = float3(-3.4e38);
    for (uint c = 0u; c < count; ++c) {
        float3 clo = float3(3.4e38);
        float3 chi = float3(-3.4e38);
        const uint word = words[c];
        if (nx_bvh_is_leaf(word)) {
            const uint first = nx_bvh_leaf_first(word);
            const uint end = first + nx_bvh_leaf_count(word);
            for (uint t = first; t < end; ++t) {
                const NxBvhTriangle triangle = parts.triangles.load(t);
                clo = min(clo, min(triangle.v0.xyz,
                                   min(triangle.v1.xyz, triangle.v2.xyz)));
                chi = max(chi, max(triangle.v0.xyz,
                                   max(triangle.v1.xyz, triangle.v2.xyz)));
            }
        } else {
            const NxBvhBounds box = parts.bounds.load(word);
            clo = float3(box.lo_x, box.lo_y, box.lo_z);
            chi = float3(box.hi_x, box.hi_y, box.hi_z);
        }
        child_lo[c] = clo;
        child_hi[c] = chi;
        lo = min(lo, clo);
        hi = max(hi, chi);
    }

    uint meta = count << 24u;
    uint lo_words[3] = {0u, 0u, 0u};
    uint hi_words[3] = {0u, 0u, 0u};
    for (uint axis = 0u; axis < 3u; ++axis) {
        const float origin = lo[axis];
        const float extent = hi[axis] - origin;
        int biased = 1;
        if (extent > 0.0)
            biased = clamp(int(ceil(log2(extent / 255.0))) + 127, 1, 254);
        for (; biased <= 254; ++biased) {
            const float step = asfloat(uint(biased) << 23u);
            bool fits = true;
            uint lo_word = 0u;
            uint hi_word = 0u;
            for (uint c = 0u; c < count && fits; ++c) {
                const float clo = child_lo[c][axis];
                const float chi = child_hi[c][axis];
                uint qlo = uint(clamp(floor((clo - origin) / step), 0.0, 255.0));
                while (qlo > 0u && nx_bvh_decode(origin, qlo, step) > clo)
                    --qlo;
                const float guess = ceil((chi - origin) / step);
                if (guess > 255.0) {
                    fits = false;
                    break;
                }
                uint qhi = max(qlo, uint(max(guess, 0.0)));
                while (qhi <= 255u && nx_bvh_decode(origin, qhi, step) < chi)
                    ++qhi;
                if (qhi > 255u) {
                    fits = false;
                    break;
                }
                lo_word |= qlo << (c * 8u);
                hi_word |= qhi << (c * 8u);
            }
            if (fits || biased == 254) {
                meta |= uint(biased) << (axis * 8u);
                lo_words[axis] = lo_word;
                hi_words[axis] = hi_word;
                break;
            }
        }
    }
    node.origin_x = lo.x;
    node.origin_y = lo.y;
    node.origin_z = lo.z;
    node.meta = meta;
    node.lo_x = lo_words[0];
    node.lo_y = lo_words[1];
    node.lo_z = lo_words[2];
    node.hi_x = hi_words[0];
    node.hi_y = hi_words[1];
    node.hi_z = hi_words[2];
    parts.nodes.store(node_index, node);

    NxBvhBounds box;
    box.lo_x = lo.x;
    box.lo_y = lo.y;
    box.lo_z = lo.z;
    box.hi_x = hi.x;
    box.hi_y = hi.y;
    box.hi_z = hi.z;
    box.pad0 = 0.0;
    box.pad1 = 0.0;
    parts.bounds.store(node_index, box);
    if (node_index == 0u) {
        // NxBvhHeader's lo_x to hi_z, from its eleventh word: one float at a
        // time, as the eleventh word lies on 8, not on 16.
        NxPtr<float> header_bounds;
        header_bounds.value = parts.header + 40u;
        header_bounds.store(0u, lo.x);
        header_bounds.store(1u, lo.y);
        header_bounds.store(2u, lo.z);
        header_bounds.store(3u, hi.x);
        header_bounds.store(4u, hi.y);
        header_bounds.store(5u, hi.z);
    }
}
