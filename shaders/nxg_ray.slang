// Ray tracing for shaders, the Slang half of nx/gfx/ray.h and nx/gfx/bvh.h: a
// scene, a ray, what it hits, and nx_trace, which walks the scene and calls
// back what a caller's IRayShade does with each hit. NxScene is a hardware top
// level, traced inline (a ray query); NxSoftwareScene is the library's own
// (nx/gfx/bvh.h), walked by this module where a device has no ray queries. A
// consumer writes its trace once over a scene type its own source picks.
//
// Metal reports no ray queries until its 3D track, and Slang makes no
// acceleration structure from an address there; on Metal a hardware trace
// misses, so a module that imports this still compiles.
module nxg_ray;

__exported import nxg_ray_layout;
import nxg_ptr;

/// A hardware top level, by its address (acceleration_structure_address).
public struct NxScene {
    public uint64_t address;

    public static NxScene from_address(uint64_t address) {
        NxScene scene;
        scene.address = address;
        return scene;
    }
    public bool is_valid() {
        return address != 0;
    }
};

/// Points `tmin` to `tmax` along `direction` from `origin` are tested, in the
/// top level's space.
public struct NxRay {
    public float3 origin;
    public float tmin;
    public float3 direction;
    public float tmax;
};

/// A triangle the ray met: how far along it, the barycentrics of its second
/// and third corners, the triangle within its geometry, the instance's custom
/// index (NxRayInstance) and place in the top level, the geometry within the
/// bottom level, and whether it faced the ray.
public struct NxRayHit {
    public float t;
    public float2 barycentrics;
    public uint primitive;
    public uint instance;
    public uint instance_index;
    public uint geometry;
    public bool front_face;
};

/// What a trace does with what it meets. any_hit sees each triangle of a
/// geometry not made opaque, in no set order, and answers whether it counts
/// (a cut-out texel does not); then closest_hit sees the nearest that
/// counted, or miss is called.
public interface IRayShade {
    [mutating]
    public bool any_hit(NxRayHit candidate);
    [mutating]
    public void closest_hit(NxRayHit hit);
    [mutating]
    public void miss();
};

#if defined(NX_TARGET_METAL)

public bool nx_trace<let Flags : uint, S : IRayShade>(NxScene scene, NxRay ray,
                                                     uint mask, inout S shade) {
    shade.miss();
    return false;
}

#else

/// Traces @p ray through @p scene, skipping instances whose mask shares no
/// bit with @p mask, with the NX_RAY_ flags `Flags`; true where it hit.
public bool nx_trace<let Flags : uint, S : IRayShade>(NxScene scene, NxRay ray,
                                                     uint mask, inout S shade) {
    RayDesc desc;
    desc.Origin = ray.origin;
    desc.TMin = ray.tmin;
    desc.Direction = ray.direction;
    desc.TMax = ray.tmax;
    RayQuery<Flags> query;
    query.TraceRayInline(RaytracingAccelerationStructure(scene.address), Flags,
                         mask, desc);
    while (query.Proceed()) {
        if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE)
            continue;
        NxRayHit candidate;
        candidate.t = query.CandidateTriangleRayT();
        candidate.barycentrics = query.CandidateTriangleBarycentrics();
        candidate.primitive = query.CandidatePrimitiveIndex();
        candidate.instance = query.CandidateInstanceID();
        candidate.instance_index = query.CandidateInstanceIndex();
        candidate.geometry = query.CandidateGeometryIndex();
        candidate.front_face = query.CandidateTriangleFrontFace();
        if (shade.any_hit(candidate))
            query.CommitNonOpaqueTriangleHit();
    }
    if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
        shade.miss();
        return false;
    }
    NxRayHit hit;
    hit.t = query.CommittedRayT();
    hit.barycentrics = query.CommittedTriangleBarycentrics();
    hit.primitive = query.CommittedPrimitiveIndex();
    hit.instance = query.CommittedInstanceID();
    hit.instance_index = query.CommittedInstanceIndex();
    hit.geometry = query.CommittedGeometryIndex();
    hit.front_face = query.CommittedTriangleFrontFace();
    shade.closest_hit(hit);
    return true;
}

#endif

/// A software top level (nx/gfx/bvh.h), by its block's address.
public struct NxSoftwareScene {
    public uint64_t address;

    public static NxSoftwareScene from_address(uint64_t address) {
        NxSoftwareScene scene;
        scene.address = address;
        return scene;
    }
    public bool is_valid() {
        return address != 0;
    }
};

/// Where a block's nodes and primitives lie.
struct NxBvhBlock {
    NxPtr<NxBvhNode> nodes;
    uint64_t primitives;
};

NxBvhBlock nx_bvh_block(uint64_t address) {
    NxPtr<NxBvhHeader> header;
    header.value = address;
    const NxBvhHeader read = header.load(0u);
    NxBvhBlock block;
    block.nodes.value = address + uint64_t(read.nodes_offset);
    block.primitives = address + uint64_t(read.primitives_offset);
    return block;
}

/// A ray in one level's space: its direction's reciprocal for the box tests,
/// with no component zero, and Woop's shear for the triangle tests, made once.
struct NxRaySpace {
    float3 origin;
    float3 direction;
    float3 inverse;
    uint kx;
    uint ky;
    uint kz;
    float sx;
    float sy;
    float sz;
};

NxRaySpace nx_ray_space(float3 origin, float3 direction) {
    NxRaySpace space;
    space.origin = origin;
    space.direction = direction;
    const float3 tiny = select(direction < 0.0, float3(-1e-20), float3(1e-20));
    space.inverse = 1.0 / select(abs(direction) < 1e-20, tiny, direction);
    const float3 size = abs(direction);
    uint kz = size.y > size.x ? 1u : 0u;
    if (size.z > size[kz])
        kz = 2u;
    uint kx = (kz + 1u) % 3u;
    uint ky = (kx + 1u) % 3u;
    if (direction[kz] < 0.0) {
        const uint swap = kx;
        kx = ky;
        ky = swap;
    }
    space.kx = kx;
    space.ky = ky;
    space.kz = kz;
    space.sx = direction[kx] / direction[kz];
    space.sy = direction[ky] / direction[kz];
    space.sz = 1.0 / direction[kz];
    return space;
}

/// Woop, Benthin and Wald's watertight test, in float alone: the double
/// fallback for an edge function of exactly zero is left out, as GPUs may have
/// no double.
bool nx_bvh_triangle(NxBvhTriangle triangle, NxRaySpace ray, float tmin,
                     float tmax, out float t, out float2 barycentrics,
                     out bool front) {
    t = 0.0;
    barycentrics = float2(0.0);
    front = false;
    const float3 a = triangle.v0.xyz - ray.origin;
    const float3 b = triangle.v1.xyz - ray.origin;
    const float3 c = triangle.v2.xyz - ray.origin;
    const float ax = a[ray.kx] - ray.sx * a[ray.kz];
    const float ay = a[ray.ky] - ray.sy * a[ray.kz];
    const float bx = b[ray.kx] - ray.sx * b[ray.kz];
    const float by = b[ray.ky] - ray.sy * b[ray.kz];
    const float cx = c[ray.kx] - ray.sx * c[ray.kz];
    const float cy = c[ray.ky] - ray.sy * c[ray.kz];
    const float u = cx * by - cy * bx;
    const float v = ax * cy - ay * cx;
    const float w = bx * ay - by * ax;
    if ((u < 0.0 || v < 0.0 || w < 0.0) && (u > 0.0 || v > 0.0 || w > 0.0))
        return false;
    const float det = u + v + w;
    if (det == 0.0)
        return false;
    const float az = ray.sz * a[ray.kz];
    const float bz = ray.sz * b[ray.kz];
    const float cz = ray.sz * c[ray.kz];
    t = (u * az + v * bz + w * cz) / det;
    if (!(t > tmin) || !(t < tmax))
        return false;
    barycentrics = float2(v, w) / det;
    // The edge functions sum positive where the corners wind counter-clockwise
    // seen from the ray, which is what the hardware calls front.
    front = det > 0.0;
    return true;
}

/// Traces @p ray through a software top level, as the hardware nx_trace does:
/// instances whose mask shares no bit with @p mask skipped, the NX_RAY_ flags
/// `Flags` and each instance's NX_RAY_INSTANCE_ flags honoured. One stack
/// serves both levels: entering an instance pushes NX_BVH_NONE, which brings
/// the walk back to the top level's space when it is popped. Children are
/// pushed farthest first, so the nearest is walked next; a stack that is full
/// drops what does not fit.
public bool nx_trace<let Flags : uint, S : IRayShade>(NxSoftwareScene scene,
                                                     NxRay ray, uint mask,
                                                     inout S shade) {
    // Rounding in the slab test can lose a box a ray grazes; widen its far
    // distance by a few ulps (Ize).
    const float WIDEN = 1.0 + 6.0 * 1.1920929e-7;
    NxRayHit hit;
    hit.t = 0.0;
    hit.barycentrics = float2(0.0);
    hit.primitive = 0u;
    hit.instance = 0u;
    hit.instance_index = 0u;
    hit.geometry = 0u;
    hit.front_face = false;
    if (!scene.is_valid()) {
        shade.miss();
        return false;
    }

    const NxBvhBlock top = nx_bvh_block(scene.address);
    const NxRaySpace world = nx_ray_space(ray.origin, ray.direction);
    NxBvhBlock block = top;
    NxRaySpace space = world;
    bool in_bottom = false;
    uint instance_custom = 0u;
    uint instance_index = 0u;
    uint instance_flags = 0u;
    float best = ray.tmax;
    bool found = false;
    bool done = false;

    uint stack[NX_BVH_STACK];
    uint depth = 0u;
    stack[depth++] = 0u;
    while (depth > 0u && !done) {
        const uint entry = stack[--depth];
        if (entry == NX_BVH_NONE) {
            block = top;
            space = world;
            in_bottom = false;
            continue;
        }
        if (nx_bvh_is_leaf(entry)) {
            const uint first = nx_bvh_leaf_first(entry);
            if (!in_bottom) {
                NxPtr<NxBvhInstance> instances;
                instances.value = block.primitives;
                const NxBvhInstance instance = instances.load(first);
                if (((instance.custom_index_and_mask >> 24u) & mask) == 0u ||
                    depth + 2u > NX_BVH_STACK)
                    continue;
                instance_custom = instance.custom_index_and_mask & 0x00FFFFFFu;
                instance_index = instance.flags & NX_BVH_INSTANCE_INDEX_MASK;
                instance_flags = instance.flags >> 24u;
                // Into the instance's space, the direction left unnormalized
                // so a distance along it is the same in both.
                const float4 origin = float4(ray.origin, 1.0);
                const float4 direction = float4(ray.direction, 0.0);
                space = nx_ray_space(
                    float3(dot(instance.row0, origin), dot(instance.row1, origin),
                           dot(instance.row2, origin)),
                    float3(dot(instance.row0, direction),
                           dot(instance.row1, direction),
                           dot(instance.row2, direction)));
                block = nx_bvh_block(instance.bottom_level);
                in_bottom = true;
                stack[depth++] = NX_BVH_NONE;
                stack[depth++] = 0u;
                continue;
            }
            NxPtr<NxBvhTriangle> triangles;
            triangles.value = block.primitives;
            const uint end = first + nx_bvh_leaf_count(entry);
            for (uint i = first; i < end; ++i) {
                const NxBvhTriangle triangle = triangles.load(i);
                float t;
                float2 barycentrics;
                bool front;
                if (!nx_bvh_triangle(triangle, space, ray.tmin, best, t,
                                     barycentrics, front))
                    continue;
                if ((instance_flags & NX_RAY_INSTANCE_FLIP_FACING) != 0u)
                    front = !front;
                if ((instance_flags & NX_RAY_INSTANCE_CULL_DISABLE) == 0u &&
                    (((Flags & NX_RAY_CULL_BACK) != 0u && !front) ||
                     ((Flags & NX_RAY_CULL_FRONT) != 0u && front)))
                    continue;
                NxRayHit candidate;
                candidate.t = t;
                candidate.barycentrics = barycentrics;
                candidate.primitive = asuint(triangle.v0.w);
                candidate.instance = instance_custom;
                candidate.instance_index = instance_index;
                candidate.geometry = asuint(triangle.v1.w);
                candidate.front_face = front;
                bool opaque = (asuint(triangle.v2.w) & NX_BVH_TRIANGLE_OPAQUE) != 0u;
                if ((instance_flags & NX_RAY_INSTANCE_FORCE_OPAQUE) != 0u)
                    opaque = true;
                if ((instance_flags & NX_RAY_INSTANCE_FORCE_NO_OPAQUE) != 0u)
                    opaque = false;
                if ((Flags & NX_RAY_OPAQUE) != 0u)
                    opaque = true;
                if (!opaque && !shade.any_hit(candidate))
                    continue;
                best = t;
                hit = candidate;
                found = true;
                if ((Flags & NX_RAY_FIRST_HIT) != 0u) {
                    done = true;
                    break;
                }
            }
            continue;
        }

        const NxBvhNode node = block.nodes.load(entry);
        const uint count = nx_bvh_child_count(node.meta);
        const float3 origin = float3(node.origin_x, node.origin_y, node.origin_z);
        const float3 step = float3(nx_bvh_step(node.meta, 0u),
                                   nx_bvh_step(node.meta, 1u),
                                   nx_bvh_step(node.meta, 2u));
        const uint words[4] = {node.children.x, node.children.y,
                               node.children.z, node.children.w};
        float near_of[4];
        uint word_of[4];
        uint hits = 0u;
        for (uint c = 0u; c < count; ++c) {
            const float3 low =
                origin + float3(float(nx_bvh_byte(node.lo_x, c)),
                                float(nx_bvh_byte(node.lo_y, c)),
                                float(nx_bvh_byte(node.lo_z, c))) * step;
            const float3 high =
                origin + float3(float(nx_bvh_byte(node.hi_x, c)),
                                float(nx_bvh_byte(node.hi_y, c)),
                                float(nx_bvh_byte(node.hi_z, c))) * step;
            const float3 t0 = (low - space.origin) * space.inverse;
            const float3 t1 = (high - space.origin) * space.inverse;
            const float3 nearest = min(t0, t1);
            const float3 farthest = max(t0, t1);
            const float near_t =
                max(max(nearest.x, nearest.y), max(nearest.z, ray.tmin));
            const float far_t =
                min(min(farthest.x, farthest.y), farthest.z) * WIDEN;
            if (near_t > min(far_t, best))
                continue;
            // Kept nearest first.
            uint at = hits;
            while (at > 0u && near_of[at - 1u] > near_t) {
                near_of[at] = near_of[at - 1u];
                word_of[at] = word_of[at - 1u];
                --at;
            }
            near_of[at] = near_t;
            word_of[at] = words[c];
            ++hits;
        }
        for (uint h = hits; h > 0u; --h)
            if (depth < NX_BVH_STACK)
                stack[depth++] = word_of[h - 1u];
    }
    if (!found) {
        shade.miss();
        return false;
    }
    shade.closest_hit(hit);
    return true;
}
