// Ray tracing for shaders, the Slang half of nx/gfx/ray.h: a scene made from
// a top level's address, a ray, what it hits, and nx_trace, which walks the
// scene and calls back what a caller's IRayShade does with each hit. A trace
// is inline (a ray query) in any stage.
//
// Metal reports no ray queries until its 3D track, and Slang makes no
// acceleration structure from an address there; on Metal every trace misses,
// so a module that imports this still compiles.
module nxg_ray;

__exported import nxg_ray_layout;

/// A top level, by its address (acceleration_structure_address).
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
