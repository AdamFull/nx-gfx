// Traces a buffer of rays through a scene, a hardware one or, where
// NX_RAY_SOFTWARE is defined (ray_scene_software.slang), the library's own:
// one source for both, as a consumer writes it. A ray is two float4s (origin
// and tmin, direction and tmax); a hit is two (t, barycentrics and whether it
// faced the ray; primitive, instance, instance index and geometry, a uint's
// bits each), t -1 for a miss. A triangle not made opaque counts where its
// first barycentric is at least `accept_from`; instances whose mask shares no
// bit with `mask` are skipped.
import nxg_ray;

#if defined(NX_RAY_SOFTWARE)
typealias Scene = NxSoftwareScene;
#else
typealias Scene = NxScene;
#endif

struct Push {
    uint64_t scene;
    float4 *rays;
    float4 *hits;
    uint count;
    float accept_from;
    uint mask;
    uint pad;
};

[[vk::push_constant]]
Push pc;

struct Record : IRayShade {
    float4 first;
    float4 second;
    float accept_from;

    [mutating]
    bool any_hit(NxRayHit candidate) {
        return candidate.barycentrics.x >= accept_from;
    }
    [mutating]
    void closest_hit(NxRayHit hit) {
        first = float4(hit.t, hit.barycentrics, hit.front_face ? 1.0 : 0.0);
        second = float4(asfloat(hit.primitive), asfloat(hit.instance),
                        asfloat(hit.instance_index), asfloat(hit.geometry));
    }
    [mutating]
    void miss() {
        first = float4(-1.0, 0.0, 0.0, 0.0);
        second = float4(0.0);
    }
};

void trace<let Flags : uint>(uint index) {
    if (index >= pc.count)
        return;
    const float4 a = pc.rays[index * 2u];
    const float4 b = pc.rays[index * 2u + 1u];
    NxRay ray;
    ray.origin = a.xyz;
    ray.tmin = a.w;
    ray.direction = b.xyz;
    ray.tmax = b.w;
    Record record;
    record.first = float4(0.0);
    record.second = float4(0.0);
    record.accept_from = pc.accept_from;
    nx_trace<Flags>(Scene.from_address(pc.scene), ray, pc.mask, record);
    pc.hits[index * 2u] = record.first;
    pc.hits[index * 2u + 1u] = record.second;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void trace_main(uint3 tid: SV_DispatchThreadID) {
    trace<NX_RAY_NONE>(tid.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void trace_cull_back(uint3 tid: SV_DispatchThreadID) {
    trace<NX_RAY_CULL_BACK>(tid.x);
}
