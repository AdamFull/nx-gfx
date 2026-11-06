// Traces two rays through nxg_ray: thread 0 at the triangle, thread 1 past
// it. Each writes its hit's distance, barycentrics and instance custom index,
// or -1 for a miss. A triangle not made opaque counts only where `accept`.
import nxg_ray;

struct Push {
    uint64_t scene;
    float4 *out;
    uint accept;
};

[[vk::push_constant]]
Push pc;

struct Record : IRayShade {
    float4 result;
    bool accept;

    [mutating]
    bool any_hit(NxRayHit candidate) {
        return accept;
    }
    [mutating]
    void closest_hit(NxRayHit hit) {
        result = float4(hit.t, hit.barycentrics, float(hit.instance));
    }
    [mutating]
    void miss() {
        result = float4(-1.0, 0.0, 0.0, 0.0);
    }
};

[shader("compute")]
[numthreads(2, 1, 1)]
void trace_main(uint3 tid: SV_DispatchThreadID) {
    NxRay ray;
    ray.origin = tid.x == 0u ? float3(0.25, 0.25, -1.0) : float3(2.0, 2.0, -1.0);
    ray.tmin = 0.0;
    ray.direction = float3(0.0, 0.0, 1.0);
    ray.tmax = 100.0;
    Record record;
    record.result = float4(0.0);
    record.accept = pc.accept != 0u;
    nx_trace<NX_RAY_NONE>(NxScene.from_address(pc.scene), ray, 0xFFu, record);
    pc.out[tid.x] = record.result;
}
