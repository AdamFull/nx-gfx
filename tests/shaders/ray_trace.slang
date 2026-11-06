// Traces two rays at a top level made from its address: thread 0 at the
// triangle, thread 1 past it. Each writes its hit's distance, barycentrics
// and instance custom index, or -1 for a miss.

struct Push {
    uint64_t scene;
    float4 *out;
};

[[vk::push_constant]]
Push pc;

[shader("compute")]
[numthreads(2, 1, 1)]
void trace_main(uint3 tid: SV_DispatchThreadID) {
    RaytracingAccelerationStructure scene =
        RaytracingAccelerationStructure(pc.scene);
    RayDesc ray;
    ray.Origin = tid.x == 0u ? float3(0.25, 0.25, -1.0) : float3(2.0, 2.0, -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 100.0;
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFFu, ray);
    while (query.Proceed()) {
    }
    float4 result = float4(-1.0, 0.0, 0.0, 0.0);
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        const float2 bary = query.CommittedTriangleBarycentrics();
        result = float4(query.CommittedRayT(), bary.x, bary.y,
                        float(query.CommittedInstanceID()));
    }
    pc.out[tid.x] = result;
}
