// The refit's two passes over a block (nxg_bvh), as a consumer writes them:
// one thread a triangle record, then one a node of a level, `first` to
// `first + count`.
import nxg_bvh;
import nxg_ptr;

struct Push {
    uint64_t block;
    uint64_t geometries;
    uint first;
    uint count;
};

[[vk::push_constant]]
Push pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void refit_triangles(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= pc.count)
        return;
    NxPtr<NxBvhRefitGeometry> geometries;
    geometries.value = pc.geometries;
    nx_bvh_refit_triangle(pc.block, geometries, pc.first + tid.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void refit_nodes(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= pc.count)
        return;
    nx_bvh_refit_node(pc.block, pc.first + tid.x);
}
