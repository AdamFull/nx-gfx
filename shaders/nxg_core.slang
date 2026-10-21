// nx::gfx's shader core: the bindless tables and their accessors, the
// null page a null address reads, and nx_readable. Slang reserves the
// module name `core`.
module nxg_core;

#include "nx/gfx/shaders/core_layout.h"

public static const uint kInvalidIndex = 0xFFFFFFFF;

#if defined(NX_TARGET_METAL)

// Declaration order fixes the byte offsets the C++ side writes slots at, so it
// must not change without updating create_bindless_heap().
public struct BindlessTables {
    public Texture2D<float4> textures[NX_BINDLESS_MAX_TEXTURES];
    public Texture2D<uint4> uint_textures[NX_BINDLESS_MAX_TEXTURES];
    public RWTexture2D<float4> rw_textures[NX_BINDLESS_MAX_RW_TEXTURES];
    public RWTexture2D<uint4> rw_uint_textures[NX_BINDLESS_MAX_RW_TEXTURES];
    public SamplerState samplers[NX_BINDLESS_MAX_SAMPLERS];
    public uint64_t null_address;
};

public ParameterBlock<BindlessTables> g_bindless;

public uint64_t nx_null_address() {
    return g_bindless.null_address;
}

public Texture2D<float4> get_texture(uint index) {
    return g_bindless.textures[index];
}
public Texture2D<uint4> get_uint_texture(uint index) {
    return g_bindless.uint_textures[index];
}
public RWTexture2D<float4> get_rw_texture(uint index) {
    return g_bindless.rw_textures[index];
}
public RWTexture2D<uint4> get_rw_uint_texture(uint index) {
    return g_bindless.rw_uint_textures[index];
}
public SamplerState get_sampler(uint index) {
    return g_bindless.samplers[index];
}

#else

[[vk::binding(NX_BINDLESS_BINDING_SAMPLERS, NX_BINDLESS_SET)]]
SamplerState g_samplers[];

[[vk::binding(NX_BINDLESS_BINDING_TEXTURES, NX_BINDLESS_SET)]]
__DynamicResource<__DynamicResourceKind.General> g_textures[];

[[vk::binding(NX_BINDLESS_BINDING_RW_TEXTURES, NX_BINDLESS_SET)]]
__DynamicResource<__DynamicResourceKind.General> g_rw_textures[];

struct NxGlobals {
    uint64_t null_address;
};

[[vk::binding(NX_BINDLESS_BINDING_GLOBALS, NX_BINDLESS_SET)]]
ConstantBuffer<NxGlobals> g_globals;

public uint64_t nx_null_address() {
    return g_globals.null_address;
}

public Texture2D<float4> get_texture(uint index) {
    return g_textures[NonUniformResourceIndex(index)]
        .asOpaqueDescriptor<Texture2D<float4>>();
}
public Texture2D<uint4> get_uint_texture(uint index) {
    return g_textures[NonUniformResourceIndex(index)]
        .asOpaqueDescriptor<Texture2D<uint4>>();
}
public RWTexture2D<float4> get_rw_texture(uint index) {
    return g_rw_textures[NonUniformResourceIndex(index)]
        .asOpaqueDescriptor<RWTexture2D<float4>>();
}
public RWTexture2D<uint4> get_rw_uint_texture(uint index) {
    return g_rw_textures[NonUniformResourceIndex(index)]
        .asOpaqueDescriptor<RWTexture2D<uint4>>();
}
public SamplerState get_sampler(uint index) {
    return g_samplers[NonUniformResourceIndex(index)];
}

#endif

// The address to read @p offset bytes past @p address from. Mali compilers
// move device loads above the check that guards them, so a null address must
// read the zeroed null page rather than fault.
public uint64_t nx_readable(uint64_t address, uint64_t offset) {
    return address != 0 ? address + offset : nx_null_address();
}

#if !defined(NX_TARGET_METAL)

// Cube, array and 3D textures share the 2D tables: a slot holds a view of its
// texture's own type, and a shader names that type when it reads the slot.
// Metal reports them absent until its 3D track, so they are Vulkan's alone.

public TextureCube<float4> get_texture_cube(uint index) {
    return g_textures[NonUniformResourceIndex(index)]
        .asOpaqueDescriptor<TextureCube<float4>>();
}
public TextureCube<uint4> get_uint_texture_cube(uint index) {
    return g_textures[NonUniformResourceIndex(index)]
        .asOpaqueDescriptor<TextureCube<uint4>>();
}
public Texture2DArray<float4> get_texture_2d_array(uint index) {
    return g_textures[NonUniformResourceIndex(index)]
        .asOpaqueDescriptor<Texture2DArray<float4>>();
}
public Texture2DArray<uint4> get_uint_texture_2d_array(uint index) {
    return g_textures[NonUniformResourceIndex(index)]
        .asOpaqueDescriptor<Texture2DArray<uint4>>();
}
public Texture3D<float4> get_texture_3d(uint index) {
    return g_textures[NonUniformResourceIndex(index)]
        .asOpaqueDescriptor<Texture3D<float4>>();
}
public Texture3D<uint4> get_uint_texture_3d(uint index) {
    return g_textures[NonUniformResourceIndex(index)]
        .asOpaqueDescriptor<Texture3D<uint4>>();
}
public Texture2DMS<float4> get_texture_2d_ms(uint index) {
    return g_textures[NonUniformResourceIndex(index)]
        .asOpaqueDescriptor<Texture2DMS<float4>>();
}
public Texture2DMS<uint4> get_uint_texture_2d_ms(uint index) {
    return g_textures[NonUniformResourceIndex(index)]
        .asOpaqueDescriptor<Texture2DMS<uint4>>();
}
// A cube's storage view is a 2D array of its faces.
public RWTexture2DArray<float4> get_rw_texture_2d_array(uint index) {
    return g_rw_textures[NonUniformResourceIndex(index)]
        .asOpaqueDescriptor<RWTexture2DArray<float4>>();
}
public RWTexture2DArray<uint4> get_rw_uint_texture_2d_array(uint index) {
    return g_rw_textures[NonUniformResourceIndex(index)]
        .asOpaqueDescriptor<RWTexture2DArray<uint4>>();
}
public RWTexture3D<float4> get_rw_texture_3d(uint index) {
    return g_rw_textures[NonUniformResourceIndex(index)]
        .asOpaqueDescriptor<RWTexture3D<float4>>();
}
public RWTexture3D<uint4> get_rw_uint_texture_3d(uint index) {
    return g_rw_textures[NonUniformResourceIndex(index)]
        .asOpaqueDescriptor<RWTexture3D<uint4>>();
}

#endif
