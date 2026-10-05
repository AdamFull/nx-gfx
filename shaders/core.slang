
#pragma once

// Slang reserves `core` for its built-in module, so the engine core is a
// guarded source include rather than a module named after this file.

#include "rendering/rhi/shaders/core_layout.h"

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
};

public ParameterBlock<BindlessTables> g_bindless;

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

float nx_sample_2d(uint texture, uint sampler, float2 uv, float) {
    return get_texture(texture).Sample(get_sampler(sampler), uv).x;
}
float2 nx_sample_2d(uint texture, uint sampler, float2 uv, float2) {
    return get_texture(texture).Sample(get_sampler(sampler), uv).xy;
}
float3 nx_sample_2d(uint texture, uint sampler, float2 uv, float3) {
    return get_texture(texture).Sample(get_sampler(sampler), uv).xyz;
}
float4 nx_sample_2d(uint texture, uint sampler, float2 uv, float4) {
    return get_texture(texture).Sample(get_sampler(sampler), uv);
}
uint nx_sample_2d(uint texture, uint sampler, float2 uv, uint) {
    return get_uint_texture(texture).Sample(get_sampler(sampler), uv).x;
}
uint2 nx_sample_2d(uint texture, uint sampler, float2 uv, uint2) {
    return get_uint_texture(texture).Sample(get_sampler(sampler), uv).xy;
}
uint3 nx_sample_2d(uint texture, uint sampler, float2 uv, uint3) {
    return get_uint_texture(texture).Sample(get_sampler(sampler), uv).xyz;
}
uint4 nx_sample_2d(uint texture, uint sampler, float2 uv, uint4) {
    return get_uint_texture(texture).Sample(get_sampler(sampler), uv);
}

float nx_sample_level_2d(uint t, uint s, float2 uv, float lod, float) {
    return get_texture(t).SampleLevel(get_sampler(s), uv, lod).x;
}
float2 nx_sample_level_2d(uint t, uint s, float2 uv, float lod, float2) {
    return get_texture(t).SampleLevel(get_sampler(s), uv, lod).xy;
}
float3 nx_sample_level_2d(uint t, uint s, float2 uv, float lod, float3) {
    return get_texture(t).SampleLevel(get_sampler(s), uv, lod).xyz;
}
float4 nx_sample_level_2d(uint t, uint s, float2 uv, float lod, float4) {
    return get_texture(t).SampleLevel(get_sampler(s), uv, lod);
}
uint nx_sample_level_2d(uint t, uint s, float2 uv, float lod, uint) {
    return get_uint_texture(t).SampleLevel(get_sampler(s), uv, lod).x;
}
uint2 nx_sample_level_2d(uint t, uint s, float2 uv, float lod, uint2) {
    return get_uint_texture(t).SampleLevel(get_sampler(s), uv, lod).xy;
}
uint3 nx_sample_level_2d(uint t, uint s, float2 uv, float lod, uint3) {
    return get_uint_texture(t).SampleLevel(get_sampler(s), uv, lod).xyz;
}
uint4 nx_sample_level_2d(uint t, uint s, float2 uv, float lod, uint4) {
    return get_uint_texture(t).SampleLevel(get_sampler(s), uv, lod);
}

float nx_sample_grad_2d(uint t, uint s, float2 uv, float2 dx, float2 dy,
                        float) {
    return get_texture(t).SampleGrad(get_sampler(s), uv, dx, dy).x;
}
float2 nx_sample_grad_2d(uint t, uint s, float2 uv, float2 dx, float2 dy,
                         float2) {
    return get_texture(t).SampleGrad(get_sampler(s), uv, dx, dy).xy;
}
float3 nx_sample_grad_2d(uint t, uint s, float2 uv, float2 dx, float2 dy,
                         float3) {
    return get_texture(t).SampleGrad(get_sampler(s), uv, dx, dy).xyz;
}
float4 nx_sample_grad_2d(uint t, uint s, float2 uv, float2 dx, float2 dy,
                         float4) {
    return get_texture(t).SampleGrad(get_sampler(s), uv, dx, dy);
}
uint nx_sample_grad_2d(uint t, uint s, float2 uv, float2 dx, float2 dy,
                       uint) {
    return get_uint_texture(t).SampleGrad(get_sampler(s), uv, dx, dy).x;
}
uint2 nx_sample_grad_2d(uint t, uint s, float2 uv, float2 dx, float2 dy,
                        uint2) {
    return get_uint_texture(t).SampleGrad(get_sampler(s), uv, dx, dy).xy;
}
uint3 nx_sample_grad_2d(uint t, uint s, float2 uv, float2 dx, float2 dy,
                        uint3) {
    return get_uint_texture(t).SampleGrad(get_sampler(s), uv, dx, dy).xyz;
}
uint4 nx_sample_grad_2d(uint t, uint s, float2 uv, float2 dx, float2 dy,
                        uint4) {
    return get_uint_texture(t).SampleGrad(get_sampler(s), uv, dx, dy);
}

float nx_load_2d(uint texture, int3 at, float) {
    return get_texture(texture).Load(at).x;
}
float2 nx_load_2d(uint texture, int3 at, float2) {
    return get_texture(texture).Load(at).xy;
}
float3 nx_load_2d(uint texture, int3 at, float3) {
    return get_texture(texture).Load(at).xyz;
}
float4 nx_load_2d(uint texture, int3 at, float4) {
    return get_texture(texture).Load(at);
}
uint nx_load_2d(uint texture, int3 at, uint) {
    return get_uint_texture(texture).Load(at).x;
}
uint2 nx_load_2d(uint texture, int3 at, uint2) {
    return get_uint_texture(texture).Load(at).xy;
}
uint3 nx_load_2d(uint texture, int3 at, uint3) {
    return get_uint_texture(texture).Load(at).xyz;
}
uint4 nx_load_2d(uint texture, int3 at, uint4) {
    return get_uint_texture(texture).Load(at);
}

public interface INxTexture2DElement {
    static This sample_2d(uint texture, uint sampler, float2 uv);
    static This sample_level_2d(uint texture, uint sampler, float2 uv,
                                float lod);
    static This sample_grad_2d(uint texture, uint sampler, float2 uv,
                               float2 dx, float2 dy);
    static This load_2d(uint texture, int3 at);
    static uint2 dimensions_2d(uint texture);
}

#define NX_DEFINE_TEXTURE_2D_ELEMENT(TYPE, GET_TEXTURE)                     \
    public extension TYPE : INxTexture2DElement {                           \
        static TYPE sample_2d(uint t, uint s, float2 uv) {                   \
            TYPE tag = {};                                                   \
            return nx_sample_2d(t, s, uv, tag);                              \
        }                                                                    \
        static TYPE sample_level_2d(uint t, uint s, float2 uv, float lod) {  \
            TYPE tag = {};                                                   \
            return nx_sample_level_2d(t, s, uv, lod, tag);                   \
        }                                                                    \
        static TYPE sample_grad_2d(uint t, uint s, float2 uv, float2 dx,     \
                                   float2 dy) {                              \
            TYPE tag = {};                                                   \
            return nx_sample_grad_2d(t, s, uv, dx, dy, tag);                 \
        }                                                                    \
        static TYPE load_2d(uint t, int3 at) {                               \
            TYPE tag = {};                                                   \
            return nx_load_2d(t, at, tag);                                   \
        }                                                                    \
        static uint2 dimensions_2d(uint t) {                                 \
            uint2 dimensions;                                                \
            GET_TEXTURE(t).GetDimensions(dimensions.x, dimensions.y);        \
            return dimensions;                                               \
        }                                                                    \
    }

NX_DEFINE_TEXTURE_2D_ELEMENT(float, get_texture)
NX_DEFINE_TEXTURE_2D_ELEMENT(float2, get_texture)
NX_DEFINE_TEXTURE_2D_ELEMENT(float3, get_texture)
NX_DEFINE_TEXTURE_2D_ELEMENT(float4, get_texture)
NX_DEFINE_TEXTURE_2D_ELEMENT(uint, get_uint_texture)
NX_DEFINE_TEXTURE_2D_ELEMENT(uint2, get_uint_texture)
NX_DEFINE_TEXTURE_2D_ELEMENT(uint3, get_uint_texture)
NX_DEFINE_TEXTURE_2D_ELEMENT(uint4, get_uint_texture)

#undef NX_DEFINE_TEXTURE_2D_ELEMENT

float nx_load_rw_2d(uint texture, uint2 at, float) {
    return get_rw_texture(texture)[at].x;
}
float2 nx_load_rw_2d(uint texture, uint2 at, float2) {
    return get_rw_texture(texture)[at].xy;
}
float3 nx_load_rw_2d(uint texture, uint2 at, float3) {
    return get_rw_texture(texture)[at].xyz;
}
float4 nx_load_rw_2d(uint texture, uint2 at, float4) {
    return get_rw_texture(texture)[at];
}
uint nx_load_rw_2d(uint texture, uint2 at, uint) {
    return get_rw_uint_texture(texture)[at].x;
}
uint2 nx_load_rw_2d(uint texture, uint2 at, uint2) {
    return get_rw_uint_texture(texture)[at].xy;
}
uint3 nx_load_rw_2d(uint texture, uint2 at, uint3) {
    return get_rw_uint_texture(texture)[at].xyz;
}
uint4 nx_load_rw_2d(uint texture, uint2 at, uint4) {
    return get_rw_uint_texture(texture)[at];
}

void nx_store_rw_2d(uint texture, uint2 at, float value) {
    get_rw_texture(texture)[at] = float4(value, 0.0, 0.0, 0.0);
}
void nx_store_rw_2d(uint texture, uint2 at, float2 value) {
    get_rw_texture(texture)[at] = float4(value, 0.0, 0.0);
}
void nx_store_rw_2d(uint texture, uint2 at, float3 value) {
    get_rw_texture(texture)[at] = float4(value, 0.0);
}
void nx_store_rw_2d(uint texture, uint2 at, float4 value) {
    get_rw_texture(texture)[at] = value;
}
void nx_store_rw_2d(uint texture, uint2 at, uint value) {
    get_rw_uint_texture(texture)[at] = uint4(value, 0u, 0u, 0u);
}
void nx_store_rw_2d(uint texture, uint2 at, uint2 value) {
    get_rw_uint_texture(texture)[at] = uint4(value, 0u, 0u);
}
void nx_store_rw_2d(uint texture, uint2 at, uint3 value) {
    get_rw_uint_texture(texture)[at] = uint4(value, 0u);
}
void nx_store_rw_2d(uint texture, uint2 at, uint4 value) {
    get_rw_uint_texture(texture)[at] = value;
}

public interface INxRWTexture2DElement {
    static This load_rw_2d(uint texture, uint2 at);
    static void store_rw_2d(uint texture, uint2 at, This value);
    static uint2 dimensions_rw_2d(uint texture);
}

#define NX_DEFINE_RW_TEXTURE_2D_ELEMENT(TYPE, GET_TEXTURE)                   \
    public extension TYPE : INxRWTexture2DElement {                          \
        static TYPE load_rw_2d(uint t, uint2 at) {                           \
            TYPE tag = {};                                                   \
            return nx_load_rw_2d(t, at, tag);                                \
        }                                                                    \
        static void store_rw_2d(uint t, uint2 at, TYPE value) {              \
            nx_store_rw_2d(t, at, value);                                    \
        }                                                                    \
        static uint2 dimensions_rw_2d(uint t) {                              \
            uint2 dimensions;                                                \
            GET_TEXTURE(t).GetDimensions(dimensions.x, dimensions.y);        \
            return dimensions;                                               \
        }                                                                    \
    }

NX_DEFINE_RW_TEXTURE_2D_ELEMENT(float, get_rw_texture)
NX_DEFINE_RW_TEXTURE_2D_ELEMENT(float2, get_rw_texture)
NX_DEFINE_RW_TEXTURE_2D_ELEMENT(float3, get_rw_texture)
NX_DEFINE_RW_TEXTURE_2D_ELEMENT(float4, get_rw_texture)
NX_DEFINE_RW_TEXTURE_2D_ELEMENT(uint, get_rw_uint_texture)
NX_DEFINE_RW_TEXTURE_2D_ELEMENT(uint2, get_rw_uint_texture)
NX_DEFINE_RW_TEXTURE_2D_ELEMENT(uint3, get_rw_uint_texture)
NX_DEFINE_RW_TEXTURE_2D_ELEMENT(uint4, get_rw_uint_texture)

#undef NX_DEFINE_RW_TEXTURE_2D_ELEMENT

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

float4 nx_texel4(float value) { return float4(value, 0.0, 0.0, 0.0); }
float4 nx_texel4(float2 value) { return float4(value, 0.0, 0.0); }
float4 nx_texel4(float3 value) { return float4(value, 0.0); }
float4 nx_texel4(float4 value) { return value; }
uint4 nx_texel4(uint value) { return uint4(value, 0u, 0u, 0u); }
uint4 nx_texel4(uint2 value) { return uint4(value, 0u, 0u); }
uint4 nx_texel4(uint3 value) { return uint4(value, 0u); }
uint4 nx_texel4(uint4 value) { return value; }

public interface INxTextureElement {
    static This sample_cube(uint texture, uint sampler, float3 direction);
    static This sample_level_cube(uint texture, uint sampler, float3 direction,
                                  float lod);
    static uint2 dimensions_cube(uint texture);
    static This sample_2d_array(uint texture, uint sampler, float3 at);
    static This sample_level_2d_array(uint texture, uint sampler, float3 at,
                                      float lod);
    static This load_2d_array(uint texture, int4 at);
    static uint3 dimensions_2d_array(uint texture);
    static This sample_3d(uint texture, uint sampler, float3 at);
    static This sample_level_3d(uint texture, uint sampler, float3 at,
                                float lod);
    static This load_3d(uint texture, int4 at);
    static uint3 dimensions_3d(uint texture);
}

#define NX_DEFINE_TEXTURE_ELEMENT(TYPE, SWIZZLE, CUBE, ARRAY, VOLUME)       \
    public extension TYPE : INxTextureElement {                           \
        static TYPE sample_cube(uint t, uint s, float3 direction) {        \
            return CUBE(t).Sample(get_sampler(s), direction)SWIZZLE;       \
        }                                                                  \
        static TYPE sample_level_cube(uint t, uint s, float3 direction,    \
                                      float lod) {                         \
            return CUBE(t)                                                 \
                .SampleLevel(get_sampler(s), direction, lod)SWIZZLE;       \
        }                                                                  \
        static uint2 dimensions_cube(uint t) {                             \
            uint2 dimensions;                                              \
            CUBE(t).GetDimensions(dimensions.x, dimensions.y);             \
            return dimensions;                                             \
        }                                                                  \
        static TYPE sample_2d_array(uint t, uint s, float3 at) {           \
            return ARRAY(t).Sample(get_sampler(s), at)SWIZZLE;             \
        }                                                                  \
        static TYPE sample_level_2d_array(uint t, uint s, float3 at,       \
                                          float lod) {                     \
            return ARRAY(t).SampleLevel(get_sampler(s), at, lod)SWIZZLE;   \
        }                                                                  \
        static TYPE load_2d_array(uint t, int4 at) {                       \
            return ARRAY(t).Load(at)SWIZZLE;                               \
        }                                                                  \
        static uint3 dimensions_2d_array(uint t) {                         \
            uint3 dimensions;                                              \
            ARRAY(t).GetDimensions(dimensions.x, dimensions.y,             \
                                   dimensions.z);                          \
            return dimensions;                                             \
        }                                                                  \
        static TYPE sample_3d(uint t, uint s, float3 at) {                 \
            return VOLUME(t).Sample(get_sampler(s), at)SWIZZLE;            \
        }                                                                  \
        static TYPE sample_level_3d(uint t, uint s, float3 at, float lod) {\
            return VOLUME(t).SampleLevel(get_sampler(s), at, lod)SWIZZLE;  \
        }                                                                  \
        static TYPE load_3d(uint t, int4 at) {                             \
            return VOLUME(t).Load(at)SWIZZLE;                              \
        }                                                                  \
        static uint3 dimensions_3d(uint t) {                               \
            uint3 dimensions;                                              \
            VOLUME(t).GetDimensions(dimensions.x, dimensions.y,            \
                                    dimensions.z);                         \
            return dimensions;                                             \
        }                                                                  \
    }

NX_DEFINE_TEXTURE_ELEMENT(float, .x, get_texture_cube, get_texture_2d_array,
                          get_texture_3d)
NX_DEFINE_TEXTURE_ELEMENT(float2, .xy, get_texture_cube, get_texture_2d_array,
                          get_texture_3d)
NX_DEFINE_TEXTURE_ELEMENT(float3, .xyz, get_texture_cube, get_texture_2d_array,
                          get_texture_3d)
NX_DEFINE_TEXTURE_ELEMENT(float4, .xyzw, get_texture_cube,
                          get_texture_2d_array, get_texture_3d)
NX_DEFINE_TEXTURE_ELEMENT(uint, .x, get_uint_texture_cube,
                          get_uint_texture_2d_array, get_uint_texture_3d)
NX_DEFINE_TEXTURE_ELEMENT(uint2, .xy, get_uint_texture_cube,
                          get_uint_texture_2d_array, get_uint_texture_3d)
NX_DEFINE_TEXTURE_ELEMENT(uint3, .xyz, get_uint_texture_cube,
                          get_uint_texture_2d_array, get_uint_texture_3d)
NX_DEFINE_TEXTURE_ELEMENT(uint4, .xyzw, get_uint_texture_cube,
                          get_uint_texture_2d_array, get_uint_texture_3d)

#undef NX_DEFINE_TEXTURE_ELEMENT

public interface INxRWTextureElement {
    static This load_rw_2d_array(uint texture, uint3 at);
    static void store_rw_2d_array(uint texture, uint3 at, This value);
    static uint3 dimensions_rw_2d_array(uint texture);
    static This load_rw_3d(uint texture, uint3 at);
    static void store_rw_3d(uint texture, uint3 at, This value);
    static uint3 dimensions_rw_3d(uint texture);
}

#define NX_DEFINE_RW_TEXTURE_ELEMENT(TYPE, SWIZZLE, ARRAY, VOLUME)          \
    public extension TYPE : INxRWTextureElement {                          \
        static TYPE load_rw_2d_array(uint t, uint3 at) {                   \
            return ARRAY(t)[at]SWIZZLE;                                    \
        }                                                                  \
        static void store_rw_2d_array(uint t, uint3 at, TYPE value) {      \
            ARRAY(t)[at] = nx_texel4(value);                               \
        }                                                                  \
        static uint3 dimensions_rw_2d_array(uint t) {                      \
            uint3 dimensions;                                              \
            ARRAY(t).GetDimensions(dimensions.x, dimensions.y,             \
                                   dimensions.z);                          \
            return dimensions;                                             \
        }                                                                  \
        static TYPE load_rw_3d(uint t, uint3 at) {                         \
            return VOLUME(t)[at]SWIZZLE;                                   \
        }                                                                  \
        static void store_rw_3d(uint t, uint3 at, TYPE value) {            \
            VOLUME(t)[at] = nx_texel4(value);                              \
        }                                                                  \
        static uint3 dimensions_rw_3d(uint t) {                            \
            uint3 dimensions;                                              \
            VOLUME(t).GetDimensions(dimensions.x, dimensions.y,            \
                                    dimensions.z);                         \
            return dimensions;                                             \
        }                                                                  \
    }

NX_DEFINE_RW_TEXTURE_ELEMENT(float, .x, get_rw_texture_2d_array,
                             get_rw_texture_3d)
NX_DEFINE_RW_TEXTURE_ELEMENT(float2, .xy, get_rw_texture_2d_array,
                             get_rw_texture_3d)
NX_DEFINE_RW_TEXTURE_ELEMENT(float3, .xyz, get_rw_texture_2d_array,
                             get_rw_texture_3d)
NX_DEFINE_RW_TEXTURE_ELEMENT(float4, .xyzw, get_rw_texture_2d_array,
                             get_rw_texture_3d)
NX_DEFINE_RW_TEXTURE_ELEMENT(uint, .x, get_rw_uint_texture_2d_array,
                             get_rw_uint_texture_3d)
NX_DEFINE_RW_TEXTURE_ELEMENT(uint2, .xy, get_rw_uint_texture_2d_array,
                             get_rw_uint_texture_3d)
NX_DEFINE_RW_TEXTURE_ELEMENT(uint3, .xyz, get_rw_uint_texture_2d_array,
                             get_rw_uint_texture_3d)
NX_DEFINE_RW_TEXTURE_ELEMENT(uint4, .xyzw, get_rw_uint_texture_2d_array,
                             get_rw_uint_texture_3d)

#undef NX_DEFINE_RW_TEXTURE_ELEMENT

#endif

#include "nx_texture.slang"
#include "nx_ptr.slang"
