
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

public struct NxSampler {
    public uint index;

    [ForceInline]
    public static NxSampler from_index(uint index) {
        NxSampler result;
        result.index = index;
        return result;
    }

    [ForceInline]
    public bool is_valid() { return index != kInvalidIndex; }

    [ForceInline]
    public SamplerState get() { return get_sampler(index); }
}

/// Compact sampled 2D texture. Its single word retains the existing ABI:
/// texture index in the upper 16 bits, sampler and flags in the lower 16.
public struct NxTexture2D<T>
    where T : INxTexture2DElement {
    public uint value;

    [ForceInline]
    public static NxTexture2D<T> from_packed(uint packed) {
        NxTexture2D<T> result;
        result.value = packed;
        return result;
    }

    [ForceInline]
    public static NxTexture2D<T> from_indices(uint texture, uint sampler) {
        return from_packed((texture << 16) |
                           (sampler & NX_TEXTURE_SAMPLER_MASK));
    }

    [ForceInline]
    public uint texture_index() { return value >> 16; }

    [ForceInline]
    public uint sampler_index() { return value & NX_TEXTURE_SAMPLER_MASK; }

    [ForceInline]
    public NxSampler sampler() { return NxSampler.from_index(sampler_index()); }

    [ForceInline]
    public bool is_linear() { return (value & NX_TEXTURE_LINEAR) != 0u; }

    [ForceInline]
    public bool is_premultiplied() {
        return (value & NX_TEXTURE_PREMULTIPLIED) != 0u;
    }

    [ForceInline]
    public bool is_valid() { return texture_index() != NX_TEXTURE_NONE; }

    [ForceInline]
    public uint2 dimensions() { return T.dimensions_2d(texture_index()); }

    [ForceInline]
    public T sample(float2 uv) {
        return T.sample_2d(texture_index(), sampler_index(), uv);
    }

    [ForceInline]
    public T sample_level(float2 uv, float lod) {
        return T.sample_level_2d(texture_index(), sampler_index(), uv, lod);
    }

    [ForceInline]
    public T sample_grad(float2 uv, float2 dx, float2 dy) {
        return T.sample_grad_2d(texture_index(), sampler_index(), uv, dx, dy);
    }

    [ForceInline]
    public T load(int3 at) {
        return T.load_2d(texture_index(), at);
    }
}

/// Escape hatches for algorithms whose signatures require a native resource
/// object (for example the reference FXAA implementation). Ordinary shader
/// code should use the typed methods above.
public extension NxTexture2D<float4> {
    [ForceInline]
    public Texture2D<float4> native_texture() {
        return get_texture(texture_index());
    }
}

public extension NxTexture2D<uint4> {
    [ForceInline]
    public Texture2D<uint4> native_texture() {
        return get_uint_texture(texture_index());
    }
}

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

public struct NxRWTexture2D<T>
    where T : INxRWTexture2DElement {
    public uint index;

    [ForceInline]
    public static NxRWTexture2D<T> from_index(uint index) {
        NxRWTexture2D<T> result;
        result.index = index;
        return result;
    }

    [ForceInline]
    public bool is_valid() { return index != kInvalidIndex; }

    [ForceInline]
    public uint2 dimensions() {
        return T.dimensions_rw_2d(index);
    }

    [ForceInline]
    public T load(uint2 at) {
        return T.load_rw_2d(index, at);
    }

    [ForceInline]
    public void store(uint2 at, T value) {
        T.store_rw_2d(index, at, value);
    }

    public __subscript(uint2 at) -> T {
        get { return load(at); }
        [nonmutating] set { store(at, newValue); }
    }
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

/// Compact sampled cube texture, in NxTexture2D's one-word ABI.
public struct NxTextureCube<T>
    where T : INxTextureElement {
    public uint value;

    [ForceInline]
    public static NxTextureCube<T> from_packed(uint packed) {
        NxTextureCube<T> result;
        result.value = packed;
        return result;
    }

    [ForceInline]
    public uint texture_index() { return value >> 16; }

    [ForceInline]
    public uint sampler_index() { return value & NX_TEXTURE_SAMPLER_MASK; }

    [ForceInline]
    public bool is_valid() { return texture_index() != NX_TEXTURE_NONE; }

    /// One face's.
    [ForceInline]
    public uint2 dimensions() { return T.dimensions_cube(texture_index()); }

    [ForceInline]
    public T sample(float3 direction) {
        return T.sample_cube(texture_index(), sampler_index(), direction);
    }

    [ForceInline]
    public T sample_level(float3 direction, float lod) {
        return T.sample_level_cube(texture_index(), sampler_index(), direction,
                                   lod);
    }
}

/// Compact sampled 2D array texture, in NxTexture2D's one-word ABI.
public struct NxTexture2DArray<T>
    where T : INxTextureElement {
    public uint value;

    [ForceInline]
    public static NxTexture2DArray<T> from_packed(uint packed) {
        NxTexture2DArray<T> result;
        result.value = packed;
        return result;
    }

    [ForceInline]
    public uint texture_index() { return value >> 16; }

    [ForceInline]
    public uint sampler_index() { return value & NX_TEXTURE_SAMPLER_MASK; }

    [ForceInline]
    public bool is_valid() { return texture_index() != NX_TEXTURE_NONE; }

    /// Width, height and layers.
    [ForceInline]
    public uint3 dimensions() {
        return T.dimensions_2d_array(texture_index());
    }

    [ForceInline]
    public T sample(float2 uv, float layer) {
        return T.sample_2d_array(texture_index(), sampler_index(),
                                 float3(uv, layer));
    }

    [ForceInline]
    public T sample_level(float2 uv, float layer, float lod) {
        return T.sample_level_2d_array(texture_index(), sampler_index(),
                                       float3(uv, layer), lod);
    }

    [ForceInline]
    public T load(int2 at, uint layer, uint mip) {
        return T.load_2d_array(texture_index(), int4(at, int(layer), int(mip)));
    }
}

/// Compact sampled 3D texture, in NxTexture2D's one-word ABI.
public struct NxTexture3D<T>
    where T : INxTextureElement {
    public uint value;

    [ForceInline]
    public static NxTexture3D<T> from_packed(uint packed) {
        NxTexture3D<T> result;
        result.value = packed;
        return result;
    }

    [ForceInline]
    public uint texture_index() { return value >> 16; }

    [ForceInline]
    public uint sampler_index() { return value & NX_TEXTURE_SAMPLER_MASK; }

    [ForceInline]
    public bool is_valid() { return texture_index() != NX_TEXTURE_NONE; }

    [ForceInline]
    public uint3 dimensions() { return T.dimensions_3d(texture_index()); }

    [ForceInline]
    public T sample(float3 at) {
        return T.sample_3d(texture_index(), sampler_index(), at);
    }

    [ForceInline]
    public T sample_level(float3 at, float lod) {
        return T.sample_level_3d(texture_index(), sampler_index(), at, lod);
    }

    [ForceInline]
    public T load(int3 at, uint mip) {
        return T.load_3d(texture_index(), int4(at, int(mip)));
    }
}

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

/// A storage 2D array, or a cube's faces as one: x, y and the layer.
public struct NxRWTexture2DArray<T>
    where T : INxRWTextureElement {
    public uint index;

    [ForceInline]
    public static NxRWTexture2DArray<T> from_index(uint index) {
        NxRWTexture2DArray<T> result;
        result.index = index;
        return result;
    }

    [ForceInline]
    public bool is_valid() { return index != kInvalidIndex; }

    [ForceInline]
    public uint3 dimensions() { return T.dimensions_rw_2d_array(index); }

    [ForceInline]
    public T load(uint3 at) { return T.load_rw_2d_array(index, at); }

    [ForceInline]
    public void store(uint3 at, T value) {
        T.store_rw_2d_array(index, at, value);
    }

    public __subscript(uint3 at) -> T {
        get { return load(at); }
        [nonmutating] set { store(at, newValue); }
    }
}

public struct NxRWTexture3D<T>
    where T : INxRWTextureElement {
    public uint index;

    [ForceInline]
    public static NxRWTexture3D<T> from_index(uint index) {
        NxRWTexture3D<T> result;
        result.index = index;
        return result;
    }

    [ForceInline]
    public bool is_valid() { return index != kInvalidIndex; }

    [ForceInline]
    public uint3 dimensions() { return T.dimensions_rw_3d(index); }

    [ForceInline]
    public T load(uint3 at) { return T.load_rw_3d(index, at); }

    [ForceInline]
    public void store(uint3 at, T value) { T.store_rw_3d(index, at, value); }

    public __subscript(uint3 at) -> T {
        get { return load(at); }
        [nonmutating] set { store(at, newValue); }
    }
}

#endif

public uint64_t stride_of<T>()
{
    uint64_t size = uint64_t(sizeof(T));
    uint64_t align = uint64_t(alignof(T));
    return (size + align - 1) & ~(align - 1);
}

F load_raw<F>(uint64_t address)
{
    // Mali/Intel Vulkan crash pipeline compilation on an access chain that
    // yields a physical pointer-to-struct; copying as uint words never forms
    // one. Do not "simplify" back to *(F*).
    static_assert(sizeof(F) % sizeof(uint) == 0, "type must be 32-bit-addressable");
    static const uint word_count = sizeof(F) / sizeof(uint);
    static const uint vector_count = word_count / 4u;
    static const uint vector_words = vector_count * 4u;
    uint words[word_count];
    uint4* vectors = (uint4*)address;
    [ForceUnroll] for (uint v = 0; v < vector_count; ++v) {
        const uint4 word = vectors[v];
        words[v * 4u + 0u] = word.x;
        words[v * 4u + 1u] = word.y;
        words[v * 4u + 2u] = word.z;
        words[v * 4u + 3u] = word.w;
    }
    uint* scalars = (uint*)address;
    [ForceUnroll] for (uint w = vector_words; w < word_count; ++w)
        words[w] = scalars[w];
    return reinterpret<F>(words);
}

void store_raw<F>(uint64_t address, in F value)
{
    static_assert(sizeof(F) % sizeof(uint) == 0, "type must be 32-bit-addressable");
    static const uint word_count = sizeof(F) / sizeof(uint);
    static const uint vector_count = word_count / 4u;
    static const uint vector_words = vector_count * 4u;
    uint words[word_count] = reinterpret<uint[word_count]>(value);
    uint4* vectors = (uint4*)address;
    [ForceUnroll] for (uint v = 0; v < vector_count; ++v) {
        const uint w = v * 4u;
        vectors[v] = uint4(words[w], words[w + 1u], words[w + 2u],
                           words[w + 3u]);
    }
    uint* scalars = (uint*)address;
    [ForceUnroll] for (uint w = vector_words; w < word_count; ++w)
        scalars[w] = words[w];
    return;
}

public struct NxPtr<T>
{
    public uint64_t value;

    public bool is_valid() { return value != 0; }

    uint64_t element_address(uint index) { return value + uint64_t(index) * stride_of<T>(); }
    uint64_t field_address(uint index, uint field_offset) { return element_address(index) + uint64_t(field_offset); }

    public T load(uint index) { return load_raw<T>(element_address(index)); }
    public void store(uint index, in T newValue) { store_raw<T>(element_address(index), newValue); }

    public __subscript(uint index) -> T
    {
        get { return load(index); }
        // A store mutates external memory, not the address; nonmutating lets
        // addr[i] = x work on a read-only (push-constant) address.
        [nonmutating] set { store(index, newValue); }
    }

    public F get_field<F>(uint index, uint field_offset) { return load_raw<F>(field_address(index, field_offset)); }
    public void set_field<F>(uint index, uint field_offset, in F fieldValue) { store_raw<F>(field_address(index, field_offset), fieldValue); }
}

public NxPtr<T> operator+<T>(NxPtr<T> a, uint n)
{
    NxPtr<T> r = a;
    r.value = a.value + uint64_t(n) * stride_of<T>();
    return r;
}
public NxPtr<T> operator+<T>(uint n, NxPtr<T> a) { return a + n; }
public NxPtr<T> operator-<T>(NxPtr<T> a, uint n)
{
    NxPtr<T> r = a;
    r.value = a.value - uint64_t(n) * stride_of<T>();
    return r;
}
public bool operator==<T>(NxPtr<T> a, NxPtr<T> b) { return a.value == b.value; }
public bool operator!=<T>(NxPtr<T> a, NxPtr<T> b) { return a.value != b.value; }
