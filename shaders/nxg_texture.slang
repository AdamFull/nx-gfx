// The bindless texture handles shaders sample, load and store through,
// the Slang half of the pair whose C++ half is nx/gfx/shaders/texture.hpp,
// with the texel types they read; the tables they reach are nxg_core's.
module nxg_texture;

__exported import nxg_core;
__exported import nxg_texture_layout;

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

public interface INxTexture2DMSElement {
    static This load_2d_ms(uint texture, int2 at, int sample);
    static uint3 dimensions_2d_ms(uint texture);
}

#define NX_DEFINE_TEXTURE_2D_MS_ELEMENT(TYPE, SWIZZLE, MS)                 \
    public extension TYPE : INxTexture2DMSElement {                        \
        static TYPE load_2d_ms(uint t, int2 at, int sample) {              \
            return MS(t).Load(at, sample)SWIZZLE;                          \
        }                                                                  \
        static uint3 dimensions_2d_ms(uint t) {                            \
            uint3 dimensions;                                              \
            MS(t).GetDimensions(dimensions.x, dimensions.y, dimensions.z); \
            return dimensions;                                             \
        }                                                                  \
    }

NX_DEFINE_TEXTURE_2D_MS_ELEMENT(float, .x, get_texture_2d_ms)
NX_DEFINE_TEXTURE_2D_MS_ELEMENT(float4, .xyzw, get_texture_2d_ms)
NX_DEFINE_TEXTURE_2D_MS_ELEMENT(uint, .x, get_uint_texture_2d_ms)
NX_DEFINE_TEXTURE_2D_MS_ELEMENT(uint4, .xyzw, get_uint_texture_2d_ms)

#undef NX_DEFINE_TEXTURE_2D_MS_ELEMENT

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

/// A sampled texture's word: its bindless slot (NX_TEXTURE_NONE for none) in
/// the upper 16 bits, its sampler and the NX_TEXTURE_LINEAR and
/// NX_TEXTURE_PREMULTIPLIED flags in the lower.
uint nx_texture_word(uint texture, uint sampler, uint flags) {
    const uint slot = min(texture, NX_TEXTURE_NONE);
    return (slot << 16) | (sampler & NX_TEXTURE_SAMPLER_MASK) |
           (flags & (NX_TEXTURE_LINEAR | NX_TEXTURE_PREMULTIPLIED));
}

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
    public static NxTexture2D<T> from_indices(uint texture, uint sampler,
                                              uint flags = 0u) {
        return from_packed(nx_texture_word(texture, sampler, flags));
    }

    [ForceInline]
    public static NxTexture2D<T> none() {
        return from_indices(NX_TEXTURE_NONE, 0u);
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

public extension NxTexture2D<float> {
    /// The four texels bilinear filtering at @p uv would blend, unfiltered,
    /// through this handle's sampler's addressing: x the one at the lesser x
    /// and greater y, then y, z and w round counter-clockwise, w at the
    /// lesser of both.
    public float4 gather(float2 uv) {
        return get_texture(texture_index())
            .GatherRed(get_sampler(sampler_index()), uv);
    }
}

public extension NxTexture2D<uint> {
    /// The four texels at @p uv, as NxTexture2D<float>::gather orders them.
    public uint4 gather(float2 uv) {
        return get_uint_texture(texture_index())
            .GatherRed(get_sampler(sampler_index()), uv);
    }
}

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

/// A multisampled 2D texture, read a sample at a time, in NxTexture2D's
/// one-word ABI; its sampler bits are unused.
public struct NxTexture2DMS<T>
    where T : INxTexture2DMSElement {
    public uint value;

    [ForceInline]
    public static NxTexture2DMS<T> from_packed(uint packed) {
        NxTexture2DMS<T> result;
        result.value = packed;
        return result;
    }

    [ForceInline]
    public static NxTexture2DMS<T> from_indices(uint texture, uint sampler,
                                                uint flags = 0u) {
        return from_packed(nx_texture_word(texture, sampler, flags));
    }

    [ForceInline]
    public uint texture_index() { return value >> 16; }

    [ForceInline]
    public bool is_valid() { return texture_index() != NX_TEXTURE_NONE; }

    /// Width, height and samples a pixel.
    [ForceInline]
    public uint3 dimensions() { return T.dimensions_2d_ms(texture_index()); }

    [ForceInline]
    public T load(int2 at, int sample) {
        return T.load_2d_ms(texture_index(), at, sample);
    }
}

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
    public static NxTextureCube<T> from_indices(uint texture, uint sampler,
                                     uint flags = 0u) {
        return from_packed(nx_texture_word(texture, sampler, flags));
    }

    [ForceInline]
    public static NxTextureCube<T> none() {
        return from_indices(NX_TEXTURE_NONE, 0u);
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
    public static NxTexture2DArray<T> from_indices(uint texture, uint sampler,
                                     uint flags = 0u) {
        return from_packed(nx_texture_word(texture, sampler, flags));
    }

    [ForceInline]
    public static NxTexture2DArray<T> none() {
        return from_indices(NX_TEXTURE_NONE, 0u);
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
    public static NxTexture3D<T> from_indices(uint texture, uint sampler,
                                     uint flags = 0u) {
        return from_packed(nx_texture_word(texture, sampler, flags));
    }

    [ForceInline]
    public static NxTexture3D<T> none() {
        return from_indices(NX_TEXTURE_NONE, 0u);
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
