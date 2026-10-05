// The bindless texture handles shaders sample, load and store through, the
// Slang half of the pair whose C++ half is nx_texture.hpp; the bindless
// tables and texel types they reach are core.slang's. Included by core.slang.

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

/// A 2D image in two words: a texture, or a region of an atlas with a
/// texture to fall back on. The atlas methods are atlas.slang's, so the
/// dependency runs one way.
public struct NxImage2DResolver<T>
    where T : INxTexture2DElement {
    public NxTexture2D<T> texture;
    public uint metadata;

    [ForceInline]
    public static NxImage2DResolver<T> from_texture(NxTexture2D<T> texture) {
        NxImage2DResolver<T> result;
        result.texture = texture;
        result.metadata = 0u;
        return result;
    }

    [ForceInline]
    public static NxImage2DResolver<T> from_atlas(NxTexture2D<T> fallback,
                                                  uint atlas, uint region) {
        NxImage2DResolver<T> result;
        result.texture = fallback;
        result.metadata = NX_IMAGE_ATLAS |
                          ((atlas & NX_IMAGE_ATLAS_INDEX_MASK) << 16) |
                          (region & NX_IMAGE_REGION_MASK);
        return result;
    }

    [ForceInline]
    public bool is_atlas() { return (metadata & NX_IMAGE_ATLAS) != 0u; }

    [ForceInline]
    public NxTexture2D<T> as_texture() { return texture; }
}
