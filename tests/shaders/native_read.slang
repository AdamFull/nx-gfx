// Reads one texel of a bindless texture into a device address, for a test to
// see that work recorded after native commands still finds its textures.
import nxg_texture;

struct Push {
    float4 *out;
    uint texture;
};

[[vk::push_constant]]
Push pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void read_main() {
    pc.out[0] = NxTexture2D<float4>.from_packed(pc.texture).load(int3(0, 0, 0));
}
