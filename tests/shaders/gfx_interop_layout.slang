// A struct with a field of each of the shader library's shared types, in a
// buffer so the reflection carries its std430 offsets;
// test_gfx_interop_layout.cpp compares them with C++'s. It imports nx::gfx's
// modules alone.
import nxg_core;
import nxg_ptr;
import nxg_texture;

#include "../gfx_interop_probe.h"

StructuredBuffer<GfxInteropProbe> gfx_interop_probe;

[shader("compute")][numthreads(1, 1, 1)] void cs_main() {}
