# nx gfx

A C++20 graphics library over Vulkan (Metal on Apple platforms) and the Slang
shader library that goes with it. It is the nx2d engine's GPU layer, usable on
its own: bindless resources, buffer device addresses, dynamic rendering,
asynchronous pipeline compilation, presentation, uploads and readbacks, GPU
crash capture and timestamps, behind free functions in `nx::gfx` over two
opaque types, `device` and `command_list`.

It builds on [nx-foundation](https://github.com/AdamFull/nx-foundation), whose
types (`nx::string_view`, `nx::result`, `nx::future`, `nx::thread_pool`,
handles) are part of its interface.

## Using it in a project

Add nx-foundation first, then nx-gfx, and link `nx::gfx`:

```text
git submodule add https://github.com/AdamFull/nx-foundation external/nx-foundation
git submodule add https://github.com/AdamFull/nx-gfx external/nx-gfx
git submodule update --init --recursive
```

```cmake
add_subdirectory(external/nx-foundation)
add_subdirectory(external/nx-gfx)
target_link_libraries(my_game PRIVATE nx::gfx)
```

Built on its own (`PROJECT_IS_TOP_LEVEL`), nx-gfx fetches nx-foundation itself
at the commit it was tested with; point `FETCHCONTENT_SOURCE_DIR_NX_FOUNDATION`
at a checkout to use that instead. Vulkan-Headers, volk and the Vulkan Memory
Allocator are submodules under `third_party/`; a consumer that already defines
`nx::vulkan_headers`, `nx::volk` and `nx::vma` keeps its own.

`nx::gfx_formats` is the backend-free part: the format enum and its helpers,
texel layouts, `texture_data` and shader reflection. Asset tools link it
without linking a GPU backend.

```cpp
#include "nx/gfx/gfx.h"
namespace gfx = nx::gfx;

auto made = gfx::create_device({.application_name = "sample",
                                .engine_name = "sample"});
if (!made)
  return fail(made.error().message);
gfx::device &dev = *made.value();

if (gfx::command_list *cmd = gfx::begin_headless_frame(dev)) {
  // record into *cmd ...
  if (!gfx::end_headless_frame(dev))
    log_error("the frame was not submitted");
}
gfx::destroy_device(&dev);
```

`nx/gfx/vulkan.h` gives a library built on the Vulkan backend (an upscaler, a
video decoder) the device's Vulkan objects and queue submission; nothing else
in the public headers names a backend type.

## Layout

| Path | What |
| --- | --- |
| `include/nx/gfx/` | the public API, one header per area; `gfx.h` has them all |
| `include/nx/gfx/shaders/` | headers shared by C++ and Slang (`interop.h`, `ptr.hpp`, `texture.hpp`, layouts) |
| `shaders/` | the Slang modules `nxg_core`, `nxg_ptr`, `nxg_texture` |
| `src/` | the implementation; `src/vulkan/`, `src/metal/` hold the backends |
| `cmake/NxGfxShaders.cmake` | `nx_gfx_compile_slang`, to build shaders against the library |
| `tests/` | `nx_gfx_tests`, on nx-foundation's nxtest |

## Shaders

Shaders import `nxg_core` (the bindless tables, the null page, `nx_readable`),
`nxg_ptr` (`NxPtr<T>`, word-wise loads and stores, atomics) and `nxg_texture`
(`NxSampler`, `NxTexture2D<T>` and the other handles). A struct both languages
see is declared once in a header that starts from `nx/gfx/shaders/interop.h`;
its rules are in that header. Its `float4` and the rest are plain structs
unless `NX_GFX_INTEROP_MATH_TYPES` names a header supplying the consumer's own
(with `NX_GFX_INTEROP_MATH_LIBRARIES` for what that header needs), defined for
every target that links the library.

`nx_gfx_compile_slang(<target> <out_dir> <include_dirs> <sources>...)`
compiles with the flags the library is validated with:
`-target spirv -emit-spirv-directly -fvk-use-entrypoint-name -profile
spirv_1_3`, or `-target metallib -DNX_TARGET_METAL=1` on Apple.
`NX_GFX_SHADER_INCLUDE_DIRS` is the include path for a consumer's own
pipeline.

## Platforms

Vulkan 1.1 with buffer device addresses and descriptor indexing on Windows,
Linux and Android; Metal on macOS and iOS. The Metal backend is maintained with
the Vulkan one but is not exercised by this repository's tests off Apple
hardware.

## Tests

`NX_GFX_TESTS` (on when built on its own) builds `nx_gfx_tests`, registered
with ctest under the label `gfx`. Tests that need a device skip where none can
be made.
