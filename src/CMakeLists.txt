# nx rhi - the render hardware interface.
#
# Sits above the PAL and below the renderer: it knows about GPUs and window
# surfaces, and nothing about sprites or cameras. Exactly one backend is
# compiled per platform, so the portable headers contain no #ifdef and no
# graphics API types. See docs/design.md section 5.

# The image decode + the cooked .nvtex container. Device-free (stb + foundation
# only), so the asset cooker links it without pulling in Vulkan/Metal. nx_rhi
# re-exports it, so a device consumer sees no change.
add_library(nx_rhi_image STATIC
        types.h
        types.cpp
        image/image.h
        image/image.cpp
        image/stb_image_impl.cpp
        image/stb_image_resize_impl.cpp
        image/texture_nva.h
        image/texture_nva.cpp)
add_library(nx::rhi_image ALIAS nx_rhi_image)
target_include_directories(nx_rhi_image PUBLIC "${CMAKE_SOURCE_DIR}/engine")
target_link_libraries(nx_rhi_image
        PUBLIC nx::foundation
        PRIVATE nx::compile_options nx::stb)

# Texture compression is a desktop cook-time dependency: only the cooker (and the
# tests) call compress_texture, never the runtime. On Android they are absent and
# the whole compress path compiles out. Compressonator's CMP_Core does BC1-7
# (incl. BC6H); astc-encoder does ASTC (CMP_Core has none).
if (NOT ANDROID)
    target_sources(nx_rhi_image PRIVATE image/cmp_encode.cpp)
    target_link_libraries(nx_rhi_image PRIVATE nx::cmp_core nx::astcenc)
    target_compile_definitions(nx_rhi_image PRIVATE NX_TEXTURE_ENCODERS=1)
endif ()
set_target_properties(nx_rhi_image PROPERTIES FOLDER "engine")

# stb's implementation TUs are third-party code; their warnings are not ours to
# fix and they are noisy under /W4.
if (MSVC)
    set_source_files_properties(
            image/stb_image_impl.cpp image/stb_image_resize_impl.cpp
            PROPERTIES COMPILE_OPTIONS "/w")
else ()
    set_source_files_properties(
            image/stb_image_impl.cpp image/stb_image_resize_impl.cpp
            PROPERTIES COMPILE_OPTIONS "-w")
endif ()

set(NX_RHI_HEADERS
        bindless.h
        descs.h
        device.h
        handles.h
        rhi.h
        shader_reflect.h
        upload.h
        upload_ring.h)

set(NX_RHI_SOURCES
        shader_reflect.cpp
        shader_reflect_json.cpp
        upload_ring.cpp)

# Vulkan everywhere except Apple platforms, which get Metal. Note this is not
# the same split as the PAL uses: macOS is GLFW for windowing but Metal for
# rendering.
if (APPLE)
    set(NX_RHI_BACKEND "Metal")
    list(APPEND NX_RHI_HEADERS
            metal/mtl_device.h
            metal/mtl_internal.h)
    list(APPEND NX_RHI_SOURCES
            metal/mtl_bindless.mm
            metal/mtl_common.mm
            metal/mtl_device.mm
            metal/mtl_pipeline.mm
            metal/mtl_swapchain.mm
            metal/mtl_upload.mm)
else ()
    set(NX_RHI_BACKEND "Vulkan")
    list(APPEND NX_RHI_HEADERS
            vulkan/vk_common.h
            vulkan/vk_device.h
            vulkan/vk_upload.h
            vulkan/vk_video.h
            vulkan/vk_vp9.h)
    list(APPEND NX_RHI_SOURCES
            vulkan/vk_bindless.cpp
            vulkan/vk_common.cpp
            vulkan/vk_device.cpp
            vulkan/vk_pipeline.cpp
            vulkan/vk_resources.cpp
            vulkan/vk_swapchain.cpp
            vulkan/vk_upload.cpp
            vulkan/vk_video.cpp
            vulkan/vk_vp9.cpp
            vulkan/vk_vma.cpp)
endif ()

# The RHI owns exactly one shader: bindless.slang, the shader-side binding
# contract. It is a module (imported, never compiled to its own .spv), so
# nothing is compiled here - it is exposed for consumers to put on their include
# path. Renderer and sample shaders are compiled by the code that owns them.
set(NX_RHI_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
        CACHE INTERNAL "Slang module search path for the bindless contract")

message(STATUS "nx_rhi backend: ${NX_RHI_BACKEND}")

add_library(nx_rhi STATIC ${NX_RHI_SOURCES} ${NX_RHI_HEADERS})
add_library(nx::rhi ALIAS nx_rhi)

# runtime is PUBLIC for one type only: SwapchainDesc carries the PAL's
# NativeWindowHandle, which exists precisely to be handed to a surface-creation
# call.
target_link_libraries(nx_rhi
        PUBLIC nx::foundation nx::rhi_image
        PRIVATE nx::compile_options)

if (APPLE)
    target_compile_definitions(nx_rhi PUBLIC NX_RHI_METAL=1)

    # ARC is required: the backend holds Objective-C objects (MTLDevice,
    # CAMetalLayer, drawables) in C++ structs and relies on the compiler to
    # retain/release them. Same arrangement as the iOS platform backend.
    set_source_files_properties(
            metal/mtl_bindless.mm
            metal/mtl_common.mm
            metal/mtl_device.mm
            metal/mtl_pipeline.mm
            metal/mtl_swapchain.mm
            metal/mtl_upload.mm
            PROPERTIES COMPILE_OPTIONS "-fobjc-arc")

    target_link_libraries(nx_rhi PUBLIC
            "-framework Metal"
            "-framework QuartzCore"  # CAMetalLayer
            "-framework Foundation")

    if (IOS)
        target_link_libraries(nx_rhi PUBLIC "-framework UIKit")
    else ()
        # AppKit only for attaching the layer to the GLFW window's NSView.
        target_link_libraries(nx_rhi PUBLIC "-framework AppKit")
    endif ()
else ()
    target_compile_definitions(nx_rhi PUBLIC NX_RHI_VULKAN=1)

    # Vulkan's canonical C initialization is `VkThing{sType}`: the first field
    # is named and every remaining field is deliberately zero-initialized.
    # Clang's aggregate warning treats all 112 of those as omissions, so scope
    # the exception to our Vulkan implementation files rather than weakening
    # warnings for the rest of the engine.
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set_source_files_properties(
                vulkan/vk_bindless.cpp
                vulkan/vk_common.cpp
                vulkan/vk_device.cpp
                vulkan/vk_pipeline.cpp
                vulkan/vk_resources.cpp
                vulkan/vk_swapchain.cpp
                vulkan/vk_upload.cpp
                vulkan/vk_video.cpp
                vulkan/vk_vp9.cpp
                PROPERTIES COMPILE_OPTIONS "-Wno-missing-field-initializers")
    endif ()

    # volk owns entry-point loading, so VMA must not link Vulkan statically
    # (STATIC=0). DYNAMIC=1 then tells VMA to resolve the rest of the API for
    # itself through the vkGetInstanceProcAddr/vkGetDeviceProcAddr pair the
    # device hands it - with both set to 0 it would instead demand every single
    # pointer up front and assert at allocator creation. Must match in every TU
    # that sees the header, hence PUBLIC.
    target_compile_definitions(nx_rhi PUBLIC
            VMA_STATIC_VULKAN_FUNCTIONS=0
            VMA_DYNAMIC_VULKAN_FUNCTIONS=1)

    target_link_libraries(nx_rhi PUBLIC nx::volk nx::vma nx::vulkan_headers)

    # VMA's single implementation TU is third-party code; its warnings are not
    # ours to fix and it is noisy under /W4.
    if (MSVC)
        set_source_files_properties(vulkan/vk_vma.cpp PROPERTIES
                COMPILE_OPTIONS "/w")
    else ()
        set_source_files_properties(vulkan/vk_vma.cpp PROPERTIES
                COMPILE_OPTIONS "-w")
    endif ()
endif ()

set_target_properties(nx_rhi PROPERTIES FOLDER "engine")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}"
        FILES ${NX_RHI_SOURCES} ${NX_RHI_HEADERS})
