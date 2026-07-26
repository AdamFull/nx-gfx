# nx rhi - the render hardware interface.
#
# Sits above the PAL and below the renderer: it knows about GPUs and window
# surfaces, and nothing about sprites or cameras. Exactly one backend is
# compiled per platform, so the portable headers contain no #ifdef and no
# graphics API types. See docs/design.md section 5.

set(NX_RHI_HEADERS
        bindless.h
        descs.h
        device.h
        handles.h
        image/image.h
        rhi.h
        spirv_reflect.h
        types.h
        upload.h)

set(NX_RHI_SOURCES
        image/image.cpp
        image/stb_image_impl.cpp
        spirv_reflect.cpp
        types.cpp)

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
            vulkan/vk_upload.h)
    list(APPEND NX_RHI_SOURCES
            vulkan/vk_bindless.cpp
            vulkan/vk_common.cpp
            vulkan/vk_device.cpp
            vulkan/vk_pipeline.cpp
            vulkan/vk_resources.cpp
            vulkan/vk_swapchain.cpp
            vulkan/vk_upload.cpp
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
        PUBLIC nx::foundation nx::runtime
        PRIVATE nx::compile_options nx::stb nx::spirv_reflect)

# stb_image's implementation TU is third-party code; its warnings are not ours
# to fix and it is noisy under /W4.
if (MSVC)
    set_source_files_properties(image/stb_image_impl.cpp PROPERTIES
            COMPILE_OPTIONS "/w")
else ()
    set_source_files_properties(image/stb_image_impl.cpp PROPERTIES
            COMPILE_OPTIONS "-w")
endif ()

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
