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
        rhi.h
        types.h)

set(NX_RHI_SOURCES
        types.cpp)

# Vulkan everywhere except Apple platforms, which get Metal. Note this is not
# the same split as the PAL uses: macOS is GLFW for windowing but Metal for
# rendering.
if (APPLE)
    set(NX_RHI_BACKEND "Metal")
    message(FATAL_ERROR "nx_rhi: the Metal backend is not implemented yet")
else ()
    set(NX_RHI_BACKEND "Vulkan")
    list(APPEND NX_RHI_HEADERS
            vulkan/vk_common.h
            vulkan/vk_device.h)
    list(APPEND NX_RHI_SOURCES
            vulkan/vk_bindless.cpp
            vulkan/vk_common.cpp
            vulkan/vk_device.cpp
            vulkan/vk_resources.cpp
            vulkan/vk_swapchain.cpp
            vulkan/vk_vma.cpp)
endif ()

# Compile the Slang shaders to SPIR-V next to the executable, under bin/shaders.
include(NxShaders)
nx_compile_slang(nx_shaders
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/triangle.slang")

message(STATUS "nx_rhi backend: ${NX_RHI_BACKEND}")

add_library(nx_rhi STATIC ${NX_RHI_SOURCES} ${NX_RHI_HEADERS})
add_library(nx::rhi ALIAS nx_rhi)

# runtime is PUBLIC for one type only: SwapchainDesc carries the PAL's
# NativeWindowHandle, which exists precisely to be handed to a surface-creation
# call.
target_link_libraries(nx_rhi
        PUBLIC nx::foundation nx::runtime
        PRIVATE nx::compile_options)

if (NOT APPLE)
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

if (TARGET nx_shaders)
    add_dependencies(nx_rhi nx_shaders)
    set_target_properties(nx_shaders PROPERTIES FOLDER "engine")
endif ()

set_target_properties(nx_rhi PROPERTIES FOLDER "engine")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}"
        FILES ${NX_RHI_SOURCES} ${NX_RHI_HEADERS})
