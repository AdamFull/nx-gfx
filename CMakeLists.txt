# nx-gfx: the nx::gfx graphics library over Vulkan (Metal on Apple) and its
# Slang shader library. Its public headers are include/nx/gfx/; src/ is
# private and includes itself from this directory ("src/vulkan/...").

set(NX_GFX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

set(NX_GFX_PUBLIC_HEADERS
        include/nx/gfx/bindless.h
        include/nx/gfx/commands.h
        include/nx/gfx/crash.h
        include/nx/gfx/device.h
        include/nx/gfx/frame.h
        include/nx/gfx/gfx.h
        include/nx/gfx/native.h
        include/nx/gfx/pipeline_job.h
        include/nx/gfx/pipelines.h
        include/nx/gfx/reflection.h
        include/nx/gfx/resources.h
        include/nx/gfx/shaders/core_layout.h
        include/nx/gfx/shaders/interop.h
        include/nx/gfx/shaders/ptr.hpp
        include/nx/gfx/shaders/shared.h
        include/nx/gfx/shaders/texture.hpp
        include/nx/gfx/shaders/texture_layout.h
        include/nx/gfx/types.h
        include/nx/gfx/upload.h)

# Formats, texel layouts and shader reflection: what the library and asset
# tools share, with no backend behind it.
add_library(nx_gfx_formats STATIC
        ${NX_GFX_PUBLIC_HEADERS}
        src/types.cpp
        src/texture_data.cpp
        src/shader_reflect.cpp
        src/shader_reflect_json.cpp)
add_library(nx::gfx_formats ALIAS nx_gfx_formats)
target_include_directories(nx_gfx_formats
        PUBLIC "${NX_GFX_ROOT}/include"
        PRIVATE "${NX_GFX_ROOT}")
target_link_libraries(nx_gfx_formats
        PUBLIC nx::foundation
        PRIVATE nx::foundation_options)
# Shared structs' float4 and the rest (nx/gfx/shaders/interop.h) are the
# consumer's own when it names a header supplying them, in every translation
# unit alike, with the targets that header needs.
if (NX_GFX_INTEROP_MATH_TYPES)
    target_compile_definitions(nx_gfx_formats PUBLIC
            "NX_GFX_INTEROP_MATH_TYPES=\"${NX_GFX_INTEROP_MATH_TYPES}\"")
    target_link_libraries(nx_gfx_formats PUBLIC ${NX_GFX_INTEROP_MATH_LIBRARIES})
endif ()
set_target_properties(nx_gfx_formats PROPERTIES FOLDER "nx-gfx")

set(NX_GFX_HEADERS src/device.h)
set(NX_GFX_SOURCES
        src/gfx.cpp
        src/gpu_crash.cpp)

if (APPLE)
    set(NX_GFX_BACKEND "Metal")
    list(APPEND NX_GFX_HEADERS
            src/metal/mtl_device.h
            src/metal/mtl_gpu_crash.h
            src/metal/mtl_internal.h)
    list(APPEND NX_GFX_SOURCES
            src/metal/mtl_bindless.mm
            src/metal/mtl_common.mm
            src/metal/mtl_device.mm
            src/metal/mtl_gpu_crash.mm
            src/metal/mtl_pipeline.mm
            src/metal/mtl_swapchain.mm
            src/metal/mtl_upload.mm)
else ()
    set(NX_GFX_BACKEND "Vulkan")
    list(APPEND NX_GFX_HEADERS
            include/nx/gfx/vulkan.h
            src/vulkan/vk_common.h
            src/vulkan/vk_gpu_crash.h
            src/vulkan/vk_gpu_profiler.h
            src/vulkan/vk_device.h
            src/vulkan/vk_mem_alloc_include.h
            src/vulkan/vk_upload.h)
    list(APPEND NX_GFX_SOURCES
            src/vulkan/vk_bindless.cpp
            src/vulkan/vk_common.cpp
            src/vulkan/vk_gpu_crash.cpp
            src/vulkan/vk_gpu_profiler.cpp
            src/vulkan/vk_device.cpp
            src/vulkan/vk_gfx.cpp
            src/vulkan/vk_pipeline.cpp
            src/vulkan/vk_resources.cpp
            src/vulkan/vk_swapchain.cpp
            src/vulkan/vk_upload.cpp
            src/vulkan/vk_vma.cpp)
    if (WIN32)
        list(APPEND NX_GFX_HEADERS src/vulkan/vk_composition_win32.h)
        list(APPEND NX_GFX_SOURCES src/vulkan/vk_composition_win32.cpp)
    endif ()
endif ()

set(NX_GFX_SHADER_INCLUDE_DIRS
        "${NX_GFX_ROOT}/shaders"
        "${NX_GFX_ROOT}/include"
        CACHE INTERNAL "The shader library's Slang modules and shared headers")

message(STATUS "nx_gfx backend: ${NX_GFX_BACKEND}")

add_library(nx_gfx STATIC ${NX_GFX_SOURCES} ${NX_GFX_HEADERS})
add_library(nx::gfx ALIAS nx_gfx)
target_include_directories(nx_gfx PRIVATE "${NX_GFX_ROOT}")
target_link_libraries(nx_gfx
        PUBLIC nx::foundation nx::gfx_formats
        PRIVATE nx::foundation_options)

if (APPLE)
    target_compile_definitions(nx_gfx PUBLIC NX_RHI_METAL=1)

    # ARC is required: the backend holds Objective-C objects (MTLDevice,
    # CAMetalLayer, drawables) in C++ structs and relies on the compiler to
    # retain/release them. Same arrangement as the iOS platform backend.
    set_source_files_properties(
            src/metal/mtl_bindless.mm
            src/metal/mtl_common.mm
            src/metal/mtl_device.mm
            src/metal/mtl_gpu_crash.mm
            src/metal/mtl_pipeline.mm
            src/metal/mtl_swapchain.mm
            src/metal/mtl_upload.mm
            PROPERTIES COMPILE_OPTIONS "-fobjc-arc")

    target_link_libraries(nx_gfx PUBLIC
            "-framework Metal"
            "-framework QuartzCore"
            "-framework Foundation")

    if (IOS)
        target_link_libraries(nx_gfx PUBLIC "-framework UIKit")
    else ()
        target_link_libraries(nx_gfx PUBLIC "-framework AppKit")
    endif ()
else ()
    target_compile_definitions(nx_gfx PUBLIC NX_RHI_VULKAN=1)

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set_source_files_properties(
                src/vulkan/vk_bindless.cpp
                src/vulkan/vk_common.cpp
                src/vulkan/vk_gpu_crash.cpp
                src/vulkan/vk_gpu_profiler.cpp
                src/vulkan/vk_device.cpp
                src/vulkan/vk_pipeline.cpp
                src/vulkan/vk_resources.cpp
                src/vulkan/vk_swapchain.cpp
                src/vulkan/vk_upload.cpp
                PROPERTIES COMPILE_OPTIONS "-Wno-missing-field-initializers")
    endif ()

    target_compile_definitions(nx_gfx PUBLIC
            VMA_STATIC_VULKAN_FUNCTIONS=0
            VMA_DYNAMIC_VULKAN_FUNCTIONS=1)

    target_link_libraries(nx_gfx PUBLIC nx::volk nx::vma nx::vulkan_headers)
    if (WIN32)
        target_link_libraries(nx_gfx PRIVATE d3d11 dxgi dcomp)
    endif ()

    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set_source_files_properties(src/vulkan/vk_vma.cpp PROPERTIES
                COMPILE_OPTIONS "-w")
    endif ()
endif ()

set_target_properties(nx_gfx PROPERTIES FOLDER "nx-gfx")
source_group(TREE "${NX_GFX_ROOT}"
        FILES ${NX_GFX_PUBLIC_HEADERS} ${NX_GFX_SOURCES} ${NX_GFX_HEADERS})

option(NX_GFX_TESTS "Build nx_gfx_tests (needs nx-foundation's nxtest)" OFF)
if (NX_GFX_TESTS)
    add_subdirectory(tests)
endif ()
