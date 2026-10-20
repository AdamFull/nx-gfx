
# Formats and texel layouts: what the RHI and the asset tools share, with no
# backend behind it.
add_library(nx_rhi_types STATIC
        include/nx/gfx/bindless.h
        include/nx/gfx/commands.h
        include/nx/gfx/device.h
        include/nx/gfx/frame.h
        include/nx/gfx/pipelines.h
        include/nx/gfx/reflection.h
        include/nx/gfx/resources.h
        include/nx/gfx/types.h
        include/nx/gfx/upload.h
        types.h
        types.cpp
        texture_data.h
        texture_data.cpp
        shader_reflect.h
        shader_reflect.cpp
        shader_reflect_json.cpp)
add_library(nx::rhi_types ALIAS nx_rhi_types)
target_include_directories(nx_rhi_types PUBLIC
        "${NX_ENGINE_ROOT}/engine"
        "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(nx_rhi_types
        PUBLIC nx::foundation
        PRIVATE nx::compile_options)
set_target_properties(nx_rhi_types PROPERTIES FOLDER "engine")

set(NX_RHI_HEADERS
        bindless.h
        descs.h
        device.h
        gpu_crash.h
        handles.h
        native_driver.h
        pipeline_compiler.h
        rhi.h
        upload.h)

set(NX_RHI_SOURCES
        gpu_crash.cpp)

if (APPLE)
    set(NX_RHI_BACKEND "Metal")
    list(APPEND NX_RHI_HEADERS
            metal/mtl_device.h
            metal/mtl_gpu_crash.h
            metal/mtl_internal.h)
    list(APPEND NX_RHI_SOURCES
            metal/mtl_bindless.mm
            metal/mtl_common.mm
            metal/mtl_device.mm
            metal/mtl_gpu_crash.mm
            metal/mtl_pipeline.mm
            metal/mtl_swapchain.mm
            metal/mtl_upload.mm)
else ()
    set(NX_RHI_BACKEND "Vulkan")
    list(APPEND NX_RHI_HEADERS
            vulkan/vk_common.h
            vulkan/vk_gpu_crash.h
            vulkan/vk_gpu_profiler.h
            vulkan/vk_device.h
            vulkan/vk_upload.h)
    list(APPEND NX_RHI_SOURCES
            vulkan/vk_bindless.cpp
            vulkan/vk_common.cpp
            vulkan/vk_gpu_crash.cpp
            vulkan/vk_gpu_profiler.cpp
            vulkan/vk_device.cpp
            vulkan/vk_pipeline.cpp
            vulkan/vk_resources.cpp
            vulkan/vk_swapchain.cpp
            vulkan/vk_upload.cpp
            vulkan/vk_vma.cpp)
    if (WIN32)
        list(APPEND NX_RHI_HEADERS vulkan/vk_composition_win32.h)
        list(APPEND NX_RHI_SOURCES vulkan/vk_composition_win32.cpp)
    endif ()
endif ()

set(NX_RHI_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
        CACHE INTERNAL "Slang module search path for the bindless contract")

message(STATUS "nx_rhi backend: ${NX_RHI_BACKEND}")

add_library(nx_rhi STATIC ${NX_RHI_SOURCES} ${NX_RHI_HEADERS})
add_library(nx::rhi ALIAS nx_rhi)

target_link_libraries(nx_rhi
        PUBLIC nx::foundation nx::rhi_types
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
            metal/mtl_gpu_crash.mm
            metal/mtl_pipeline.mm
            metal/mtl_swapchain.mm
            metal/mtl_upload.mm
            PROPERTIES COMPILE_OPTIONS "-fobjc-arc")

    target_link_libraries(nx_rhi PUBLIC
            "-framework Metal"
            "-framework QuartzCore"
            "-framework Foundation")

    if (IOS)
        target_link_libraries(nx_rhi PUBLIC "-framework UIKit")
    else ()
        target_link_libraries(nx_rhi PUBLIC "-framework AppKit")
    endif ()
else ()
    target_compile_definitions(nx_rhi PUBLIC NX_RHI_VULKAN=1)

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set_source_files_properties(
                vulkan/vk_bindless.cpp
                vulkan/vk_common.cpp
                vulkan/vk_gpu_crash.cpp
                vulkan/vk_gpu_profiler.cpp
                vulkan/vk_device.cpp
                vulkan/vk_pipeline.cpp
                vulkan/vk_resources.cpp
                vulkan/vk_swapchain.cpp
                vulkan/vk_upload.cpp
                PROPERTIES COMPILE_OPTIONS "-Wno-missing-field-initializers")
    endif ()

    target_compile_definitions(nx_rhi PUBLIC
            VMA_STATIC_VULKAN_FUNCTIONS=0
            VMA_DYNAMIC_VULKAN_FUNCTIONS=1)

    target_link_libraries(nx_rhi PUBLIC nx::volk nx::vma nx::vulkan_headers)
    if (WIN32)
        target_link_libraries(nx_rhi PRIVATE d3d11 dxgi dcomp)
    endif ()

    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set_source_files_properties(vulkan/vk_vma.cpp PROPERTIES
                COMPILE_OPTIONS "-w")
    endif ()
endif ()

set_target_properties(nx_rhi PROPERTIES FOLDER "engine")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}"
        FILES ${NX_RHI_SOURCES} ${NX_RHI_HEADERS})
