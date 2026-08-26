
add_library(nx_rhi_image STATIC
        types.h
        types.cpp
        image/image.h
        image/image.cpp
        image/stb_image_impl.cpp
        image/texture_import.h
        image/texture_import.cpp
        image/texture_nva.h
        image/texture_nva.cpp
        shader_nva.h
        shader_nva.cpp)
add_library(nx::rhi_image ALIAS nx_rhi_image)
target_include_directories(nx_rhi_image PUBLIC "${NX_ENGINE_ROOT}/engine")
target_link_libraries(nx_rhi_image
        PUBLIC nx::foundation
        PRIVATE nx::compile_options nx::stb nx::webp_decoder)

set_target_properties(nx_rhi_image PROPERTIES FOLDER "engine")

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set_source_files_properties(
            image/stb_image_impl.cpp
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

set(NX_RHI_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
        CACHE INTERNAL "Slang module search path for the bindless contract")

message(STATUS "nx_rhi backend: ${NX_RHI_BACKEND}")

add_library(nx_rhi STATIC ${NX_RHI_SOURCES} ${NX_RHI_HEADERS})
add_library(nx::rhi ALIAS nx_rhi)

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
                vulkan/vk_device.cpp
                vulkan/vk_pipeline.cpp
                vulkan/vk_resources.cpp
                vulkan/vk_swapchain.cpp
                vulkan/vk_upload.cpp
                vulkan/vk_video.cpp
                vulkan/vk_vp9.cpp
                PROPERTIES COMPILE_OPTIONS "-Wno-missing-field-initializers")
    endif ()

    target_compile_definitions(nx_rhi PUBLIC
            VMA_STATIC_VULKAN_FUNCTIONS=0
            VMA_DYNAMIC_VULKAN_FUNCTIONS=1)

    target_link_libraries(nx_rhi PUBLIC nx::volk nx::vma nx::vulkan_headers)

    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set_source_files_properties(vulkan/vk_vma.cpp PROPERTIES
                COMPILE_OPTIONS "-w")
    endif ()
endif ()

set_target_properties(nx_rhi PROPERTIES FOLDER "engine")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}"
        FILES ${NX_RHI_SOURCES} ${NX_RHI_HEADERS})
