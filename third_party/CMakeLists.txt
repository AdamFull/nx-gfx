# Vulkan-Headers, volk and the Vulkan Memory Allocator (git submodules), for the
# Vulkan backend. A consumer that already defines nx::vulkan_headers, nx::volk
# and nx::vma keeps its own.
#
# Headers, loader and allocator come from submodules rather than the installed
# Vulkan SDK, so the build does not depend on VULKAN_SDK being set (or correct).
# volk loads the ICD at run time, so nothing links vulkan-1 either.

if (APPLE OR TARGET nx::vulkan_headers)
    return()
endif ()

foreach (_submodule Vulkan-Headers volk VulkanMemoryAllocator)
    if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_submodule}/CMakeLists.txt")
        message(FATAL_ERROR
                "nx-gfx's third_party/${_submodule} is empty. Fetch the "
                "submodules first:\n  git submodule update --init --recursive")
    endif ()
endforeach ()

# Directory-scoped: third-party suites should never land in a ctest list.
set(BUILD_TESTING OFF)

add_subdirectory(Vulkan-Headers EXCLUDE_FROM_ALL)
add_library(nx::vulkan_headers ALIAS Vulkan-Headers)

# volk needs the platform surface extension defined at compile time, because
# the WSI entry points it loads differ per platform.
if (WIN32)
    set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_WIN32_KHR CACHE STRING "" FORCE)
elseif (ANDROID)
    set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_ANDROID_KHR CACHE STRING "" FORCE)
elseif (UNIX)
    # Both display backends are compiled; the windowing layer decides which is
    # live at run time.
    set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_XLIB_KHR VK_USE_PLATFORM_WAYLAND_KHR
            CACHE STRING "" FORCE)
endif ()

set(VOLK_INSTALL OFF CACHE BOOL "" FORCE)
set(VOLK_HEADERS_ONLY OFF CACHE BOOL "" FORCE)

# volk hunts for Vulkan headers in this order: VULKAN_HEADERS_INSTALL_DIR,
# FindVulkan, $VULKAN_SDK, then the Vulkan-Headers target. The $VULKAN_SDK
# branch wins before it ever reaches our submodule - and a stale VULKAN_SDK
# (left behind by an SDK upgrade) points at a directory that no longer exists,
# which fails the build with a missing vulkan/vk_platform.h. Setting the first
# option explicitly pins volk to the submodule and takes the SDK out of the
# picture entirely.
set(VULKAN_HEADERS_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Vulkan-Headers"
        CACHE PATH "" FORCE)

add_subdirectory(volk EXCLUDE_FROM_ALL)
add_library(nx::volk ALIAS volk)

add_subdirectory(VulkanMemoryAllocator EXCLUDE_FROM_ALL)
add_library(nx::vma ALIAS VulkanMemoryAllocator)

# Neither header set is our code: keep their warnings out of builds that use
# them.
foreach (_vk_target Vulkan-Headers VulkanMemoryAllocator volk)
    get_target_property(_vk_includes ${_vk_target} INTERFACE_INCLUDE_DIRECTORIES)
    if (_vk_includes)
        set_target_properties(${_vk_target} PROPERTIES
                INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_vk_includes}")
    endif ()
    set_target_properties(${_vk_target} PROPERTIES FOLDER "nx-gfx/third_party")
endforeach ()
unset(_vk_includes)
