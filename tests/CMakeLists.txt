# nx_gfx_tests: the library's own suite, on nxtest from nx-foundation.
include("${NX_GFX_ROOT}/cmake/NxGfxShaders.cmake")

add_executable(nx_gfx_tests
        gfx_interop_probe.h
        owned_device.h
        pipeline_test_utils.h
        test_gfx_api.cpp
        test_gfx_interop_layout.cpp
        test_gpu_crash.cpp
        test_native.cpp
        test_pipeline_compiler.cpp)
target_include_directories(nx_gfx_tests PRIVATE "${NX_GFX_ROOT}")
target_link_libraries(nx_gfx_tests
        PRIVATE nx::test nx::gfx nx::foundation_options)

set(NX_GFX_TEST_SHADERS "${CMAKE_CURRENT_BINARY_DIR}/shaders")
target_compile_definitions(nx_gfx_tests PRIVATE
        NX_GFX_TEST_SHADER_DIR="${NX_GFX_TEST_SHADERS}")
nx_gfx_compile_slang(nx_gfx_test_shaders "${NX_GFX_TEST_SHADERS}" ""
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/gfx_interop_layout.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/native_read.slang")
if (TARGET nx_gfx_test_shaders)
    add_dependencies(nx_gfx_tests nx_gfx_test_shaders)
    set_target_properties(nx_gfx_test_shaders PROPERTIES FOLDER "nx-gfx")
endif ()
set_target_properties(nx_gfx_tests PROPERTIES FOLDER "nx-gfx")

nxtest_discover_tests(nx_gfx_tests LABELS gfx TIMEOUT 180)
# Again on the render-pass route, which a device with dynamic rendering takes
# only when told to.
nxtest_discover_tests(nx_gfx_tests PREFIX "render passes: "
        LABELS gfx routes TIMEOUT 180
        ENVIRONMENT NX_GFX_DISABLE_ROUTES=dynamic_rendering)
