# nx_gfx_compile_slang(<target> <out_dir> <include_dirs> <sources>...): compiles
# each Slang source to <out_dir>/<name>.spv (.metallib on Apple) and its
# reflection JSON, with the shader library's modules and headers on the
# include path, under a custom target. A consumer with its own Slang pipeline
# uses NX_GFX_SHADER_INCLUDE_DIRS and these flags instead.

if (NOT NX_SLANGC)
    find_program(NX_SLANGC NAMES slangc HINTS "$ENV{VULKAN_SDK}/bin")
endif ()

function(nx_gfx_compile_slang target out_dir include_dirs)
    if (NOT NX_SLANGC)
        message(STATUS "slangc not found; ${target} is not built")
        return()
    endif ()

    set(_include_args "")
    foreach (_dir ${NX_GFX_SHADER_INCLUDE_DIRS} ${include_dirs})
        list(APPEND _include_args -I "${_dir}")
    endforeach ()

    set(_normalize "")
    set(_normalize_dep "")
    if (APPLE)
        set(_ext "metallib")
        set(_target_args -target metallib -DNX_TARGET_METAL=1)
    else ()
        set(_ext "spv")
        set(_target_args -target spirv -emit-spirv-directly
                -fvk-use-entrypoint-name -profile spirv_1_3)
        # Every module goes through nx::gfx::normalize_spirv after slangc.
        if (TARGET nx_gfx_spirv)
            set(_normalize COMMAND "$<TARGET_FILE:nx_gfx_spirv>" normalize)
            set(_normalize_dep nx_gfx_spirv)
        elseif (NX_GFX_SPIRV_TOOL)
            set(_normalize COMMAND "${NX_GFX_SPIRV_TOOL}" normalize)
        else ()
            message(WARNING "${target}: no nx_gfx_spirv to run (set "
                    "NX_GFX_SPIRV_TOOL to a host build); modules keep what "
                    "slangc declares")
        endif ()
    endif ()

    set(_outputs "")
    foreach (_src ${ARGN})
        get_filename_component(_name "${_src}" NAME_WE)
        set(_out "${out_dir}/${_name}.${_ext}")
        set(_refl "${out_dir}/${_name}.refl.json")
        set(_dep "${CMAKE_CURRENT_BINARY_DIR}/${_name}.${_ext}.d")
        set(_normalize_args "")
        if (_normalize)
            set(_normalize_args "${_out}" "${_out}")
        endif ()
        add_custom_command(
                OUTPUT "${_out}" "${_refl}"
                COMMAND "${CMAKE_COMMAND}" -E make_directory "${out_dir}"
                COMMAND "${NX_SLANGC}" "${_src}" ${_target_args} ${_include_args}
                -o "${_out}" -reflection-json "${_refl}" -depfile "${_dep}"
                ${_normalize} ${_normalize_args}
                MAIN_DEPENDENCY "${_src}"
                DEPENDS ${_normalize_dep}
                DEPFILE "${_dep}"
                COMMENT "slangc ${_name}.slang -> ${_name}.${_ext} + reflection"
                VERBATIM)
        list(APPEND _outputs "${_out}" "${_refl}")
    endforeach ()
    add_custom_target(${target} DEPENDS ${_outputs})
endfunction()
