cmake_minimum_required(VERSION 3.22)

set(SHADER_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)

if(WIN32 OR LINUX)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(OptiX_INSTALL_DIR $ENV{OPTIX_DIR})
    find_package(OptiX REQUIRED)

    # openpbr.h includes its own headers relative to this root, so the directory
    # itself has to be on the path -- ${CMAKE_SOURCE_DIR}/third_party above is
    # one level too high.
    #
    # The *generated* copy goes first. Adobe's OpenPBR cannot be compiled by nvcc
    # as shipped -- 211 of its definitions carry no execution space, and its eight
    # lookup tables are host globals -- and neither is reachable from the
    # preprocessor. openpbr_device_headers.py rewrites both into the build tree,
    # and the submodule stays byte-identical to upstream. Ordering is the whole
    # mechanism: `#include "openpbr.h"` and every relative include beneath it
    # resolve here, and only fall through to the real root if this copy is
    # missing a file -- which would mean the rewrite is stale.
    set(OPENPBR_DEVICE_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/openpbr_device)
    file(GLOB_RECURSE OPENPBR_VENDORED_HEADERS
        ${CMAKE_SOURCE_DIR}/third_party/openpbr_bsdf/*.h)
    add_custom_command(
        OUTPUT ${OPENPBR_DEVICE_INCLUDE_DIR}/openpbr.h
        COMMAND ${CMAKE_COMMAND} -E env python3
            ${CMAKE_SOURCE_DIR}/tools/openpbr_device_headers.py
            ${CMAKE_SOURCE_DIR}/third_party/openpbr_bsdf
            ${OPENPBR_DEVICE_INCLUDE_DIR}
        DEPENDS ${CMAKE_SOURCE_DIR}/tools/openpbr_device_headers.py ${OPENPBR_VENDORED_HEADERS}
        COMMENT "Rewriting OpenPBR headers for device compilation"
        VERBATIM
    )
    add_custom_target(openpbr_device_headers DEPENDS ${OPENPBR_DEVICE_INCLUDE_DIR}/openpbr.h)

    set(OPTIX_SHADER_INCLUDE_DIRS
        ${OPENPBR_DEVICE_INCLUDE_DIR}
        ${OptiX_INCLUDE}
        ${CMAKE_SOURCE_DIR}/src/render/include/strelka/render
        ${CMAKE_SOURCE_DIR}/src/render/optix
        ${CMAKE_SOURCE_DIR}/src/render
        ${CMAKE_SOURCE_DIR}/src/render/include
        ${CMAKE_SOURCE_DIR}/third_party
        ${SHADER_COMMON_DIR}
        ${CMAKE_SOURCE_DIR}/src/material/include
        ${CMAKE_SOURCE_DIR}/third_party/openpbr_bsdf
        ${CMAKE_CURRENT_SOURCE_DIR}/optix
    )

    # Target Ada for the OptiX IR baseline and keep line tables for Nsight.
    # `-c` lets ccache recognise the custom command as compilation.
    set(OPTIX_NVCC_FLAGS --gpu-architecture=compute_89 --use_fast_math --relocatable-device-code=true
        --expt-relaxed-constexpr --generate-line-info -c)

    # Keep OpenPBR device assertions in Debug; release assertions abort a whole launch.
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND OPTIX_NVCC_FLAGS -DNDEBUG)
    endif()

    set(NVCC_COMMAND ${CMAKE_CUDA_COMPILER})
    if(CMAKE_CUDA_COMPILER_LAUNCHER)
        list(PREPEND NVCC_COMMAND ${CMAKE_CUDA_COMPILER_LAUNCHER})
    endif()
    set(OPTIX_INCLUDE_FLAGS)
    foreach(include_dir IN LISTS OPTIX_SHADER_INCLUDE_DIRS)
        list(APPEND OPTIX_INCLUDE_FLAGS "-I${include_dir}")
    endforeach()
    file(GLOB_RECURSE OPTIX_SHADER_HEADERS CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/*.h
        ${SHADER_COMMON_DIR}/*.h
        ${CMAKE_SOURCE_DIR}/src/material/include/*.h
        ${CMAKE_SOURCE_DIR}/src/render/optix/*.h
    )

    set(OPTIXIR_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/OptixRender.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/OptixRender_closest_hit.cu
    )
    set(generated_optixir_files)
    foreach(source IN LISTS OPTIXIR_SOURCES)
        get_filename_component(source_name "${source}" NAME)
        set(optixir "${CMAKE_CURRENT_BINARY_DIR}/strelka_shaders_generated_${source_name}.optixir")
        add_custom_command(
            OUTPUT "${optixir}"
            COMMAND ${NVCC_COMMAND} ${OPTIX_NVCC_FLAGS} ${OPTIX_INCLUDE_FLAGS}
                    --optix-ir "${source}" -o "${optixir}"
            DEPENDS "${source}" ${OPTIX_SHADER_HEADERS} ${OPENPBR_DEVICE_INCLUDE_DIR}/openpbr.h
            COMMENT "Compiling ${source_name} to OptiX IR"
            COMMAND_EXPAND_LISTS
            VERBATIM
        )
        list(APPEND generated_optixir_files "${optixir}")
    endforeach()
    add_custom_target(generate_optixir DEPENDS ${generated_optixir_files})

    set(CUDA_KERNEL_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/postprocessing/Tonemappers.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/skinning/skinning.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/postprocessing/DenoiseGuides.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/sharc_resolve.cu
    )

    add_library(strelka_shaders STATIC
        ${CUDA_KERNEL_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/postprocessing/Tonemappers.h
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/postprocessing/DenoiseGuides.h
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/postprocessing/Guides.h
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/skinning/skinning.h
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/sharc.h
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/sharc_grid.h
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/sharc_resolve.h
    )
    add_dependencies(strelka_shaders generate_optixir)

    target_include_directories(strelka_shaders
        PRIVATE ${OPTIX_SHADER_INCLUDE_DIRS}
        PUBLIC
            ${SHADER_COMMON_DIR}
            ${CMAKE_SOURCE_DIR}/src/material/include
    )

    # Copy OPTIXIR files to output directory using file-level dependencies
    # (POST_BUILD can be skipped by ninja when the target doesn't need relinking)
    set(OPTIXIR_OUTPUT_DIR "${OUTPUT_DIRECTORY}/optix")
    set(copied_optixir_files "")
    foreach(optixir_file IN LISTS generated_optixir_files)
        get_filename_component(optixir_name "${optixir_file}" NAME)
        set(dest_file "${OPTIXIR_OUTPUT_DIR}/${optixir_name}")
        add_custom_command(
            OUTPUT "${dest_file}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${OPTIXIR_OUTPUT_DIR}"
            COMMAND ${CMAKE_COMMAND} -E copy "${optixir_file}" "${dest_file}"
            DEPENDS "${optixir_file}"
            COMMENT "Copying ${optixir_name} to output directory"
        )
        list(APPEND copied_optixir_files "${dest_file}")
    endforeach()
    add_custom_target(copy_optixir ALL DEPENDS ${copied_optixir_files})
    add_dependencies(strelka_shaders copy_optixir)

    # The renderer loads these by name at startup, so an installed binary without
    # them is a binary that cannot draw. Same relative path as in the build tree;
    # resolveResourcePath() finds it under share/strelka once the executable is
    # in bin/.
    if(NOT APPLE)
        install(FILES ${copied_optixir_files} DESTINATION ${CMAKE_INSTALL_DATADIR}/strelka/optix)
    endif()
endif()

if(APPLE)
    # Fail at configure time with an actionable message instead of an opaque
    # "cannot execute tool 'metal'" error in the middle of the build.
    execute_process(
        COMMAND xcrun -sdk macosx --find metal
        RESULT_VARIABLE METAL_COMPILER_FOUND
        OUTPUT_VARIABLE METAL_COMPILER_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT METAL_COMPILER_FOUND EQUAL 0)
        message(FATAL_ERROR
            "The Metal shader compiler was not found.\n"
            "Install the Metal toolchain with:\n"
            "    xcodebuild -downloadComponent MetalToolchain\n"
            "(and make sure `xcode-select -p` points at a full Xcode install).")
    endif()

    set(METAL_INCLUDE_FLAGS
        -I ${CMAKE_SOURCE_DIR}/src/material/include
        -I ${CMAKE_SOURCE_DIR}/third_party/openpbr_bsdf
        -I ${SHADER_COMMON_DIR}
        -I ${CMAKE_CURRENT_SOURCE_DIR}/metal
    )

    # -gline-tables-only/-frecord-sources keep Metal frame captures usable.
    #
    # metal3.1 rather than 3.0 for one feature: curve intersection. `curve_data`,
    # `geometry_type::curve` and `intersection_result::curve_parameter` do not
    # exist before it, and hair is the one thing in these scenes that must not be
    # triangulated. Nothing else in the tree needs 3.1, and the Metal 4 pipelines
    # are built from this same library either way -- MTL4::Compiler takes a
    # metal3.x metallib, and the acceleration structures both paths traverse are
    # built through the Metal 3 API regardless of which one submits the frame.
    set(METAL_BUILD_FLAGS -Wall -std=metal3.1)
    # Two kernels that do the same arithmetic in a different order (the megakernel
    # and the wavefront tracer) fuse multiply-adds differently, which leaves a
    # ~1e-7 relative difference in most pixels and, through chaotic paths, a
    # visible one in a handful. Turning contraction off makes the two bit-identical
    # and is the only way to tell an arithmetic difference from a logic bug.
    option(STRELKA_METAL_STRICT_FP "Disable Metal FMA contraction (for A/B image comparison)" OFF)
    if(STRELKA_METAL_STRICT_FP)
        list(APPEND METAL_BUILD_FLAGS -ffp-contract=off)
    endif()
    option(STRELKA_METAL_PROFILE_SOURCES
        "Embed Metal line tables and sources for Shader Profiler captures" OFF)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR STRELKA_METAL_PROFILE_SOURCES)
        list(APPEND METAL_BUILD_FLAGS -gline-tables-only -frecord-sources)
    endif()

    set(SHADER_OUTPUT_DIR ${OUTPUT_DIRECTORY}/metal/shaders)
    set(SHADER_AIR_DIR ${CMAKE_CURRENT_BINARY_DIR}/air)
    # Keep Clang's Metal module cache inside the build tree. A global cache under
    # the developer's home directory makes hermetic/sandboxed builds fail and
    # lets two unrelated Xcode toolchains poison each other's precompiled modules.
    list(APPEND METAL_BUILD_FLAGS -fmodules-cache-path=${SHADER_AIR_DIR}/module-cache)

    # Copy fullScreen.metal (no compilation needed, loaded at runtime)
    add_custom_command(
        OUTPUT ${SHADER_OUTPUT_DIR}/fullScreen.metal
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/metal/fullScreen.metal
            ${SHADER_OUTPUT_DIR}/fullScreen.metal
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/metal/fullScreen.metal
        COMMENT "Copying fullScreen.metal"
    )

    # Compile Metal shaders.
    #
    # `metal -MD` emits a Make-style depfile listing every transitively included
    # header (ShaderTypes.h, bsdf.h, material_math.h, ...). Feeding it back to the
    # generator via DEPFILE is what makes header edits actually trigger a rebuild;
    # without it the metallibs silently go stale after a header change or a
    # `git checkout`, and the app runs against shaders that no longer match the
    # host-side struct layouts.
    foreach(shader wavefront tonemapper skinning)
        add_custom_command(
            OUTPUT ${SHADER_OUTPUT_DIR}/${shader}.metallib
            COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_AIR_DIR}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_AIR_DIR}/module-cache
            COMMAND xcrun -sdk macosx metal ${METAL_BUILD_FLAGS}
                ${METAL_INCLUDE_FLAGS}
                -MD -MF ${SHADER_AIR_DIR}/${shader}.d
                -MT ${SHADER_OUTPUT_DIR}/${shader}.metallib
                -c ${CMAKE_CURRENT_SOURCE_DIR}/metal/${shader}.metal
                -o ${SHADER_AIR_DIR}/${shader}.air
            COMMAND xcrun -sdk macosx metallib
                ${SHADER_AIR_DIR}/${shader}.air
                -o ${SHADER_OUTPUT_DIR}/${shader}.metallib
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/metal/${shader}.metal
            DEPFILE ${SHADER_AIR_DIR}/${shader}.d
            COMMENT "Compiling ${shader}.metal"
            VERBATIM
        )
    endforeach()

    add_custom_target(strelka_shaders ALL
        DEPENDS
            ${SHADER_OUTPUT_DIR}/wavefront.metallib
            ${SHADER_OUTPUT_DIR}/tonemapper.metallib
            ${SHADER_OUTPUT_DIR}/skinning.metallib
            ${SHADER_OUTPUT_DIR}/fullScreen.metal
    )

    # Same layout as the build tree: metal/shaders next to the installed binaries.
    install(FILES
        ${SHADER_OUTPUT_DIR}/wavefront.metallib
        ${SHADER_OUTPUT_DIR}/tonemapper.metallib
        ${SHADER_OUTPUT_DIR}/skinning.metallib
        ${SHADER_OUTPUT_DIR}/fullScreen.metal
        DESTINATION metal/shaders
    )
endif()
