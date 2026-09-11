cmake_minimum_required(VERSION 3.22)

set(SHADER_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)

if(WIN32 OR LINUX)
    find_package(CUDA REQUIRED)

    # nvcc through ccache, when there is one.
    #
    # The device half of an edit-build cycle is one nvcc invocation and 14.6 s of
    # it: OptixRender_closest_hit.cu is the megakernel and its OPTIXIR is 14.9 MB.
    # An A/B measurement pays that again for the flip back, and ccache does cache
    # an --optix-ir compilation -- measured on a two-line probe, 0.183 s cold and
    # 0.003 s warm -- so the flip back costs nothing.
    #
    # A wrapper script rather than a launcher variable: this file drives nvcc
    # through FindCUDA's run_nvcc.cmake, which executes CUDA_NVCC_EXECUTABLE as a
    # single program and has no notion of a launcher. UNIX only for the same
    # reason -- a shell script is not what Windows would execute.
    if(UNIX AND STRELKA_CCACHE AND NOT CUDA_NVCC_EXECUTABLE MATCHES "ccache-nvcc$")
        set(STRELKA_CCACHE_NVCC "${CMAKE_BINARY_DIR}/ccache-nvcc")
        file(WRITE "${STRELKA_CCACHE_NVCC}"
             "#!/bin/sh\nexec \"${STRELKA_CCACHE}\" \"${CUDA_NVCC_EXECUTABLE}\" \"$@\"\n")
        file(CHMOD "${STRELKA_CCACHE_NVCC}" PERMISSIONS
             OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
        set(CUDA_NVCC_EXECUTABLE "${STRELKA_CCACHE_NVCC}" CACHE FILEPATH "" FORCE)
        message(STATUS "nvcc through ccache: ${STRELKA_CCACHE_NVCC}")
    endif()

    set(OptiX_INSTALL_DIR $ENV{OPTIX_DIR})
    find_package(OptiX REQUIRED)

    include_directories(${OptiX_INCLUDE})
    include_directories(${CMAKE_SOURCE_DIR}/src/render/include/strelka/render)
    include_directories(${CMAKE_SOURCE_DIR}/src/render/optix)
    include_directories(${CMAKE_SOURCE_DIR}/src/render)
    include_directories(${CMAKE_SOURCE_DIR}/src/render/include)
    include_directories(${CMAKE_SOURCE_DIR}/third_party)
    include_directories(${SHADER_COMMON_DIR})
    include_directories(${CMAKE_SOURCE_DIR}/src/material/include)
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

    include_directories(${OPENPBR_DEVICE_INCLUDE_DIR})
    include_directories(${CMAKE_SOURCE_DIR}/third_party/openpbr_bsdf)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/optix)

    # CUDA flags for OptiX shaders (targeting SM 89 / Ada Lovelace: RTX 4090+)
    #
    # --expt-relaxed-constexpr: openpbr spells its small scalar helpers
    # OPENPBR_CONSTEXPR_FUNCTION, which the CUDA interop expands to plain
    # `static constexpr` -- a __host__ function that its __device__ callers then
    # cannot call. The flag is nvcc's own answer to exactly that, and it is what
    # the diagnostic suggests; the alternative is 30 more rewrite rules in
    # openpbr_device_headers.py for functions whose bodies are one line each.
    # -lineinfo is what lets Nsight Compute attribute a stall to a source line
    # rather than to an address. It adds a debug section and changes no code:
    # measured on kids_room, 22.2 ms/sample either way. Without it the PC
    # samples land on SASS and the question "which line stalls" has no answer.
    # -c is here for ccache and for nothing else. FindCUDA spells the OPTIXIR
    # rule `nvcc file.cu -optix-ir -o out`, with no -c, and ccache reads a
    # command line without one as a link step and refuses to cache it ("No -c
    # option found" in its log). The device half of an edit-build cycle is one
    # nvcc invocation and 14.6 s of it -- OptixRender_closest_hit.cu is the
    # megakernel, its OPTIXIR is 14.9 MB -- so an A/B measurement paid that twice
    # and a revert paid it again.
    #
    # It changes nothing about the output: the OPTIXIR compiled with it is byte
    # for byte the one compiled without, checked on this file at equal output
    # paths. --optix-ir is already a device-only compilation; -c only says so.
    set(OPTIX_NVCC_FLAGS --gpu-architecture=compute_89 --use_fast_math --relocatable-device-code=true
        --expt-relaxed-constexpr --generate-line-info -c)

    # NDEBUG outside a Debug build, which is what compiles openpbr's ~40
    # OPENPBR_ASSERTs out of device code. They are preconditions on quantities a
    # path tracer computes in floating point -- "this normal is unit to 1e-6",
    # "this random number is in [0,1)" -- and the CUDA interop maps them to
    # assert(). A device assert does not warn and does not skip the sample: it
    # aborts the entire launch with cudaErrorAssert and leaves the context
    # unusable, so one ray in a hundred million ends the render.
    #
    # Kept live in Debug on purpose. They are good asserts: the first one to fire
    # here was real (see openpbr_bridge.h's note on the geometry basis), and a
    # Debug run is how the next one gets found.
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND OPTIX_NVCC_FLAGS -DNDEBUG)
    endif()

    # Ray generation + miss (compiled to OPTIXIR)
    cuda_wrap_srcs(strelka_shaders OPTIXIR generated_optix_files
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/OptixRender.cu
        ${cmake_options}
        OPTIONS ${OPTIX_NVCC_FLAGS} ${options}
    )

    # Closest-hit shader (compiled to OPTIXIR, loaded as separate module)
    cuda_wrap_srcs(strelka_shaders OPTIXIR generated_hit_files
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/OptixRender_closest_hit.cu
        ${cmake_options}
        OPTIONS ${OPTIX_NVCC_FLAGS} ${options}
    )

    # Utility CUDA kernels (tonemappers, skinning) — compiled to object files
    cuda_compile(compiled_tonemap_files
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/postprocessing/Tonemappers.cu
    )
    cuda_compile(compiled_skinning_files
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/skinning/skinning.cu
    )
    # env_cdf.cu is gone: the environment is sampled from the alias table the
    # Metal backend already uses, built on the host.
    cuda_compile(compiled_denoise_guide_files
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/postprocessing/DenoiseGuides.cu
    )
    # The radiance cache's resolve pass. A plain CUDA kernel, not OPTIXIR: it
    # traces nothing and runs once per frame between launches.
    cuda_compile(compiled_sharc_files
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/sharc_resolve.cu
    )

    add_library(strelka_shaders STATIC
        ${generated_optix_files}
        ${generated_hit_files}
        ${compiled_tonemap_files}
        ${compiled_skinning_files}
        ${compiled_denoise_guide_files}
        ${compiled_sharc_files}
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/postprocessing/Tonemappers.h
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/postprocessing/DenoiseGuides.h
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/postprocessing/Guides.h
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/skinning/skinning.h
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/sharc.h
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/sharc_grid.h
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/sharc_resolve.h
    )
    set_target_properties(strelka_shaders PROPERTIES LINKER_LANGUAGE CXX)
    # The rewritten OpenPBR headers are an input to every OPTIXIR compile above,
    # but cuda_wrap_srcs cannot be told about a generated include directory, so
    # the ordering has to be stated at target level.
    add_dependencies(strelka_shaders openpbr_device_headers)

    target_include_directories(strelka_shaders
        PUBLIC
            ${SHADER_COMMON_DIR}
            ${CMAKE_SOURCE_DIR}/src/material/include
    )

    # Copy OPTIXIR files to output directory using file-level dependencies
    # (POST_BUILD can be skipped by ninja when the target doesn't need relinking)
    set(OPTIXIR_OUTPUT_DIR "${OUTPUT_DIRECTORY}/optix")
    set(copied_optixir_files "")
    foreach(optixir_file ${generated_optix_files} ${generated_hit_files})
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
    add_dependencies(copy_optixir strelka_shaders)

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
