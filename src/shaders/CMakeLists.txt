cmake_minimum_required(VERSION 3.22)

set(SHADER_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)

if(WIN32 OR LINUX)
    find_package(CUDA REQUIRED)
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
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/optix)

    # CUDA flags for OptiX shaders (targeting SM 89 / Ada Lovelace: RTX 4090+)
    set(OPTIX_NVCC_FLAGS --gpu-architecture=compute_89 --use_fast_math --relocatable-device-code=true)

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

    # Utility CUDA kernels (tonemappers, skinning, env CDF) — compiled to object files
    cuda_compile(compiled_tonemap_files
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/postprocessing/Tonemappers.cu
    )
    cuda_compile(compiled_skinning_files
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/skinning/skinning.cu
    )
    cuda_compile(compiled_env_cdf_files
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/env_cdf.cu
    )

    add_library(strelka_shaders STATIC
        ${generated_optix_files}
        ${generated_hit_files}
        ${compiled_tonemap_files}
        ${compiled_skinning_files}
        ${compiled_env_cdf_files}
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/postprocessing/Tonemappers.h
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/skinning/skinning.h
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/env_cdf.h
    )
    set_target_properties(strelka_shaders PROPERTIES LINKER_LANGUAGE CXX)

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
        -I ${SHADER_COMMON_DIR}
        -I ${CMAKE_CURRENT_SOURCE_DIR}/metal
    )

    # -gline-tables-only/-frecord-sources keep Metal frame captures usable.
    set(METAL_BUILD_FLAGS -Wall -std=metal3.0)
    # Two kernels that do the same arithmetic in a different order (the megakernel
    # and the wavefront tracer) fuse multiply-adds differently, which leaves a
    # ~1e-7 relative difference in most pixels and, through chaotic paths, a
    # visible one in a handful. Turning contraction off makes the two bit-identical
    # and is the only way to tell an arithmetic difference from a logic bug.
    option(STRELKA_METAL_STRICT_FP "Disable Metal FMA contraction (for A/B image comparison)" OFF)
    if(STRELKA_METAL_STRICT_FP)
        list(APPEND METAL_BUILD_FLAGS -ffp-contract=off)
    endif()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND METAL_BUILD_FLAGS -gline-tables-only -frecord-sources)
    endif()

    set(SHADER_OUTPUT_DIR ${OUTPUT_DIRECTORY}/metal/shaders)
    set(SHADER_AIR_DIR ${CMAKE_CURRENT_BINARY_DIR}/air)

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
endif()
