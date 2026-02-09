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
    set(METAL_INCLUDE_FLAGS
        -I ${CMAKE_SOURCE_DIR}/src/material/include
        -I ${SHADER_COMMON_DIR}
        -I ${CMAKE_CURRENT_SOURCE_DIR}/metal
    )

    set(SHADER_OUTPUT_DIR ${OUTPUT_DIRECTORY}/metal/shaders)

    # Copy fullScreen.metal (no compilation needed, loaded at runtime)
    add_custom_command(
        OUTPUT ${SHADER_OUTPUT_DIR}/fullScreen.metal
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/metal/fullScreen.metal
            ${SHADER_OUTPUT_DIR}/fullScreen.metal
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/metal/fullScreen.metal
    )

    # Compile Metal shaders
    foreach(shader pathtrace tonemapper skinning bdpt_camera_subpath bdpt_light_subpath bdpt_connect vcm_clear vcm_hash_grid_build vcm_merge)
        add_custom_command(
            OUTPUT ${SHADER_OUTPUT_DIR}/${shader}.metallib
            COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
            COMMAND xcrun -sdk macosx metal -Wall -std=metal3.0
                ${METAL_INCLUDE_FLAGS}
                -c ${CMAKE_CURRENT_SOURCE_DIR}/metal/${shader}.metal
                -o ${CMAKE_BINARY_DIR}/${shader}.air
            COMMAND xcrun -sdk macosx metallib
                ${CMAKE_BINARY_DIR}/${shader}.air
                -o ${SHADER_OUTPUT_DIR}/${shader}.metallib
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/metal/${shader}.metal
            COMMENT "Compiling ${shader}.metal"
        )
    endforeach()

    add_custom_target(strelka_shaders ALL
        DEPENDS
            ${SHADER_OUTPUT_DIR}/pathtrace.metallib
            ${SHADER_OUTPUT_DIR}/tonemapper.metallib
            ${SHADER_OUTPUT_DIR}/skinning.metallib
            ${SHADER_OUTPUT_DIR}/bdpt_camera_subpath.metallib
            ${SHADER_OUTPUT_DIR}/bdpt_light_subpath.metallib
            ${SHADER_OUTPUT_DIR}/bdpt_connect.metallib
            ${SHADER_OUTPUT_DIR}/vcm_clear.metallib
            ${SHADER_OUTPUT_DIR}/vcm_hash_grid_build.metallib
            ${SHADER_OUTPUT_DIR}/vcm_merge.metallib
            ${SHADER_OUTPUT_DIR}/fullScreen.metal
    )
endif()
