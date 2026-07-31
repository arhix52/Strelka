cmake_minimum_required(VERSION 3.22)

find_package(glm REQUIRED)
find_package(stb REQUIRED)

set(RENDER_SOURCES_COMMON
    ${CMAKE_CURRENT_SOURCE_DIR}/include/strelka/render/render.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/strelka/render/buffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/strelka/render/common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/strelka/render/Camera.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/strelka/render/materials.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/render.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Camera.cpp
)

if(WIN32 OR LINUX)
    find_package(CUDAToolkit REQUIRED)
    set(OptiX_INSTALL_DIR $ENV{OPTIX_DIR})
    find_package(OptiX REQUIRED)

    set(RENDER_SOURCES
        ${RENDER_SOURCES_COMMON}
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/OptixBuffer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/OptixBuffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/OptixRender.h
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/OptixRender.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/OptixRenderParams.h
        ${CMAKE_CURRENT_SOURCE_DIR}/optix/cuda_checks.h
    )

    add_library(strelka_render STATIC ${RENDER_SOURCES})

    target_include_directories(strelka_render
        PUBLIC
            ${OptiX_INCLUDE}
            ${CUDAToolkit_INCLUDE_DIRS}
            ${CMAKE_SOURCE_DIR}/third_party
            ${CMAKE_SOURCE_DIR}/src/shaders/common
            ${CMAKE_SOURCE_DIR}/src/shaders/optix
    )

    find_package(tinyexr REQUIRED)

    target_link_libraries(strelka_render
        PUBLIC
            CUDA::cudart
            CUDA::cuda_driver
            strelka_foundation
            strelka_scene
            strelka_material
            glm::glm
            stb::stb
            tinyexr::tinyexr
    )
endif()

if(APPLE)
    find_package(tinyexr REQUIRED)

    set(RENDER_SOURCES
        ${RENDER_SOURCES_COMMON}
        ${CMAKE_CURRENT_SOURCE_DIR}/metal/MetalRender.h
        ${CMAKE_CURRENT_SOURCE_DIR}/metal/MetalRender.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/metal/MetalBuffer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/metal/MetalBuffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/metal/Metal4Context.h
        ${CMAKE_CURRENT_SOURCE_DIR}/metal/Metal4Context.mm
    )

    add_library(strelka_render STATIC ${RENDER_SOURCES})

    target_include_directories(strelka_render
        PRIVATE
            ${ROOT_HOME}/third_party/metal_cpp/
            ${ROOT_HOME}/third_party/metal_cpp_extensions/
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src/shaders/common
            ${CMAKE_SOURCE_DIR}/src/shaders/metal
    )

    target_link_libraries(strelka_render
        PRIVATE
            "-framework Foundation"
            "-framework QuartzCore"
            "-framework Metal"
            "-framework MetalKit"
    )
    target_link_libraries(strelka_render
        PUBLIC
            strelka_foundation
            strelka_scene
            strelka_material
            glm::glm
            stb::stb
            tinyexr::tinyexr
    )
endif()

target_include_directories(strelka_render
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)
