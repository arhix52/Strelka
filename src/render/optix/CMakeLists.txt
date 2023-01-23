cmake_minimum_required(VERSION 3.20)

enable_language(CUDA)

# Find at least a 5.0 version of CUDA.
find_package(CUDA 5.0 REQUIRED)
message(STATUS "CUDA Toolkit Root: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA Include DIR: ${CUDA_INCLUDE_DIRS}")
# Search for the OptiX libraries and include files.
set(OptiX_INSTALL_DIR $ENV{OPTIX_DIR})
message(STATUS "OptiX SDK DIR: ${OptiX_INSTALL_DIR}")
# Search for the OptiX libraries and include files.
find_package(OptiX REQUIRED)
message(STATUS "OptiX Include DIR: ${OptiX_INCLUDE}")
include_directories(${OptiX_INCLUDE})

set(RENDERLIB_NAME render)

# Render lib
set(RENDER_SOURCES
    ${ROOT_HOME}/include/render/optix/buffer.h
    ${ROOT_HOME}/include/render/optix/Camera.h
    ${ROOT_HOME}/include/render/optix/OptiXRender.h
    ${ROOT_HOME}/include/render/optix/materials.h
    ${ROOT_HOME}/include/render/optix/Lights.h

    ${ROOT_HOME}/src/render/optix/buffer.cpp
    ${ROOT_HOME}/src/render/optix/Camera.cpp
    ${ROOT_HOME}/src/render/optix/OptiXRender.cpp
)
set(RENDER_OPTIX_SOURCES
    ${ROOT_HOME}/include/render/optix/OptixRenderParams.h
    ${ROOT_HOME}/src/render/optix/OptixRender.cu
)
include_directories(${ROOT_HOME}/external/glm)
include_directories(${ROOT_HOME}/include/render/optix/)
# Create the rules to build the PTX and/or OPTIX files.
CUDA_WRAP_SRCS( ${RENDERLIB_NAME} OPTIXIR generated_files2 ${RENDER_OPTIX_SOURCES} ${cmake_options} OPTIONS ${options} )
list(APPEND generated_files ${generated_files2})

# set_source_files_properties(${RENDER_SOURCES} PROPERTIES LANGUAGE CUDA)

add_library(${RENDERLIB_NAME} STATIC 
    ${RENDER_SOURCES}
    ${generated_files}
)

target_include_directories(${RENDERLIB_NAME} PUBLIC ${ROOT_HOME}/include/)
target_include_directories(${RENDERLIB_NAME} PUBLIC ${ROOT_HOME}/include/render)
target_include_directories(${RENDERLIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${RENDERLIB_NAME} PUBLIC ${ROOT_HOME}/external/glm)

target_include_directories(${RENDERLIB_NAME} PUBLIC ${OptiX_INCLUDE})
target_include_directories(${RENDERLIB_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})

add_custom_command(TARGET ${RENDERLIB_NAME} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
        "${OUTPUT_DIRECTORY}/optix"
        )
add_custom_command(TARGET ${RENDERLIB_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${generated_files}
        "${OUTPUT_DIRECTORY}/optix"
        )

target_link_libraries(${RENDERLIB_NAME} PUBLIC 
    ${CUDA_LIBRARIES}
    glad
    scene
    materialmanager
)
