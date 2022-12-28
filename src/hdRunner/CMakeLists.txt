cmake_minimum_required(VERSION 3.20)

find_package(USD REQUIRED HINTS ${USD_DIR} NAMES pxr)
message(STATUS "USD LIBRARY: ${USD_DIR}")

set(HD_RUNNER_NAME HydraRunner)

# Application
set(HD_RUNNER_SOURCES
    ${ROOT_HOME}/src/hdRunner/main.cpp
    ${ROOT_HOME}/src/hdRunner/SimpleRenderTask.h
    ${ROOT_HOME}/src/hdRunner/SimpleRenderTask.cpp
)
add_executable(${HD_RUNNER_NAME} ${HD_RUNNER_SOURCES})
set_target_properties(
  ${HD_RUNNER_NAME}
  PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED TRUE
    CXX_EXTENSIONS OFF
)

# target_include_directories(${PROJECT_NAME} PUBLIC ${Vulkan_INCLUDE_DIR})
target_include_directories(${HD_RUNNER_NAME} PUBLIC ${ROOT_HOME}/external/glfw/include)
target_include_directories(${HD_RUNNER_NAME} PUBLIC ${ROOT_HOME}/external/cxxopts/include)
target_include_directories(${HD_RUNNER_NAME} PUBLIC ${ROOT_HOME}/include/)
target_include_directories(${HD_RUNNER_NAME} PUBLIC hd)
target_link_libraries(${HD_RUNNER_NAME} PUBLIC 
    # ${SETTINGSLIB_NAME}
    # ${RESMNGR_NAME}
    # ${SHADERLIB_NAME}
    # ${MATERIALLIB_NAME}
    # MaterialXCore
    # MaterialXFormat
    # MaterialXGenMdl
    scene
    settings
    display
    render
    # ${RENDERLIB_NAME}
    # ${UILIB_NAME}
    # ${TEXTURELIB_NAME}
    # ${TRACY_LIB}
    # ${Vulkan_LIBRARY}
    # "${CMAKE_CURRENT_SOURCE_DIR}/external/glfw/glfw3.lib"
    ${OPENGL_LIBRARIES}
    glad
    ar cameraUtil hd hf hgi hio usd usdGeom usdImaging    
)  # Link the executable to library (if it uses it).
