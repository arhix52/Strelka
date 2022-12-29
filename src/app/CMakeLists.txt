cmake_minimum_required(VERSION 3.16)

find_package(USD REQUIRED HINTS ${USD_DIR} NAMES pxr)
message(STATUS "USD LIBRARY: ${USD_DIR}")
find_package( OpenGL REQUIRED )

# Application
set(RUNNER_SOURCES
    ${ROOT_HOME}/src/app/main.cpp
    # ${ROOT_HOME}/src/hdRunner/SimpleRenderTask.h
    # ${ROOT_HOME}/src/hdRunner/SimpleRenderTask.cpp
)
set(RUNNER_NAME runner)

add_executable(${RUNNER_NAME} ${RUNNER_SOURCES})
set_target_properties(
  ${RUNNER_NAME}
  PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED TRUE
    CXX_EXTENSIONS OFF
)

if( WIN32 )
  target_compile_definitions( ${RUNNER_NAME} PUBLIC GLAD_GLAPI_EXPORT )
endif()

# target_include_directories(${PROJECT_NAME} PUBLIC ${Vulkan_INCLUDE_DIR})
target_include_directories(${RUNNER_NAME} PUBLIC ${ROOT_HOME}/external/glfw/include)
target_include_directories(${RUNNER_NAME} PUBLIC ${ROOT_HOME}/external/cxxopts/include)
target_include_directories(${RUNNER_NAME} PUBLIC ${ROOT_HOME}/include/)
target_include_directories(${RUNNER_NAME} PUBLIC ${ROOT_HOME}/include/display)
# target_include_directories(${RUNNER_NAME} PUBLIC hd)
target_link_libraries(${RUNNER_NAME} PUBLIC 
    # ${DISPLAYLIB_NAME}
    # ${SCENELIB_NAME}
    # ${RENDERLIB_NAME}
    ${ROOT_HOME}/external/glfw/glfw3.lib
    # ${OPENGL_LIBRARIES}
    scene
    display
    render
    # PRIVATE
    glad
    # ar cameraUtil hd hf hgi hio usd usdGeom usdImaging    
)  # Link the executable to library (if it uses it).
