cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 17)

find_package(USD REQUIRED HINTS ${USD_DIR} NAMES pxr)
message(STATUS "USD LIBRARY: ${USD_DIR}")

set(RUNNER_SOURCES
    ${ROOT_HOME}/src/hdRunnerMac/main.cpp
    ${ROOT_HOME}/src/hdRunner/SimpleRenderTask.h
    ${ROOT_HOME}/src/hdRunner/SimpleRenderTask.cpp)
set(RUNNER_NAME HdRunner)

add_executable(${RUNNER_NAME} ${RUNNER_SOURCES})
set_target_properties(
  ${RUNNER_NAME}
  PROPERTIES LINKER_LANGUAGE CXX
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED TRUE
             CXX_EXTENSIONS OFF
             INSTALL_RPATH_USE_LINK_PATH TRUE
             # The other libs in the plugin dir have no "lib" prefix, so let's
             # match this
             PREFIX "")

target_include_directories(${RUNNER_NAME} PUBLIC ${ROOT_HOME}/include/render)
target_include_directories(${RUNNER_NAME}
                           PUBLIC ${ROOT_HOME}/src/render/metal/shaders)
target_include_directories(${RUNNER_NAME}
                           PUBLIC ${ROOT_HOME}/external/cxxopts/include)
target_include_directories(${RUNNER_NAME} PUBLIC ${ROOT_HOME}/include/)

target_link_libraries(${RUNNER_NAME} PUBLIC "-framework Foundation")
target_link_libraries(${RUNNER_NAME} PUBLIC "-framework QuartzCore")
target_link_libraries(${RUNNER_NAME} PUBLIC "-framework Metal")
target_link_libraries(${RUNNER_NAME} PUBLIC "-framework MetalKit")
target_link_libraries(
  ${RUNNER_NAME}
  PUBLIC scene
         settings
         render
         ar
         cameraUtil
         hd
         hf
         hgi
         hio
         usd
         usdGeom
         usdImaging)

# Create symbolic links in final directory before building
add_custom_command(
  TARGET ${RUNNER_NAME}
  PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink
          "${ROOT_HOME}/src/render/metal/shaders" "${OUTPUT_DIRECTORY}/shaders")

# install_name_tool -change @rpath/Python3.framework/Versions/3.9/Python3
# @rpath/../Python3 ./work/Strelka/build/src/hdRunnerMac/HdRunner
