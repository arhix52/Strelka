cmake_minimum_required(VERSION 3.20)

find_package(cxxopts REQUIRED)
find_package(USD REQUIRED HINTS ${USD_DIR} NAMES pxr)
message(STATUS "USD LIBRARY: ${USD_DIR}")

set(HD_RUNNER_NAME HydraRunner)

# WAR: https://github.com/PixarAnimationStudios/OpenUSD/issues/2634
add_compile_definitions(BOOST_NO_CXX98_FUNCTION_BASE)

# Application
set(HD_RUNNER_SOURCES
    ${ROOT_HOME}/src/hdRunner/main.cpp
    ${ROOT_HOME}/src/hdRunner/SimpleRenderTask.h
    ${ROOT_HOME}/src/hdRunner/SimpleRenderTask.cpp)
add_executable(${HD_RUNNER_NAME} ${HD_RUNNER_SOURCES})
set_target_properties(
  ${HD_RUNNER_NAME}
  PROPERTIES LINKER_LANGUAGE CXX
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED TRUE
             CXX_EXTENSIONS OFF)

add_custom_command(
  TARGET ${HD_RUNNER_NAME}
  PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "${OUTPUT_DIRECTORY}/data/materials/mtlx")
add_custom_command(
  TARGET ${HD_RUNNER_NAME}
  PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "${OUTPUT_DIRECTORY}/data/materials/mdl")

target_include_directories(${HD_RUNNER_NAME} PUBLIC ${ROOT_HOME}/include/)
target_include_directories(${HD_RUNNER_NAME} PUBLIC hd)
target_link_libraries(
  ${HD_RUNNER_NAME}
  PUBLIC cxxopts::cxxopts
         scene
         settings
         display
         render
         logger
         ar
         cameraUtil
         hd
         hf
         hgi
         hio
         usd
         usdGeom
         usdImaging) # Link the executable to library (if it uses it).
if(WIN32)
  # Add the linker options to suppress the MSVCRT conflict warning
  target_link_options(${HD_RUNNER_NAME} PRIVATE "/NODEFAULTLIB:MSVCRT")
  target_link_options(${HD_RUNNER_NAME} PRIVATE "/NODEFAULTLIB:LIBCMT")
  
  target_link_libraries(${HD_RUNNER_NAME} PUBLIC ${OPENGL_LIBRARIES})
endif()
