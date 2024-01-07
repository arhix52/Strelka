cmake_minimum_required(VERSION 3.20)

# packages
find_package(cxxopts REQUIRED)
include(FindVulkan)
find_package(Vulkan REQUIRED)

if(Vulkan_FOUND)
  set(XR_USE_GRAPHICS_API_VULKAN TRUE)
add_definitions(-DXR_USE_GRAPHICS_API_VULKAN)
  message(STATUS "Enabling Vulkan support")
elseif(BUILD_ALL_EXTENSIONS)
  message(FATAL_ERROR "Vulkan headers not found")
endif()

set(XR_RUNNER_NAME XrRunner)

include(${ROOT_HOME}/external/openxr_loader_windows-1.0.32/OpenXRConfig.cmake)

# Application
set(XR_RUNNER_SOURCES
    ${ROOT_HOME}/src/XrRunner/OpenXRProgram.h
    ${ROOT_HOME}/src/XrRunner/OpenXRProgram.cpp
    ${ROOT_HOME}/src/XrRunner/main.cpp )
add_executable(${XR_RUNNER_NAME} ${XR_RUNNER_SOURCES})

set_target_properties(
  ${XR_RUNNER_NAME}
  PROPERTIES LINKER_LANGUAGE CXX
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED TRUE
             CXX_EXTENSIONS OFF)

target_include_directories(${XR_RUNNER_NAME} PUBLIC ${ROOT_HOME}/include/)
target_include_directories(${XR_RUNNER_NAME} PUBLIC ${ROOT_HOME}/src/XrRunner/)
target_include_directories(${XR_RUNNER_NAME} PRIVATE ${Vulkan_INCLUDE_DIRS})

target_link_libraries(
  ${XR_RUNNER_NAME}
  PUBLIC cxxopts::cxxopts
  scene
  settings
  render
  logger
  OpenXR::openxr_loader
  ${Vulkan_LIBRARIES}
)
