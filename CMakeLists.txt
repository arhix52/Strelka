cmake_minimum_required(VERSION 3.20)
project(Strelka LANGUAGES CXX)
# enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

if(WIN32)
  add_compile_definitions(NOMINMAX)
  add_compile_definitions(_USE_MATH_DEFINES)
  add_compile_definitions(_ALLOW_ITERATOR_DEBUG_LEVEL_MISMATCH)
endif()

# find_package(OpenGL REQUIRED)
# # Find at least a 5.0 version of CUDA.
# find_package(CUDA 5.0 REQUIRED)
# message(STATUS "CUDA Include DIR: ${CUDA_INCLUDE_DIRS}")
# include_directories(${CUDA_INCLUDE_DIRS})
# # Search for the OptiX libraries and include files.
# set(OptiX_INSTALL_DIR $ENV{OPTIX_DIR})
# message(STATUS "OptiX SDK DIR: ${OptiX_INSTALL_DIR}")
# # Search for the OptiX libraries and include files.
# find_package(OptiX REQUIRED)
# message(STATUS "OptiX Include DIR: ${OptiX_INCLUDE}")

# include_directories(${OptiX_INCLUDE})
# include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Hack to get intellisense working for CUDA includes
if(MSVC)
  set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES
      "$(VC_IncludePath);$(WindowsSDK_IncludePath);${OptiX_INCLUDE}")
endif()

if(WIN32)
  add_subdirectory(external/glad)
  set_target_properties(glad PROPERTIES LINKER_LANGUAGE CXX)
endif()

set(ROOT_HOME ${CMAKE_CURRENT_LIST_DIR})
set(OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(src/Settings)
add_subdirectory(src/render)
add_subdirectory(src/hdStrelka)
add_subdirectory(src/scene)
add_subdirectory(src/materialmanager)
# add_subdirectory(src/app)

if(WIN32)
  add_subdirectory(src/display)
  add_subdirectory(src/hdRunner)
endif()
if(APPLE)
  add_subdirectory(src/hdRunnerMac)
endif()

add_subdirectory(tests)
