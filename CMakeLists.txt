cmake_minimum_required(VERSION 3.24)
project(Strelka LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(UNIX AND NOT APPLE)
  set(LINUX TRUE)
endif()

if(WIN32)
  add_compile_definitions(NOMINMAX)
  add_compile_definitions(_USE_MATH_DEFINES)
  add_compile_definitions(_ALLOW_ITERATOR_DEBUG_LEVEL_MISMATCH)
endif()

if(WIN32 OR LINUX)
  add_subdirectory(external/glad)
  set_target_properties(glad PROPERTIES LINKER_LANGUAGE CXX)
endif()

set(ROOT_HOME ${CMAKE_CURRENT_LIST_DIR})
set(OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(src/log)
add_subdirectory(src/settings)
add_subdirectory(src/render)
add_subdirectory(src/HdStrelka)
add_subdirectory(src/scene)
# add_subdirectory(src/materialmanager)
add_subdirectory(src/display)
add_subdirectory(src/hdRunner)

if(WIN32 OR LINUX)
  add_subdirectory(tests)
endif()
