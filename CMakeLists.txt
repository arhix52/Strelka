cmake_minimum_required(VERSION 3.22)
project(Strelka LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Platform detection
if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if(WIN32)
    add_compile_definitions(NOMINMAX)
    add_compile_definitions(_USE_MATH_DEFINES)
    add_compile_definitions(_ALLOW_ITERATOR_DEBUG_LEVEL_MISMATCH)
endif()

set(ROOT_HOME ${CMAKE_CURRENT_LIST_DIR})
set(OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Executables must sit next to the runtime asset trees they load
# (${OUTPUT_DIRECTORY}/metal/shaders, ${OUTPUT_DIRECTORY}/optix). Without this
# the editor lands in <build>/src/editor/ and only starts if the working
# directory happens to be the build root.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIRECTORY})
foreach(cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg} ${OUTPUT_DIRECTORY})
endforeach()

# Build options
option(STRELKA_BUILD_EDITOR "Build interactive editor application" ON)
option(STRELKA_BUILD_CLI    "Build headless CLI renderer" ON)
option(STRELKA_BUILD_HYDRA  "Build Hydra render delegate (requires OpenUSD)" OFF)
option(STRELKA_BUILD_TESTS  "Build unit tests" ON)

# Core libraries (always built)
add_subdirectory(src/foundation)
add_subdirectory(src/scene)
add_subdirectory(src/material)
add_subdirectory(src/sceneloader)

# Render + Shaders require CUDA/OptiX on Linux/Win, Metal on macOS
# Gated behind editor, CLI, or hydra since they require a render backend
if(STRELKA_BUILD_EDITOR OR STRELKA_BUILD_CLI OR STRELKA_BUILD_HYDRA)
    add_subdirectory(src/render)
    add_subdirectory(src/shaders)
endif()

# Editor (optional, default ON)
if(STRELKA_BUILD_EDITOR)
    add_subdirectory(src/display)
    add_subdirectory(src/editor)
endif()

# Headless CLI renderer (optional, default ON)
if(STRELKA_BUILD_CLI)
    add_subdirectory(src/cli)
endif()

# Hydra delegate (optional, default OFF)
if(STRELKA_BUILD_HYDRA)
    find_package(pxr REQUIRED HINTS ${USD_DIR})
    add_subdirectory(src/plugins/hydra)
endif()

# Tests
if(STRELKA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
