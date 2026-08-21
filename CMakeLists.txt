cmake_minimum_required(VERSION 3.22)
project(Strelka LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
include(${CMAKE_SOURCE_DIR}/cmake/CompilerWarnings.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/StaticAnalyzers.cmake)

# GLM 1.0.3's Conan recipe exports glm::glm only for the compiled library.
# Header-only builds (required: see conanfile.py glm/*:header_only) expose
# glm::glm-header-only. Alias so the rest of the tree can keep linking glm::glm.
#
# The FORCE macros must be compile definitions, not only #defines in
# glm_wrapper.hpp: GLM 1.0 turns a missing GLM_ENABLE_EXPERIMENTAL into a hard
# error on every gtx/ header, and several TUs include those after a raw
# <glm/glm.hpp>. A pre-1.0 include order that only warned now fails to compile.
find_package(glm REQUIRED)
if(NOT TARGET glm::glm)
    if(TARGET glm::glm-header-only)
        add_library(glm::glm ALIAS glm::glm-header-only)
    else()
        message(FATAL_ERROR "find_package(glm) succeeded but neither glm::glm nor glm::glm-header-only exists")
    endif()
endif()
add_compile_definitions(
    GLM_ENABLE_EXPERIMENTAL
    GLM_FORCE_SILENT_WARNINGS
    GLM_FORCE_CTOR_INIT
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
)

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
option(STRELKA_BUILD_TESTS  "Build unit tests" ON)

# --- MaterialX (third_party/materialx, pinned to v1.39.5) -------------------
#
# Only MaterialXCore and MaterialXFormat: the document model and its XML reader.
# Every shader generator is off, and that is a decision rather than thrift --
# see docs/material_api_proposal.md's successor and the plan behind
# openpbr/openpbr_shim.h. MaterialXGenMsl exists and MaterialXGenCuda does not,
# so generating shader code from a .mtlx would make the feature Metal-only,
# which is exactly the constraint that ruled codegen out. What the loader needs
# instead is the parsed document, which Core and Format alone provide.
#
# EXCLUDE_FROM_ALL so MaterialX's own install rules and samples stay out of this
# build; the two libraries are still built because the loader links them.
set(MATERIALX_BUILD_GEN_GLSL      OFF CACHE BOOL "" FORCE)
set(MATERIALX_BUILD_GEN_OSL       OFF CACHE BOOL "" FORCE)
set(MATERIALX_BUILD_GEN_MDL       OFF CACHE BOOL "" FORCE)
set(MATERIALX_BUILD_GEN_MSL       OFF CACHE BOOL "" FORCE)
set(MATERIALX_BUILD_GEN_SLANG     OFF CACHE BOOL "" FORCE)
set(MATERIALX_BUILD_RENDER        OFF CACHE BOOL "" FORCE)
set(MATERIALX_BUILD_TESTS         OFF CACHE BOOL "" FORCE)
set(MATERIALX_BUILD_PYTHON        OFF CACHE BOOL "" FORCE)
set(MATERIALX_BUILD_VIEWER        OFF CACHE BOOL "" FORCE)
set(MATERIALX_BUILD_GRAPH_EDITOR  OFF CACHE BOOL "" FORCE)
set(MATERIALX_BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)
set(MATERIALX_INSTALL_RESOURCES   OFF CACHE BOOL "" FORCE)
# A cache shared with every other MaterialX build on the machine is the opposite
# of hermetic, and the Conan recipe turns it off for the same reason.
set(MATERIALX_BUILD_USE_CCACHE    OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/materialx EXCLUDE_FROM_ALL)

# The data library (stdlib, pbrlib, bxdf/...) is needed at *runtime*, not just
# at build time: mx::loadLibraries resolves every nodedef and every unspecified
# input default out of it, so a document that says <open_pbr_surface> means
# nothing without it. Copied next to the binaries, exactly like the metallibs,
# and found through oka::resolveResourcePath.
#
# This is also why the conan-center materialx package is not used: its recipe
# deletes the libraries/ directory from the package.
set(MATERIALX_RUNTIME_LIBRARIES ${OUTPUT_DIRECTORY}/materialx/libraries)
add_custom_target(strelka_materialx_libraries ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${MATERIALX_RUNTIME_LIBRARIES}
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/third_party/materialx/libraries ${MATERIALX_RUNTIME_LIBRARIES}
    COMMENT "Copying the MaterialX data library"
    VERBATIM)
install(DIRECTORY ${MATERIALX_RUNTIME_LIBRARIES} DESTINATION materialx)

# Core libraries (always built)
add_subdirectory(src/foundation)
add_subdirectory(src/scene)
add_subdirectory(src/material)
add_subdirectory(src/sceneloader)

# Render + Shaders require CUDA/OptiX on Linux/Win, Metal on macOS
if(STRELKA_BUILD_EDITOR OR STRELKA_BUILD_CLI OR STRELKA_BUILD_TESTS)
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

# Tests
if(STRELKA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Vendored sources are not ours to analyze; see cmake/StaticAnalyzers.cmake.
strelka_skip_clang_tidy_on_vendored_targets()

# macOS package layout: binaries and metallibs share the install prefix root so
# resolveResourcePath("metal/shaders/...") keeps working without path changes.
if(APPLE)
    include(GNUInstallDirs)
    install(FILES "${CMAKE_SOURCE_DIR}/LICENSE" DESTINATION .)
    install(FILES "${CMAKE_SOURCE_DIR}/README.md" DESTINATION .)
endif()
