cmake_minimum_required(VERSION 3.22)

# --- strelka_material (INTERFACE / header-only) ---
# BSDF API headers consumed by shaders (CUDA/Metal) and C++ code (tests, render)

add_library(strelka_material INTERFACE)

target_include_directories(strelka_material
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Adobe's OpenPBR 1.1.1 BSDF (third_party/openpbr_bsdf), reached only through
# openpbr/openpbr_shim.h -- see that header for why it is never included direct.
# Carried on this target rather than -I'd at each use site because four
# different compilers consume it: nvcc, the Metal compiler, the OptiX host
# translation units and the doctest suite.
#
# SYSTEM so its warnings and clang-tidy findings are not ours: it is vendored
# verbatim, and its MSL path alone raises 56 -Wc++17-extensions warnings that
# nobody here can act on.
target_include_directories(strelka_material
    SYSTEM INTERFACE
        ${CMAKE_SOURCE_DIR}/third_party/openpbr_bsdf
)
