cmake_minimum_required(VERSION 3.22)

# --- strelka_material (INTERFACE / header-only) ---
# BSDF API headers consumed by shaders (CUDA/Metal) and C++ code (tests, render)

add_library(strelka_material INTERFACE)

target_include_directories(strelka_material
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
