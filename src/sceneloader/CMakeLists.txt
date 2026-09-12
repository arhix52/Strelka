cmake_minimum_required(VERSION 3.22)

find_package(glm REQUIRED)
find_package(TinyGLTF REQUIRED)
find_package(nlohmann_json REQUIRED)

set(SCENELOADER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/strelka/sceneloader/gltfloader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/strelka/sceneloader/sceneserializer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/strelka/sceneloader/iesloader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/strelka/sceneloader/light_json.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/strelka/sceneloader/material_sidecar.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/strelka/sceneloader/materialx_loader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/strelka/sceneloader/static_blas_partition.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/materialx_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gltfloader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/static_blas_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sceneserializer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/iesloader.cpp
)

add_library(strelka_sceneloader STATIC ${SCENELOADER_SOURCES})
strelka_apply_warnings(strelka_sceneloader)

target_include_directories(strelka_sceneloader
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(strelka_sceneloader
    PUBLIC
        strelka_foundation
        strelka_scene
        glm::glm
        TinyGLTF::TinyGLTF
        # Core is the document model, Format its XML reader. No generator: see
        # the note beside add_subdirectory(third_party/materialx) in the root.
        MaterialXCore
        MaterialXFormat
        nlohmann_json::nlohmann_json
)
