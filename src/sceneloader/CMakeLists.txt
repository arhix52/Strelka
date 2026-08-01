cmake_minimum_required(VERSION 3.22)

find_package(glm REQUIRED)
find_package(TinyGLTF REQUIRED)
find_package(nlohmann_json REQUIRED)

set(SCENELOADER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/strelka/sceneloader/gltfloader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gltfloader.cpp
)

add_library(strelka_sceneloader STATIC ${SCENELOADER_SOURCES})

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
        nlohmann_json::nlohmann_json
)
