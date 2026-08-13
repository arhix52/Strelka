cmake_minimum_required(VERSION 3.22)

find_package(glm REQUIRED)

set(SCENE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/strelka/scene/scene.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/strelka/scene/camera.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/strelka/scene/glm_wrapper.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/scene.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/camera.cpp
)

add_library(strelka_scene STATIC ${SCENE_SOURCES})
strelka_apply_warnings(strelka_scene)

target_include_directories(strelka_scene
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/shaders/common
)

target_link_libraries(strelka_scene
    PUBLIC
        strelka_foundation
        strelka_material
        glm::glm
)
