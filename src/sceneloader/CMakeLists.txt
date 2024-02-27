cmake_minimum_required(VERSION 3.20)

find_package(glm REQUIRED)
find_package(TinyGLTF REQUIRED)

# include(${ROOT_HOME}/cmake/StaticAnalyzers.cmake)

# Scene
set(SCENELOADER_SOURCES
        ${ROOT_HOME}/include/sceneloader/gltfloader.h
        ${ROOT_HOME}/src/sceneloader/gltfloader.cpp
    )
set(SCENELOADER_NAME sceneloader)
add_library(${SCENELOADER_NAME} OBJECT ${SCENELOADER_SOURCES})

target_include_directories(${SCENELOADER_NAME} PUBLIC ${ROOT_HOME}/include/)
target_include_directories(${SCENELOADER_NAME} PUBLIC ${ROOT_HOME}/include/sceneloader/)
target_include_directories(${SCENELOADER_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${SCENELOADER_NAME} PUBLIC logger scene glm::glm TinyGLTF::TinyGLTF)
