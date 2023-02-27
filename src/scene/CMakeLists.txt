cmake_minimum_required(VERSION 3.20)

include(${ROOT_HOME}/cmake/StaticAnalyzers.cmake)

# Scene
set(SCENE_SOURCES
        ${ROOT_HOME}/include/scene/scene.h
        ${ROOT_HOME}/include/scene/camera.h
        ${ROOT_HOME}/src/scene/scene.cpp
        ${ROOT_HOME}/src/scene/camera.cpp
    )
set(SCENELIB_NAME scene)
add_library(${SCENELIB_NAME} OBJECT ${SCENE_SOURCES})
target_include_directories(${SCENELIB_NAME} PUBLIC ${ROOT_HOME}/external/glm)
target_include_directories(${SCENELIB_NAME} PUBLIC ${ROOT_HOME}/include/scene)
target_include_directories(${SCENELIB_NAME} PUBLIC ${ROOT_HOME}/include/)
target_include_directories(${SCENELIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
