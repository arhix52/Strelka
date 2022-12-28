cmake_minimum_required(VERSION 3.16)

# Settings
set(SETTINGSLIB_SOURCES
        ${ROOT_HOME}/include/settings/settings.h
        ${ROOT_HOME}/src/settings/settings.cpp
        )
set(SETTINGSLIB_NAME settings)
add_library(${SETTINGSLIB_NAME} OBJECT ${SETTINGSLIB_SOURCES})
target_include_directories(${SETTINGSLIB_NAME} PUBLIC ${ROOT_HOME}/include/)
target_include_directories(${SETTINGSLIB_NAME} PUBLIC ${ROOT_HOME}/external/glm)
