cmake_minimum_required(VERSION 3.20)

find_package(glm REQUIRED)

# Settings
set(SETTINGSLIB_SOURCES
        ${ROOT_HOME}/include/settings/settings.h
        ${ROOT_HOME}/src/settings/settings.cpp
        )
set(SETTINGSLIB_NAME settings)
add_library(${SETTINGSLIB_NAME} OBJECT ${SETTINGSLIB_SOURCES})
target_include_directories(${SETTINGSLIB_NAME} PUBLIC ${ROOT_HOME}/include/settings)
target_include_directories(${SETTINGSLIB_NAME} PUBLIC ${ROOT_HOME}/include/log)
target_link_libraries(${SETTINGSLIB_NAME} PUBLIC logger glm::glm)
