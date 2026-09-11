cmake_minimum_required(VERSION 3.22)

find_package(spdlog REQUIRED)
find_package(glm REQUIRED)

# --- strelka_foundation (STATIC) ---
# Combines logger + settings into one foundation library

set(FOUNDATION_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/log/log.h
    ${CMAKE_CURRENT_SOURCE_DIR}/log/logmanager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/log/logmanager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/settings/settings.h
    ${CMAKE_CURRENT_SOURCE_DIR}/settings/settings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/env/env.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paths/paths.h
    ${CMAKE_CURRENT_SOURCE_DIR}/progress/loadprogress.h
    ${CMAKE_CURRENT_SOURCE_DIR}/memory/hugepages.h
)

add_library(strelka_foundation STATIC ${FOUNDATION_SOURCES})
strelka_apply_warnings(strelka_foundation)

target_include_directories(strelka_foundation
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/log
        ${CMAKE_CURRENT_SOURCE_DIR}/settings
        ${CMAKE_CURRENT_SOURCE_DIR}/env
        ${CMAKE_CURRENT_SOURCE_DIR}/paths
        ${CMAKE_CURRENT_SOURCE_DIR}/progress
        ${CMAKE_CURRENT_SOURCE_DIR}/memory
)

target_link_libraries(strelka_foundation
    PUBLIC
        spdlog::spdlog
        glm::glm
)
