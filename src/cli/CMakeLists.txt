cmake_minimum_required(VERSION 3.22)

find_package(cxxopts REQUIRED)
find_package(tomlplusplus REQUIRED)
find_package(tinyexr REQUIRED)
find_package(stb REQUIRED)

add_executable(StrelkaCLI
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/HeadlessApp.h
    ${CMAKE_CURRENT_SOURCE_DIR}/HeadlessApp.cpp
)

set_target_properties(StrelkaCLI PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED TRUE
    CXX_EXTENSIONS OFF
)

target_link_libraries(StrelkaCLI
    PRIVATE
        cxxopts::cxxopts
        tomlplusplus::tomlplusplus
        tinyexr::tinyexr
        stb::stb
        strelka_render
        strelka_sceneloader
        strelka_foundation
        strelka_scene
)

# On Apple, strelka_shaders is a custom_target (metallib compilation), not a library.
if(APPLE)
    add_dependencies(StrelkaCLI strelka_shaders)
endif()
