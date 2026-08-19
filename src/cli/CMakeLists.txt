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
strelka_apply_warnings(StrelkaCLI)

set_target_properties(StrelkaCLI PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED TRUE
    CXX_EXTENSIONS OFF
)

# The tone curve is shared with the shaders rather than copied here; see the
# note at the top of tonemappers.h.
target_include_directories(StrelkaCLI
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/shaders/common
        ${CMAKE_SOURCE_DIR}/src/editor
)

# strelka_render already PUBLIC-links tinyexr, stb, strelka_foundation and
# strelka_scene on both backends, and strelka_sceneloader PUBLIC-links the last
# two as well. Naming them again here put the same archives on the link line
# twice and made ld warn about it. Link only what nothing else provides and let
# usage requirements carry the rest, as StrelkaEditor already does.
target_link_libraries(StrelkaCLI
    PRIVATE
        cxxopts::cxxopts
        tomlplusplus::tomlplusplus
        strelka_render
        strelka_sceneloader
)

# On Apple, strelka_shaders is a custom_target (metallib compilation), not a library.
if(APPLE)
    add_dependencies(StrelkaCLI strelka_shaders)
    install(TARGETS StrelkaCLI RUNTIME DESTINATION .)
endif()
