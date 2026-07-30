cmake_minimum_required(VERSION 3.22)

find_package(cxxopts REQUIRED)
find_package(imgui REQUIRED)
find_package(imguizmo REQUIRED)

set(EDITOR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/EditorApp.h
    ${CMAKE_CURRENT_SOURCE_DIR}/EditorApp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CameraController.h
    ${CMAKE_CURRENT_SOURCE_DIR}/panels/ViewportPanel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/panels/RenderSettingsPanel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/panels/AnimationPanel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/panels/PropertyPanel.cpp
    ${ROOT_HOME}/third_party/imgui_file_dialog/ImGuiFileDialog.cpp
)

add_executable(StrelkaEditor ${EDITOR_SOURCES})

# Ship the default docking layout next to the binary. On first run the editor
# copies it to its own imgui.ini, so a fresh checkout starts with the panels
# already arranged instead of an empty dockspace.
add_custom_command(
    OUTPUT ${OUTPUT_DIRECTORY}/default_layout.ini
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/default_layout.ini
        ${OUTPUT_DIRECTORY}/default_layout.ini
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/default_layout.ini
    COMMENT "Copying default ImGui layout"
)
add_custom_target(strelka_editor_layout ALL DEPENDS ${OUTPUT_DIRECTORY}/default_layout.ini)
add_dependencies(StrelkaEditor strelka_editor_layout)

set_target_properties(StrelkaEditor PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED TRUE
    CXX_EXTENSIONS OFF
)

if(WIN32)
    target_compile_definitions(StrelkaEditor PUBLIC GLAD_GLAPI_EXPORT)
endif()

target_include_directories(StrelkaEditor
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ROOT_HOME}/third_party/imgui_file_dialog
)

# strelka_display already PUBLIC-links strelka_render, strelka_foundation and
# imgui, and strelka_render PUBLIC-links strelka_scene — listing them again here
# made the linker warn about duplicate libraries. Link only what this target
# uses directly and let usage requirements propagate the rest.
#
# This is an executable: its link interface is never consumed, so PRIVATE.
set(EDITOR_LINK_LIBS
    cxxopts::cxxopts
    imguizmo::imguizmo
    strelka_display
    strelka_sceneloader
)

if(APPLE)
    # On Apple, strelka_shaders is a custom_target (metallib compilation), not a
    # library. Use add_dependencies for build ordering; no link needed.
    target_link_libraries(StrelkaEditor PRIVATE ${EDITOR_LINK_LIBS})
    add_dependencies(StrelkaEditor strelka_shaders)
else()
    target_link_libraries(StrelkaEditor PRIVATE ${EDITOR_LINK_LIBS} strelka_shaders)
endif()
