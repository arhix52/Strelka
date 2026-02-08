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

if(APPLE)
    # On Apple, strelka_shaders is a custom_target (metallib compilation), not a library.
    # Use add_dependencies for build ordering; no link needed.
    target_link_libraries(StrelkaEditor
        PUBLIC
            cxxopts::cxxopts
            imgui::imgui
            imguizmo::imguizmo
            strelka_render
            strelka_display
            strelka_sceneloader
            strelka_foundation
            strelka_scene
    )
    add_dependencies(StrelkaEditor strelka_shaders)
else()
    target_link_libraries(StrelkaEditor
        PUBLIC
            cxxopts::cxxopts
            imgui::imgui
            imguizmo::imguizmo
            strelka_render
            strelka_shaders
            strelka_display
            strelka_sceneloader
            strelka_foundation
            strelka_scene
    )
endif()
