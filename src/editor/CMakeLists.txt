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
    ${CMAKE_CURRENT_SOURCE_DIR}/panels/CameraPanel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/panels/GamepadSettings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/panels/LoadingOverlay.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/panels/AnimationPanel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/panels/PropertyPanel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/panels/OutlinerPanel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/panels/MaterialPanel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/panels/MemoryPanel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/panels/RenderDebugPanel.cpp
)

add_library(strelka_vendor_file_dialog OBJECT
    ${ROOT_HOME}/third_party/imgui_file_dialog/ImGuiFileDialog.cpp
)
strelka_apply_vendor_warnings(strelka_vendor_file_dialog)
# ImGui's headers and defines, not its archive. These objects are folded into
# StrelkaEditor, which already links imgui through strelka_display; naming the
# library here as well put the same archive on the link line twice and made ld
# warn about it. Spelled with target properties rather than $<COMPILE_ONLY:> so
# it still works on the CMake version this project claims as its minimum.
target_include_directories(strelka_vendor_file_dialog PRIVATE
    ${ROOT_HOME}/third_party/imgui_file_dialog
    $<TARGET_PROPERTY:imgui::imgui,INTERFACE_INCLUDE_DIRECTORIES>
)
target_compile_definitions(strelka_vendor_file_dialog PRIVATE
    $<TARGET_PROPERTY:imgui::imgui,INTERFACE_COMPILE_DEFINITIONS>
)

if(APPLE)
    add_executable(StrelkaEditor MACOSX_BUNDLE ${EDITOR_SOURCES})
else()
    add_executable(StrelkaEditor ${EDITOR_SOURCES})
endif()
strelka_apply_warnings(StrelkaEditor)

if(APPLE)
    # CMake copies files carrying MACOSX_PACKAGE_LOCATION into the built bundle
    # and keeps those copies up to date. The CLI retains the flat runtime trees
    # produced by src/shaders and the root MaterialX target.
    set(STRELKA_EDITOR_METAL_RESOURCES
        ${OUTPUT_DIRECTORY}/metal/shaders/wavefront.metallib
        ${OUTPUT_DIRECTORY}/metal/shaders/tonemapper.metallib
        ${OUTPUT_DIRECTORY}/metal/shaders/skinning.metallib
        ${OUTPUT_DIRECTORY}/metal/shaders/fullScreen.metal
    )
    set_source_files_properties(${STRELKA_EDITOR_METAL_RESOURCES} PROPERTIES
        GENERATED TRUE
        MACOSX_PACKAGE_LOCATION "Resources/metal/shaders"
    )

    file(GLOB_RECURSE STRELKA_EDITOR_MATERIALX_RESOURCES
        CONFIGURE_DEPENDS
        LIST_DIRECTORIES FALSE
        "${CMAKE_SOURCE_DIR}/third_party/materialx/libraries/*"
    )
    foreach(resource IN LISTS STRELKA_EDITOR_MATERIALX_RESOURCES)
        file(RELATIVE_PATH relative "${CMAKE_SOURCE_DIR}/third_party/materialx/libraries" "${resource}")
        get_filename_component(relative_directory "${relative}" DIRECTORY)
        set_source_files_properties("${resource}" PROPERTIES
            MACOSX_PACKAGE_LOCATION "Resources/materialx/libraries/${relative_directory}"
        )
    endforeach()

    set(STRELKA_EDITOR_ROOT_RESOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/default_layout.ini
        ${CMAKE_SOURCE_DIR}/LICENSE
        ${CMAKE_SOURCE_DIR}/resources/macos/Strelka.icns
    )
    set_source_files_properties(${STRELKA_EDITOR_ROOT_RESOURCES} PROPERTIES
        MACOSX_PACKAGE_LOCATION "Resources"
    )
    target_sources(StrelkaEditor PRIVATE
        ${STRELKA_EDITOR_METAL_RESOURCES}
        ${STRELKA_EDITOR_MATERIALX_RESOURCES}
        ${STRELKA_EDITOR_ROOT_RESOURCES}
    )
else()
    # Ship the default docking layout next to the binary. On first run the
    # editor copies it to imgui.ini.
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
endif()

set_target_properties(StrelkaEditor PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED TRUE
    CXX_EXTENSIONS OFF
)

target_include_directories(StrelkaEditor
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ROOT_HOME}/third_party/imgui_file_dialog
        ${CMAKE_BINARY_DIR}/generated
)

# strelka_display already PUBLIC-links strelka_render, strelka_foundation and
# imgui, and strelka_render PUBLIC-links strelka_scene — listing them again here
# made the linker warn about duplicate libraries. Link only what this target
# uses directly and let usage requirements propagate the rest.
#
# This is an executable: its link interface is never consumed, so PRIVATE.
set(EDITOR_LINK_LIBS
    cxxopts::cxxopts
    strelka_display
    strelka_sceneloader
)

if(APPLE)
    # On Apple, strelka_shaders is a custom_target (metallib compilation), not a
    # library. Use add_dependencies for build ordering; no link needed.
    target_link_libraries(StrelkaEditor PRIVATE ${EDITOR_LINK_LIBS} strelka_vendor_file_dialog)
    add_dependencies(StrelkaEditor strelka_shaders)
else()
    target_link_libraries(StrelkaEditor PRIVATE ${EDITOR_LINK_LIBS} strelka_vendor_file_dialog strelka_shaders)
endif()

if(APPLE)
    set_target_properties(StrelkaEditor PROPERTIES
        OUTPUT_NAME Strelka
        MACOSX_BUNDLE_INFO_PLIST ${CMAKE_SOURCE_DIR}/resources/macos/Info.plist.in
        MACOSX_BUNDLE_BUNDLE_NAME Strelka
        MACOSX_BUNDLE_GUI_IDENTIFIER ${STRELKA_BUNDLE_IDENTIFIER}
        MACOSX_BUNDLE_ICON_FILE Strelka.icns
        MACOSX_BUNDLE_SHORT_VERSION_STRING ${STRELKA_MARKETING_VERSION}
        MACOSX_BUNDLE_BUNDLE_VERSION ${STRELKA_BUILD_NUMBER}
        MACOSX_BUNDLE_COPYRIGHT "Copyright © 2024-2026 Strelka contributors"
    )
    install(TARGETS StrelkaEditor BUNDLE DESTINATION .)
else()
    # bin/ and share/strelka, which is where a Linux package puts a program and
    # the things that are not one. resolveResourcePath() looks in
    # ../share/strelka relative to the executable for exactly this layout.
    install(TARGETS StrelkaEditor RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/default_layout.ini
            DESTINATION ${CMAKE_INSTALL_DATADIR}/strelka)
endif()
