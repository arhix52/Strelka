cmake_minimum_required(VERSION 3.22)

find_package(glad REQUIRED)
find_package(imgui REQUIRED)
find_package(imguizmo REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

set(DISPLAY_SOURCES_COMMON
    ${CMAKE_CURRENT_SOURCE_DIR}/include/strelka/display/display.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/display.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/display_factory.cpp
)

if(WIN32 OR LINUX)
    find_package(CUDAToolkit REQUIRED)

    set(IMGUI_SOURCES
        ${ROOT_HOME}/external/imgui/imgui_impl_glfw.cpp
        ${ROOT_HOME}/external/imgui/imgui_impl_opengl3.cpp
        ${ROOT_HOME}/external/imgui/imgui_impl_glfw.h
        ${ROOT_HOME}/external/imgui/imgui_impl_opengl3.h
    )

    set(DISPLAY_SOURCES
        ${DISPLAY_SOURCES_COMMON}
        ${CMAKE_CURRENT_SOURCE_DIR}/opengl/glfw_display.h
        ${CMAKE_CURRENT_SOURCE_DIR}/opengl/glfw_display.cpp
    )

    add_library(strelka_display STATIC ${DISPLAY_SOURCES} ${IMGUI_SOURCES})

    target_include_directories(strelka_display
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${ROOT_HOME}/external/imgui/
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(strelka_display
        PUBLIC
            ${OPENGL_LIBRARIES}
            strelka_foundation
            strelka_render
            imgui::imgui
            imguizmo::imguizmo
            glfw
            glm::glm
            glad::glad
    )
endif()

if(APPLE)
    set(IMGUI_SOURCES
        ${ROOT_HOME}/external/imgui/imgui_impl_glfw.cpp
        ${ROOT_HOME}/external/imgui/imgui_impl_metal.mm
        ${ROOT_HOME}/external/imgui/imgui_impl_glfw.h
        ${ROOT_HOME}/external/imgui/imgui_impl_metal.h
    )

    set(DISPLAY_SOURCES
        ${DISPLAY_SOURCES_COMMON}
        ${CMAKE_CURRENT_SOURCE_DIR}/metal/glfw_display.h
        ${CMAKE_CURRENT_SOURCE_DIR}/metal/glfw_display.mm
    )

    add_library(strelka_display STATIC ${DISPLAY_SOURCES} ${IMGUI_SOURCES})

    target_include_directories(strelka_display
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${ROOT_HOME}/external/imgui/
            ${ROOT_HOME}/third_party/metal_cpp/
    )

    target_link_libraries(strelka_display
        PRIVATE
            "-framework Foundation"
            "-framework QuartzCore"
            "-framework CoreGraphics"
            "-framework Metal"
            "-framework MetalKit"
    )
    target_link_libraries(strelka_display
        PUBLIC
            strelka_foundation
            imgui::imgui
            glfw
            glm::glm
    )
endif()
