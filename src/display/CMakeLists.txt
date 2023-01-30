cmake_minimum_required(VERSION 3.20)

set(DISPLAY_SOURCES
    ${ROOT_HOME}/include/display/glfwdisplay.h
    ${ROOT_HOME}/src/display/glfwdisplay.cpp
    )

set(IMGUI_SOURCES
        # Imgui
        ${ROOT_HOME}/external/imgui/src/imgui.cpp
        ${ROOT_HOME}/external/imgui/src/imgui_impl_opengl3.cpp
        ${ROOT_HOME}/external/imgui/src/imgui_impl_glfw.cpp
        ${ROOT_HOME}/external/imgui/src/imgui_draw.cpp
        ${ROOT_HOME}/external/imgui/src/imgui_widgets.cpp
        ${ROOT_HOME}/external/imgui/src/imgui_tables.cpp
        )
set(DISPLAYLIB_NAME display)
add_library(${DISPLAYLIB_NAME} STATIC ${DISPLAY_SOURCES} ${IMGUI_SOURCES})
target_include_directories(${DISPLAYLIB_NAME} PUBLIC ${ROOT_HOME}/include/)
target_include_directories(${DISPLAYLIB_NAME} PUBLIC ${ROOT_HOME}/include/display)

target_include_directories(${DISPLAYLIB_NAME} PUBLIC ${ROOT_HOME}/external/glm)
target_include_directories(${DISPLAYLIB_NAME} PUBLIC ${ROOT_HOME}/external/glfw/include)
target_include_directories(${DISPLAYLIB_NAME} PUBLIC ${ROOT_HOME}/external/glad/include)
target_include_directories(${DISPLAYLIB_NAME} PUBLIC ${ROOT_HOME}/external/imgui/include)

target_link_libraries(${DISPLAYLIB_NAME} PUBLIC 
    "${ROOT_HOME}/external/glfw/glfw3.lib"
    ${OPENGL_LIBRARIES}
    PRIVATE
    glad
)
