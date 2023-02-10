cmake_minimum_required(VERSION 3.24)
set(DISPLAYLIB_NAME display)
set(DISPLAY_SOURCES_COMMON ${ROOT_HOME}/include/display/Display.h
                           ${ROOT_HOME}/src/display/Display.cpp
                           ${ROOT_HOME}/src/display/DisplayFactory.cpp)
if(WIN32)

  set(DISPLAY_SOURCES
      ${DISPLAY_SOURCES_COMMON} ${ROOT_HOME}/src/display/opengl/glfwdisplay.h
      ${ROOT_HOME}/src/display/opengl/glfwdisplay.cpp)

  set(IMGUI_SOURCES
      # Imgui
      ${ROOT_HOME}/external/imgui/src/imgui.cpp
      ${ROOT_HOME}/external/imgui/src/imgui_impl_opengl3.cpp
      ${ROOT_HOME}/external/imgui/src/imgui_impl_glfw.cpp
      ${ROOT_HOME}/external/imgui/src/imgui_draw.cpp
      ${ROOT_HOME}/external/imgui/src/imgui_widgets.cpp
      ${ROOT_HOME}/external/imgui/src/imgui_tables.cpp)
  add_library(${DISPLAYLIB_NAME} STATIC ${DISPLAY_SOURCES} ${IMGUI_SOURCES})
  target_include_directories(${DISPLAYLIB_NAME} PUBLIC ${ROOT_HOME}/include/)
  target_include_directories(${DISPLAYLIB_NAME}
                             PUBLIC ${ROOT_HOME}/include/display)

  target_include_directories(${DISPLAYLIB_NAME}
                             PUBLIC ${ROOT_HOME}/external/glm)
  target_include_directories(${DISPLAYLIB_NAME}
                             PUBLIC ${ROOT_HOME}/external/glfw/include)
  target_include_directories(${DISPLAYLIB_NAME}
                             PUBLIC ${ROOT_HOME}/external/glad/include)
  target_include_directories(${DISPLAYLIB_NAME}
                             PUBLIC ${ROOT_HOME}/external/imgui/include)

  target_link_libraries(
    ${DISPLAYLIB_NAME}
    PUBLIC "${ROOT_HOME}/external/glfw/glfw3.lib" ${OPENGL_LIBRARIES}
    PRIVATE glad)
endif()

if(APPLE)
  set(DISPLAY_SOURCES
      ${DISPLAY_SOURCES_COMMON} ${ROOT_HOME}/src/display/metal/glfwdisplay.h
      ${ROOT_HOME}/src/display/metal/glfwdisplay.mm)

  set(IMGUI_SOURCES
      # Imgui
      ${ROOT_HOME}/external/imgui/src/imgui.cpp
      ${ROOT_HOME}/external/imgui/src/imgui_impl_glfw.cpp
      ${ROOT_HOME}/external/imgui/src/imgui_impl_metal.mm
      ${ROOT_HOME}/external/imgui/src/imgui_draw.cpp
      ${ROOT_HOME}/external/imgui/src/imgui_widgets.cpp
      ${ROOT_HOME}/external/imgui/src/imgui_tables.cpp)

      add_library(${DISPLAYLIB_NAME} STATIC ${DISPLAY_SOURCES} ${IMGUI_SOURCES})
      target_include_directories(${DISPLAYLIB_NAME} PUBLIC ${ROOT_HOME}/external/imgui/include)
      target_include_directories(${DISPLAYLIB_NAME} PUBLIC ${ROOT_HOME}/external/glfw-3.3.8.bin.MACOS/include)

endif()
