cmake_minimum_required(VERSION 3.24)
set(DISPLAYLIB_NAME display)

find_package(imgui REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

set(DISPLAY_SOURCES_COMMON
    ${ROOT_HOME}/include/display/Display.h ${ROOT_HOME}/src/display/Display.cpp
    ${ROOT_HOME}/src/display/DisplayFactory.cpp)
if(WIN32 OR LINUX)

  set(DISPLAY_SOURCES
      ${DISPLAY_SOURCES_COMMON} ${ROOT_HOME}/src/display/opengl/glfwdisplay.h
      ${ROOT_HOME}/src/display/opengl/glfwdisplay.cpp)

  add_library(${DISPLAYLIB_NAME} STATIC ${DISPLAY_SOURCES})
  target_include_directories(${DISPLAYLIB_NAME} PUBLIC ${ROOT_HOME}/include/)
  target_include_directories(${DISPLAYLIB_NAME}
                             PUBLIC ${ROOT_HOME}/include/display)

  target_include_directories(${DISPLAYLIB_NAME}
                             PUBLIC ${ROOT_HOME}/external/glm)
  target_include_directories(${DISPLAYLIB_NAME}
                             PUBLIC ${ROOT_HOME}/external/glfw/include)
  target_include_directories(${DISPLAYLIB_NAME}
                             PUBLIC ${ROOT_HOME}/external/glad/include)

  if(WIN32)
    target_link_libraries(
      ${DISPLAYLIB_NAME}
      PUBLIC "${ROOT_HOME}/external/glfw/glfw3.lib" ${OPENGL_LIBRARIES}
      PRIVATE glad)
  endif(WIN32)

  if(LINUX)
    target_link_libraries(
      ${DISPLAYLIB_NAME}
      PUBLIC ${OPENGL_LIBRARIES} glfw
      PRIVATE glad)
  endif(LINUX)

endif()

if(APPLE)
  set(DISPLAY_SOURCES
      ${DISPLAY_SOURCES_COMMON} ${ROOT_HOME}/src/display/metal/glfwdisplay.h
      ${ROOT_HOME}/src/display/metal/glfwdisplay.mm
      ${ROOT_HOME}/src/display/metal/metalcpphelp.cpp)

  set(IMGUI_SOURCES
      # Imgui
      ${ROOT_HOME}/external/imgui/imgui_impl_glfw.cpp
      ${ROOT_HOME}/external/imgui/imgui_impl_metal.mm
      ${ROOT_HOME}/external/imgui/imgui_impl_opengl3.cpp
      ${ROOT_HOME}/external/imgui/imgui_impl_glfw.h
      ${ROOT_HOME}/external/imgui/imgui_impl_metal.h
      ${ROOT_HOME}/external/imgui/imgui_impl_opengl3.h)

  add_library(${DISPLAYLIB_NAME} STATIC ${DISPLAY_SOURCES} ${IMGUI_SOURCES})
  target_include_directories(${DISPLAYLIB_NAME}
                             PUBLIC ${ROOT_HOME}/external/imgui/)
  target_include_directories(${DISPLAYLIB_NAME} PUBLIC ${ROOT_HOME}/include/)
  target_include_directories(${DISPLAYLIB_NAME}
                             PUBLIC ${ROOT_HOME}/include/display)
  target_include_directories(${DISPLAYLIB_NAME}
                             PUBLIC ${ROOT_HOME}/external/glm)
  target_include_directories(${DISPLAYLIB_NAME}
                             PUBLIC ${ROOT_HOME}/external/metal-cpp/)
  target_include_directories(${DISPLAYLIB_NAME} PUBLIC ${ROOT_HOME})

  target_link_libraries(${DISPLAYLIB_NAME} PUBLIC "-framework Foundation")
  target_link_libraries(${DISPLAYLIB_NAME} PUBLIC "-framework QuartzCore")
  target_link_libraries(${DISPLAYLIB_NAME} PUBLIC "-framework CoreGraphics")
  target_link_libraries(${DISPLAYLIB_NAME} PUBLIC "-framework Metal")
  target_link_libraries(${DISPLAYLIB_NAME} PUBLIC "-framework MetalKit")
  target_link_libraries(${DISPLAYLIB_NAME} PUBLIC logger)
  target_link_libraries(${DISPLAYLIB_NAME} PUBLIC imgui::imgui glfw glm::glm)

endif()
