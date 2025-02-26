cmake_minimum_required(VERSION 3.22)
set(DISPLAYLIB_NAME display)

find_package(glad REQUIRED)
find_package(imgui REQUIRED)
find_package(imguizmo REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

set(DISPLAY_SOURCES_COMMON
    ${ROOT_HOME}/include/display/Display.h ${ROOT_HOME}/src/display/Display.cpp
    ${ROOT_HOME}/src/display/DisplayFactory.cpp)
if(WIN32 OR LINUX)
  enable_language(CUDA)
  find_package(CUDA REQUIRED)
  message(STATUS "CUDA Toolkit Root: ${CUDA_TOOLKIT_ROOT_DIR}")
  message(STATUS "CUDA Include DIR: ${CUDA_INCLUDE_DIRS}")

  set(IMGUI_SOURCES
    # Imgui
    ${ROOT_HOME}/external/imgui/imgui_impl_glfw.cpp
    ${ROOT_HOME}/external/imgui/imgui_impl_opengl3.cpp
    ${ROOT_HOME}/external/imgui/imgui_impl_glfw.h
    ${ROOT_HOME}/external/imgui/imgui_impl_opengl3.h)

  set(DISPLAY_SOURCES
      ${DISPLAY_SOURCES_COMMON} ${ROOT_HOME}/src/display/opengl/glfwdisplay.h
      ${ROOT_HOME}/src/display/opengl/glfwdisplay.cpp)

  add_library(${DISPLAYLIB_NAME} STATIC ${DISPLAY_SOURCES} ${IMGUI_SOURCES})
  target_include_directories(${DISPLAYLIB_NAME} PUBLIC
    ${ROOT_HOME}/include/
    ${ROOT_HOME}/include/display
    ${ROOT_HOME}/external/imgui/
  )

  include_directories(${CUDA_INCLUDE_DIRS})
  link_directories(${CUDA_LIBRARIES})
  target_link_libraries(${DISPLAYLIB_NAME}
    PUBLIC ${OPENGL_LIBRARIES}
    logger imgui::imgui imguizmo::imguizmo glfw glm::glm glad::glad)

endif()

if(APPLE)
  set(DISPLAY_SOURCES
      ${DISPLAY_SOURCES_COMMON} ${ROOT_HOME}/src/display/metal/glfwdisplay.h
      ${ROOT_HOME}/src/display/metal/glfwdisplay.mm
      )

  set(IMGUI_SOURCES
      # Imgui
      ${ROOT_HOME}/external/imgui/imgui_impl_glfw.cpp
      ${ROOT_HOME}/external/imgui/imgui_impl_metal.mm
      ${ROOT_HOME}/external/imgui/imgui_impl_glfw.h
      ${ROOT_HOME}/external/imgui/imgui_impl_metal.h
      )

  add_library(${DISPLAYLIB_NAME} STATIC ${DISPLAY_SOURCES} ${IMGUI_SOURCES})
  target_include_directories(${DISPLAYLIB_NAME} PUBLIC
    ${ROOT_HOME}/external/imgui/
    ${ROOT_HOME}/include/
    ${ROOT_HOME}/include/display
    ${ROOT_HOME}/external/metal-cpp/
    ${ROOT_HOME}
  )

  target_link_libraries(${DISPLAYLIB_NAME} PRIVATE "-framework Foundation")
  target_link_libraries(${DISPLAYLIB_NAME} PRIVATE "-framework QuartzCore")
  target_link_libraries(${DISPLAYLIB_NAME} PRIVATE "-framework CoreGraphics")
  target_link_libraries(${DISPLAYLIB_NAME} PRIVATE "-framework Metal")
  target_link_libraries(${DISPLAYLIB_NAME} PRIVATE "-framework MetalKit")
  target_link_libraries(${DISPLAYLIB_NAME} PUBLIC logger imgui::imgui glfw glm::glm)

endif()
