cmake_minimum_required(VERSION 3.16)

find_package(cxxopts REQUIRED)
find_package(imgui REQUIRED)
find_package(imguizmo REQUIRED)

# Application
set(EDITOR_SOURCES
    ${ROOT_HOME}/src/app/main.cpp
    ${ROOT_HOME}/src/app/Editor.h
    ${ROOT_HOME}/src/app/CameraController.h
    ${ROOT_HOME}/external/ImGuiFileDialog/ImGuiFileDialog.cpp
)
set(EDITOR_NAME Editor)

add_executable(${EDITOR_NAME} ${EDITOR_SOURCES})
set_target_properties(
  ${EDITOR_NAME}
  PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED TRUE
    CXX_EXTENSIONS OFF
)

if( WIN32 )
  target_compile_definitions( ${EDITOR_NAME} PUBLIC GLAD_GLAPI_EXPORT )
endif()

target_include_directories(${EDITOR_NAME} PUBLIC ${ROOT_HOME}/include/)
target_include_directories(${EDITOR_NAME} PUBLIC ${ROOT_HOME}/src/app/)
target_include_directories(${EDITOR_NAME} PUBLIC ${ROOT_HOME}/external/ImGuiFileDialog)

target_link_libraries(${EDITOR_NAME} PUBLIC 
  cxxopts::cxxopts
  imgui::imgui
  imguizmo::imguizmo
  scene
  sceneloader
  settings
  display
  render
  logger
)
