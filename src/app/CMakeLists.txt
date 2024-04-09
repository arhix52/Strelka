cmake_minimum_required(VERSION 3.16)

find_package(cxxopts REQUIRED)

# Application
set(RUNNER_SOURCES
    ${ROOT_HOME}/src/app/main.cpp
)
set(RUNNER_NAME runner)

add_executable(${RUNNER_NAME} ${RUNNER_SOURCES})
set_target_properties(
  ${RUNNER_NAME}
  PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED TRUE
    CXX_EXTENSIONS OFF
)

if( WIN32 )
  target_compile_definitions( ${RUNNER_NAME} PUBLIC GLAD_GLAPI_EXPORT )
endif()

target_include_directories(${RUNNER_NAME} PUBLIC ${ROOT_HOME}/include/)

target_link_libraries(${RUNNER_NAME} PUBLIC 
  cxxopts::cxxopts
  scene
  sceneloader
  settings
  display
  render
  logger
)
