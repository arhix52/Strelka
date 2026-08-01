cmake_minimum_required(VERSION 3.22)

find_package(USD REQUIRED HINTS ${USD_DIR} NAMES pxr)
message(STATUS "USD LIBRARY: ${USD_DIR}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_definitions(BOOST_NO_CXX98_FUNCTION_BASE)

# Hydra plugin
set(HD_PLUGIN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/RendererPlugin.h
    ${CMAKE_CURRENT_SOURCE_DIR}/RendererPlugin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RenderParam.h
    ${CMAKE_CURRENT_SOURCE_DIR}/RenderBuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/RenderDelegate.h
    ${CMAKE_CURRENT_SOURCE_DIR}/RenderPass.h
    ${CMAKE_CURRENT_SOURCE_DIR}/RenderBuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RenderDelegate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RenderPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Instancer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Instancer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Material.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Material.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Camera.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Camera.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Mesh.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Mesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BasisCurves.h
    ${CMAKE_CURRENT_SOURCE_DIR}/BasisCurves.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Light.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Light.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Tokens.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Tokens.cpp)
set(HD_PLUGIN_NAME HdStrelka)
set(Boost_USE_STATIC_LIBS OFF)
set(BUILD_SHARED_LIBS ON)

add_library(${HD_PLUGIN_NAME} SHARED ${HD_PLUGIN_SOURCES})
set_target_properties(
  ${HD_PLUGIN_NAME}
  PROPERTIES LINKER_LANGUAGE CXX
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF
             INSTALL_RPATH_USE_LINK_PATH TRUE
             PREFIX "")

target_compile_definitions(
  ${HD_PLUGIN_NAME}
  PUBLIC MFB_PACKAGE_NAME=${HD_PLUGIN_NAME}
  PRIVATE
    "$<$<OR:$<CONFIG:>,$<CONFIG:Debug>>:TBB_USE_DEBUG>")

# Workaround for https://github.com/PixarAnimationStudios/USD/issues/1279
if(MSVC_VERSION GREATER_EQUAL 1920)
  target_compile_options(${HD_PLUGIN_NAME} PRIVATE "/Zc:inline-")
endif()
set(Boost_USE_STATIC_LIBS OFF)
set(BUILD_SHARED_LIBS ON)

target_include_directories(${HD_PLUGIN_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${HD_PLUGIN_NAME} PRIVATE usdImaging)

# Add the linker options to suppress the MSVCRT conflict warning
if(WIN32)
  target_link_options(${HD_PLUGIN_NAME} PRIVATE "/NODEFAULTLIB:MSVCRT")
  target_link_options(${HD_PLUGIN_NAME} PRIVATE "/NODEFAULTLIB:LIBCMT")
endif()

if(WIN32 OR LINUX)
  target_link_libraries(
    ${HD_PLUGIN_NAME}
    PRIVATE usdImaging
            strelka_render
            strelka_scene
    PUBLIC hd)
endif()

if(APPLE)
  target_link_libraries(
    ${HD_PLUGIN_NAME}
    PRIVATE usdImaging strelka_render strelka_scene
    PUBLIC hd)
  target_link_libraries(${HD_PLUGIN_NAME} PUBLIC "-framework Foundation")
  target_link_libraries(${HD_PLUGIN_NAME} PUBLIC "-framework QuartzCore")
  target_link_libraries(${HD_PLUGIN_NAME} PUBLIC "-framework Metal")
  target_link_libraries(${HD_PLUGIN_NAME} PUBLIC "-framework MetalKit")
endif(APPLE)

target_link_libraries(${HD_PLUGIN_NAME} PUBLIC strelka_foundation)

set(PLUGINFO_PATH "${CMAKE_CURRENT_BINARY_DIR}/plugInfo.json")

set(CMAKE_INSTALL_PREFIX "${USD_DIR}/plugin/usd/")

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/plugInfo.json.in PLUGINFO)
file(
  GENERATE
  OUTPUT "${PLUGINFO_PATH}"
  CONTENT ${PLUGINFO})

install(
  FILES "${PLUGINFO_PATH}"
  DESTINATION "${CMAKE_INSTALL_PREFIX}/HdStrelka/resources"
  COMPONENT ${HD_PLUGIN_NAME})

install(
  TARGETS ${HD_PLUGIN_NAME}
  LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}" COMPONENT ${HD_PLUGIN_NAME}
  RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}" COMPONENT ${HD_PLUGIN_NAME}
  ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}" COMPONENT ${HD_PLUGIN_NAME})
