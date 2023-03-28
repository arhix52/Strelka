cmake_minimum_required(VERSION 3.24)

find_package(USD REQUIRED HINTS ${USD_DIR} NAMES pxr)
message(STATUS "USD LIBRARY: ${USD_DIR}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_definitions(TRACY_ENABLE)

# Hydra plugin
set(HD_PLUGIN_SOURCES
    ${ROOT_HOME}/include/HdStrelka/RendererPlugin.h
    ${ROOT_HOME}/src/HdStrelka/RendererPlugin.cpp
    ${ROOT_HOME}/src/HdStrelka/RenderParam.h
    ${ROOT_HOME}/src/HdStrelka/RenderBuffer.h
    ${ROOT_HOME}/src/HdStrelka/RenderDelegate.h
    ${ROOT_HOME}/src/HdStrelka/RenderPass.h
    ${ROOT_HOME}/src/HdStrelka/RenderBuffer.cpp
    ${ROOT_HOME}/src/HdStrelka/RenderDelegate.cpp
    ${ROOT_HOME}/src/HdStrelka/RenderPass.cpp
    ${ROOT_HOME}/src/HdStrelka/Instancer.h
    ${ROOT_HOME}/src/HdStrelka/Instancer.cpp
    ${ROOT_HOME}/src/HdStrelka/Material.h
    ${ROOT_HOME}/src/HdStrelka/Material.cpp
    ${ROOT_HOME}/src/HdStrelka/Camera.h
    ${ROOT_HOME}/src/HdStrelka/Camera.cpp
    ${ROOT_HOME}/src/HdStrelka/Mesh.h
    ${ROOT_HOME}/src/HdStrelka/Mesh.cpp
    ${ROOT_HOME}/src/HdStrelka/BasisCurves.h
    ${ROOT_HOME}/src/HdStrelka/BasisCurves.cpp
    ${ROOT_HOME}/src/HdStrelka/Light.h
    ${ROOT_HOME}/src/HdStrelka/Light.cpp
    ${ROOT_HOME}/src/HdStrelka/Tokens.h
    ${ROOT_HOME}/src/HdStrelka/Tokens.cpp
    ${ROOT_HOME}/src/HdStrelka/MaterialNetworkTranslator.h
    ${ROOT_HOME}/src/HdStrelka/MaterialNetworkTranslator.cpp
    ${ROOT_HOME}/src/HdStrelka/MdlParserPlugin.h
    ${ROOT_HOME}/src/HdStrelka/MdlParserPlugin.cpp
    ${ROOT_HOME}/src/HdStrelka/MdlDiscoveryPlugin.h
    ${ROOT_HOME}/src/HdStrelka/MdlDiscoveryPlugin.cpp)
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
             # The other libs in the plugin dir have no "lib" prefix, so let's
             # match this
             PREFIX "")

target_compile_definitions(
  ${HD_PLUGIN_NAME}
  PUBLIC # Required for PLUG_THIS_PLUGIN macro
         MFB_PACKAGE_NAME=${HD_PLUGIN_NAME}
  PRIVATE
    # Workaround for
    # https://github.com/PixarAnimationStudios/USD/issues/1471#issuecomment-799813477
    "$<$<OR:$<CONFIG:>,$<CONFIG:Debug>>:TBB_USE_DEBUG>")

# Workaround for https://github.com/PixarAnimationStudios/USD/issues/1279
if(MSVC_VERSION GREATER_EQUAL 1920)
  target_compile_options(${HD_PLUGIN_NAME} PRIVATE "/Zc:inline-")
endif()
set(Boost_USE_STATIC_LIBS OFF)
set(BUILD_SHARED_LIBS ON)

target_include_directories(${HD_PLUGIN_NAME} PUBLIC ${ROOT_HOME}/include/log)
target_include_directories(${HD_PLUGIN_NAME}
                           PUBLIC ${ROOT_HOME}/include/hdStrelka)
target_include_directories(${HD_PLUGIN_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${HD_PLUGIN_NAME} PUBLIC hd)
target_include_directories(${HD_PLUGIN_NAME} PRIVATE usdImaging hdMtlx)
target_include_directories(${HD_PLUGIN_NAME} PUBLIC ${ROOT_HOME}/external/tracy/public/)

if(WIN32)
  target_link_libraries(
    ${HD_PLUGIN_NAME}
    PRIVATE usdImaging
            hdMtlx
            ${OPENGL_LIBRARIES}
            glad
            render
            scene
            materialmanager
    PUBLIC hd)
endif(WIN32)

if(APPLE)
  target_link_libraries(
    ${HD_PLUGIN_NAME}
    PRIVATE usdImaging hdMtlx render scene materialmanager
    PUBLIC hd)
  target_link_libraries(${HD_PLUGIN_NAME} PUBLIC "-framework Foundation")
  target_link_libraries(${HD_PLUGIN_NAME} PUBLIC "-framework QuartzCore")
  target_link_libraries(${HD_PLUGIN_NAME} PUBLIC "-framework Metal")
  target_link_libraries(${HD_PLUGIN_NAME} PUBLIC "-framework MetalKit")
endif(APPLE)

target_link_libraries(${HD_PLUGIN_NAME} PUBLIC logger)

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
