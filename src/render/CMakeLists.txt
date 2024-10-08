cmake_minimum_required(VERSION 3.22)
set(RENDERLIB_NAME render)

find_package(glm REQUIRED)
find_package(stb REQUIRED)

# include(${ROOT_HOME}/cmake/StaticAnalyzers.cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(RENDER_SOURCES_COMMON ${ROOT_HOME}/include/render/render.h
                          ${ROOT_HOME}/src/render/render.cpp)

if(WIN32 OR LINUX)
  enable_language(CUDA)
  find_package(CUDA REQUIRED)
  message(STATUS "CUDA Toolkit Root: ${CUDA_TOOLKIT_ROOT_DIR}")
  message(STATUS "CUDA Include DIR: ${CUDA_INCLUDE_DIRS}")
  # Search for the OptiX libraries and include files.
  set(OptiX_INSTALL_DIR $ENV{OPTIX_DIR})
  message(STATUS "OptiX SDK DIR: ${OptiX_INSTALL_DIR}")
  # Search for the OptiX libraries and include files.
  find_package(OptiX REQUIRED)
  message(STATUS "OptiX Include DIR: ${OptiX_INCLUDE}")
  include_directories(${OptiX_INCLUDE})

  # Hack to get intellisense working for CUDA includes
  if(MSVC)
    set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES
        "$(VC_IncludePath);$(WindowsSDK_IncludePath);${OptiX_INCLUDE}")
  endif()

  set(RENDER_SOURCES
      ${RENDER_SOURCES_COMMON}
      ${ROOT_HOME}/include/render/Camera.h
      ${ROOT_HOME}/include/render/materials.h
      ${ROOT_HOME}/include/render/Lights.h
      ${ROOT_HOME}/src/render/Camera.cpp
      ${ROOT_HOME}/src/render/optix/OptixBuffer.h
      ${ROOT_HOME}/src/render/optix/OptixBuffer.cpp
      ${ROOT_HOME}/src/render/optix/OptixRender.h
      ${ROOT_HOME}/src/render/optix/OptixRender.cpp)
  set(RENDER_OPTIX_SOURCES ${ROOT_HOME}/include/render/optix/OptixRenderParams.h
                           ${ROOT_HOME}/src/render/optix/OptixRender.cu)
  include_directories(${ROOT_HOME}/include/render)
  include_directories(${ROOT_HOME}/src/render/optix)
  include_directories(${ROOT_HOME})
  # Create the rules to build the PTX and/or OPTIX files.
  cuda_wrap_srcs(
    ${RENDERLIB_NAME}
    OPTIXIR
    generated_files2
    ${RENDER_OPTIX_SOURCES}
    ${cmake_options}
    OPTIONS
    ${options})
  list(APPEND generated_files ${generated_files2})

  set(RENDER_CUDA_SOURCES 
    ${ROOT_HOME}/src/render/optix/postprocessing/Tonemappers.h
    ${ROOT_HOME}/src/render/optix/postprocessing/Tonemappers.cu
  )

  add_library(${RENDERLIB_NAME} STATIC ${RENDER_SOURCES} ${generated_files} ${RENDER_CUDA_SOURCES})

endif()

if(WIN32 OR LINUX)
  target_include_directories(${RENDERLIB_NAME} PUBLIC ${OptiX_INCLUDE})
  target_include_directories(${RENDERLIB_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
  add_custom_command(
    TARGET ${RENDERLIB_NAME}
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${OUTPUT_DIRECTORY}/optix")
  add_custom_command(
    TARGET ${RENDERLIB_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${generated_files}
            "${OUTPUT_DIRECTORY}/optix")

  target_link_libraries(${RENDERLIB_NAME} PUBLIC ${CUDA_LIBRARIES} scene
                                                 materialmanager glm::glm stb::stb)
endif()

if(APPLE)
  set(RENDER_SOURCES
      ${RENDER_SOURCES_COMMON}
      ${ROOT_HOME}/src/render/metal/MetalRender.h
      ${ROOT_HOME}/src/render/metal/MetalRender.cpp
      ${ROOT_HOME}/src/render/metal/MetalBuffer.h
      ${ROOT_HOME}/src/render/metal/MetalBuffer.cpp)

  add_library(${RENDERLIB_NAME} STATIC ${RENDER_SOURCES})

  target_include_directories(${RENDERLIB_NAME}
                             PRIVATE ${ROOT_HOME}/external/metal-cpp/)
  target_include_directories(${RENDERLIB_NAME}
                             PRIVATE ${ROOT_HOME}/external/metal-cpp-extensions/)
  target_link_libraries(${RENDERLIB_NAME} PRIVATE "-framework Foundation")
  target_link_libraries(${RENDERLIB_NAME} PRIVATE "-framework QuartzCore")
  target_link_libraries(${RENDERLIB_NAME} PRIVATE "-framework Metal")
  target_link_libraries(${RENDERLIB_NAME} PRIVATE "-framework MetalKit")
  target_link_libraries(${RENDERLIB_NAME} PRIVATE glm::glm stb::stb)

  add_custom_command(
    TARGET ${RENDERLIB_NAME}
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "${OUTPUT_DIRECTORY}/metal/shaders")

  add_custom_command(
    TARGET ${RENDERLIB_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/src/render/metal/shaders/fullScreen.metal ${OUTPUT_DIRECTORY}/metal/shaders
  )
  
  set(METAL_SOURCES ${ROOT_HOME}/src/render/metal/shaders/pathtrace.metal)
  set(METAL_SOURCES_TM ${ROOT_HOME}/src/render/metal/shaders/tonemapper.metal)

  set(SHADER_OUTPUT_DIR ${OUTPUT_DIRECTORY}/metal/shaders)

# Create a custom command to compile and link pathtrace.metal shader
add_custom_command(
    OUTPUT ${SHADER_OUTPUT_DIR}/pathtrace.metallib
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E echo "Compiling pathtrace.metal shader..."
    COMMAND xcrun -sdk macosx metal -Wall -c ${METAL_SOURCES} -o pathtrace.air -I ${ROOT_HOME}/src/render/metal/shaders
    COMMAND ${CMAKE_COMMAND} -E echo "Linking pathtrace.metal shader..."
    COMMAND xcrun -sdk macosx metallib pathtrace.air -o ${SHADER_OUTPUT_DIR}/pathtrace.metallib
    WORKING_DIRECTORY ${CMAKE_CFG_INTDIR}
    MAIN_DEPENDENCY ${METAL_SOURCES}
    VERBATIM
)

# Create a custom command to compile and link tonemapper.metal shader
add_custom_command(
    OUTPUT ${SHADER_OUTPUT_DIR}/tonemapper.metallib
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E echo "Compiling tonemapper.metal shader..."
    COMMAND xcrun -sdk macosx metal -Wall -c ${METAL_SOURCES_TM} -o tonemapper.air -I ${ROOT_HOME}/src/render/metal/shaders
    COMMAND ${CMAKE_COMMAND} -E echo "Linking tonemapper.metal shader..."
    COMMAND xcrun -sdk macosx metallib tonemapper.air -o ${SHADER_OUTPUT_DIR}/tonemapper.metallib
    WORKING_DIRECTORY ${CMAKE_CFG_INTDIR}
    MAIN_DEPENDENCY ${METAL_SOURCES_TM}
    VERBATIM
)
# Create custom targets that will use the custom commands
add_custom_target(PathtraceShader ALL DEPENDS ${SHADER_OUTPUT_DIR}/pathtrace.metallib)
add_custom_target(TonemapperShader ALL DEPENDS ${SHADER_OUTPUT_DIR}/tonemapper.metallib)

endif(APPLE)

target_link_libraries(${RENDERLIB_NAME} PUBLIC logger)

target_include_directories(${RENDERLIB_NAME} PUBLIC ${ROOT_HOME}/include/)
target_include_directories(${RENDERLIB_NAME} PUBLIC ${ROOT_HOME}/include/render)
target_include_directories(${RENDERLIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${RENDERLIB_NAME} PUBLIC ${ROOT_HOME}/external/mdl-sdk/include)

