cmake_minimum_required(VERSION 3.20)

find_package(MaterialX REQUIRED)

set(MATERIALLIB_NAME materialmanager)

# Material Manager
set(MATERIALMANAGER_SOURCES
    ${ROOT_HOME}/include/materialmanager/materialmanager.h
    ${ROOT_HOME}/include/materialmanager/mdlNeurayLoader.h
    ${ROOT_HOME}/include/materialmanager/mdlRuntime.h
    ${ROOT_HOME}/include/materialmanager/mdlPtxCodeGen.h
    ${ROOT_HOME}/include/materialmanager/mtlxMdlCodeGen.h
    ${ROOT_HOME}/include/materialmanager/mdlLogger.h
    ${ROOT_HOME}/include/materialmanager/mdlMaterialCompiler.h
    ${ROOT_HOME}/src/materialmanager/materialmanager.cpp
    ${ROOT_HOME}/src/materialmanager/mdlNeurayLoader.cpp
    ${ROOT_HOME}/src/materialmanager/mdlMaterialCompiler.cpp
    ${ROOT_HOME}/src/materialmanager/mdlRuntime.cpp
    ${ROOT_HOME}/src/materialmanager/mdlPtxCodeGen.cpp
    ${ROOT_HOME}/src/materialmanager/mtlxMdlCodeGen.cpp
    ${ROOT_HOME}/src/materialmanager/mdlLogger.cpp)

include_directories(${ROOT_HOME}/external/mdl-sdk/include)

add_library(${MATERIALLIB_NAME} STATIC ${MATERIALMANAGER_SOURCES})
target_include_directories(${MATERIALLIB_NAME}
                           PRIVATE ${ROOT_HOME}/external/glm)
target_include_directories(
  ${MATERIALLIB_NAME}
  PUBLIC ${ROOT_HOME}/include/materialmanager
  PRIVATE ${ROOT_HOME}/include/render/)
target_link_libraries(
  ${MATERIALLIB_NAME} PRIVATE logger MaterialXCore MaterialXFormat
                              MaterialXGenMdl)

add_custom_command(
  TARGET ${MATERIALLIB_NAME}
  PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "${OUTPUT_DIRECTORY}/optix/shaders")

if(WIN32)
  find_program(clang_PATH clang)
  message(STATUS "clang_PATH: ${clang_PATH}")

  set(OptiX_INSTALL_DIR $ENV{OPTIX_DIR})
  message(STATUS "OptiX SDK DIR: ${OptiX_INSTALL_DIR}")
  find_package(OptiX REQUIRED)
  message(STATUS "OptiX Include DIR: ${OptiX_INCLUDE}")
  include_directories(${OptiX_INCLUDE})

  add_custom_command(
    TARGET ${MATERIALLIB_NAME}
    PRE_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E copy
      "${ROOT_HOME}/external/mdl-sdk/nt-x86-64/lib/libmdl_sdk.dll"
      "${OUTPUT_DIRECTORY}")
  add_custom_command(
    TARGET ${MATERIALLIB_NAME}
    PRE_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E copy
      "${ROOT_HOME}/external/mdl-sdk/nt-x86-64/lib/nv_freeimage.dll"
      "${OUTPUT_DIRECTORY}")

  set(_CLANG_FLAGS ${_CLANG_FLAGS} --cuda-path=${CUDA_TOOLKIT_ROOT_DIR})

  # generate bytecode for closest hit radiance shader Note: the
  # _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH define is needed to allow
  # compilation with Clang 7 and VS2019 headers
  add_custom_command(
    TARGET ${MATERIALLIB_NAME}
    PRE_BUILD
    # OUTPUT OptixRender_radiance_closest_hit.d
    # OptixRender_radiance_closest_hit.bc Target: Ada: sm_89 - unsupported by
    # clang 12 Ampere: sm_86 unsupported by clang 12 Turing: sm_75
    COMMAND ${CMAKE_COMMAND} -E echo
            "Compile OptixRender_radiance_closest_hit bytecode using clang..."
    COMMAND
      ${clang_PATH} -I ${OptiX_INCLUDE} -I ${ROOT_HOME}/external/mdl-sdk/include
      -I ${ROOT_HOME}/src/render/ -I ${ROOT_HOME}/include/render -I
      ${ROOT_HOME}/external/glm -I ${ROOT_HOME} ${_CLANG_FLAGS}
      -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -emit-llvm -c -O3 -ffast-math
      -fcuda-flush-denormals-to-zero -fno-vectorize --cuda-gpu-arch=sm_75
      --cuda-device-only
      ${ROOT_HOME}/src/render/optix/OptixRender_radiance_closest_hit.cu -o
      ${OUTPUT_DIRECTORY}/optix/shaders/OptixRender_radiance_closest_hit.bc -MD
      -MT ${OUTPUT_DIRECTORY}/optix/shaders/OptixRender_radiance_closest_hit.bc
      -MP -MF
      ${OUTPUT_DIRECTORY}/optix/shaders/OptixRender_radiance_closest_hit.d
    WORKING_DIRECTORY ${CMAKE_CFG_INTDIR}
    DEPENDS "OptixRender_radiance_closest_hit.cu" ${PROJECT_SOURCES} VERBATIM)
endif(WIN32)
