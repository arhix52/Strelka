cmake_minimum_required(VERSION 3.20)
project(StrelkaOptiX LANGUAGES CXX CUDA)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

if(WIN32)
  add_compile_definitions(NOMINMAX)
  add_compile_definitions(_USE_MATH_DEFINES)
  add_compile_definitions(_ALLOW_ITERATOR_DEBUG_LEVEL_MISMATCH)
endif()

find_package( OpenGL REQUIRED )
# Find at least a 5.0 version of CUDA.
find_package(CUDA 5.0 REQUIRED)
message(STATUS "CUDA Include DIR: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
# Search for the OptiX libraries and include files.
set(OptiX_INSTALL_DIR $ENV{OPTIX_DIR})
message(STATUS "OptiX SDK DIR: ${OptiX_INSTALL_DIR}")
# Search for the OptiX libraries and include files.
find_package(OptiX REQUIRED)
message(STATUS "OptiX Include DIR: ${OptiX_INCLUDE}")
include_directories(${OptiX_INCLUDE})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Hack to get intellisense working for CUDA includes
if (MSVC)
  set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES "$(VC_IncludePath);$(WindowsSDK_IncludePath);${OptiX_INCLUDE}")
endif()

add_subdirectory(external/glad)
set_target_properties(glad PROPERTIES LINKER_LANGUAGE CXX)

set(ROOT_HOME ${CMAKE_CURRENT_LIST_DIR})
set(OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(src/Settings)
add_subdirectory(src/display)
add_subdirectory(src/render)
add_subdirectory(src/hdStrelka)
add_subdirectory(src/scene)
add_subdirectory(src/materialmanager)
add_subdirectory(src/app)
add_subdirectory(src/hdRunner)

add_subdirectory(tests)


# # Application
# set(STRELKA_SOURCES
#   include/optixTriangle.h
#   src/main.cpp
# )

# set(STRELKA_OPTIX_SOURCES 
#     #include/optixTriangle.h
#     src/optixTriangle.cu
# )

# # Create the rules to build the PTX and/or OPTIX files.
# CUDA_WRAP_SRCS( ${PROJECT_NAME} OPTIXIR generated_files2 ${STRELKA_OPTIX_SOURCES} ${cmake_options} OPTIONS ${options} )
# list(APPEND generated_files ${generated_files2})

# set_source_files_properties(${STRELKA_SOURCES} PROPERTIES LANGUAGE CUDA)
# add_executable(${PROJECT_NAME} 
#   ${STRELKA_SOURCES}
#   ${generated_files}
# )
# set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "72")

# target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include/)
# target_include_directories(${PROJECT_NAME} PUBLIC ${OptiX_INCLUDE})
# #target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
# #target_include_directories(${PROJECT_NAME} PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
# if( WIN32 )
#   target_compile_definitions( ${PROJECT_NAME} PUBLIC GLAD_GLAPI_EXPORT )
# endif()

# target_include_directories(${PROJECT_NAME} PUBLIC external/stb)
# target_include_directories(${PROJECT_NAME} PUBLIC external/glm)
# target_include_directories(${PROJECT_NAME} PUBLIC external/glfw/include)
# target_include_directories(${PROJECT_NAME} PUBLIC external/glad/include)

# target_link_libraries(${PROJECT_NAME} PUBLIC 
#     "${CMAKE_CURRENT_SOURCE_DIR}/external/glfw/glfw3.lib"
#     ${OPENGL_LIBRARIES}
#     ${CUDA_LIBRARIES}
#     PRIVATE
#     glad
# )
