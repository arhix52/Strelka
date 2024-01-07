cmake_minimum_required(VERSION 3.20)

# packages
find_package(cxxopts REQUIRED)
find_package(GLM REQUIRED)
include(FindVulkan)
find_package(Vulkan REQUIRED)

if(Vulkan_FOUND)
  set(XR_USE_GRAPHICS_API_VULKAN TRUE)
add_definitions(-DXR_USE_GRAPHICS_API_VULKAN)
  message(STATUS "Enabling Vulkan support")
elseif(BUILD_ALL_EXTENSIONS)
  message(FATAL_ERROR "Vulkan headers not found")
endif()


if($ENV{VULKAN_SDK})
file(TO_CMAKE_PATH "$ENV{VULKAN_SDK}" VULKAN_SDK)
file(
    GLOB
    glslc_folders
    CONFIGURE_DEPENDS
    ${VULKAN_SDK}/*
)
endif()
find_program(
GLSL_COMPILER glslc
PATHS ${glslc_folders}
HINTS "${Vulkan_GLSLC_EXECUTABLE}"
)

find_program(
    GLSLANG_VALIDATOR glslangValidator
    HINTS "${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}"
)

if(GLSL_COMPILER)
    message(STATUS "Found glslc: ${GLSL_COMPILER}")
elseif(GLSLANG_VALIDATOR)
    message(STATUS "Found glslangValidator: ${GLSLANG_VALIDATOR}")
else()
    message(STATUS "Could NOT find glslc, using precompiled .spv files")
endif()

set(HELPER_FOLDER "Helpers")

function(compile_glsl run_target_name)
    set(glsl_output_files "")
    foreach(in_file IN LISTS ARGN)
        get_filename_component(glsl_stage "${in_file}" NAME_WE)
        set(out_file "${CMAKE_CURRENT_BINARY_DIR}/${glsl_stage}.spv")
        # if(GLSL_COMPILER)
        #     # Run glslc if we can find it
        #     add_custom_command(
        #         OUTPUT "${out_file}"
        #         COMMAND
        #             "${GLSL_COMPILER}" -mfmt=c -fshader-stage=${glsl_stage}
        #             "${in_file}" -o "${out_file}"
        #         DEPENDS "${in_file}"
        #         WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        #         VERBATIM
        #     )
        # else
        if(GLSLANG_VALIDATOR)
            # Run glslangValidator if we can find it
            add_custom_command(
                OUTPUT "${out_file}"
                COMMAND
                    "${GLSLANG_VALIDATOR}" -V -S ${glsl_stage} "${in_file}" 
                    -o "${out_file}"
                DEPENDS "${in_file}"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                VERBATIM
            )
        else()
            # Use the precompiled .spv files
            get_filename_component(glsl_src_dir "${in_file}" DIRECTORY)
            set(precompiled_file "${glsl_src_dir}/${glsl_stage}.spv")
            configure_file("${precompiled_file}" "${out_file}" COPYONLY)
        endif()
        list(APPEND glsl_output_files "${out_file}")
    endforeach()
    add_custom_target(${run_target_name} ALL DEPENDS ${glsl_output_files})
    set_target_properties(${run_target_name} PROPERTIES FOLDER ${HELPER_FOLDER})

endfunction()

set(XR_RUNNER_NAME XrRunner)

include(${ROOT_HOME}/external/openxr_loader_windows-1.0.32/OpenXRConfig.cmake)

# Application
set(XR_RUNNER_SOURCES
    ${ROOT_HOME}/src/XrRunner/OpenXRProgram.h
    ${ROOT_HOME}/src/XrRunner/OpenXRProgram.cpp
    ${ROOT_HOME}/src/XrRunner/main.cpp )

set(VULKAN_SHADERS shaders/frag.glsl shaders/vert.glsl)

add_executable(${XR_RUNNER_NAME} ${XR_RUNNER_SOURCES} ${VULKAN_SHADERS})

source_group("Shaders" FILES ${VULKAN_SHADERS})

compile_glsl(xr_glsl_compiles ${VULKAN_SHADERS})
add_dependencies(${XR_RUNNER_NAME} xr_glsl_compiles)

set_target_properties(
  ${XR_RUNNER_NAME}
  PROPERTIES LINKER_LANGUAGE CXX
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED TRUE
             CXX_EXTENSIONS OFF)

target_include_directories(${XR_RUNNER_NAME} PUBLIC ${ROOT_HOME}/include/)
target_include_directories(${XR_RUNNER_NAME} PUBLIC ${ROOT_HOME}/src/XrRunner/)
target_include_directories(${XR_RUNNER_NAME} PRIVATE ${Vulkan_INCLUDE_DIRS})

target_link_libraries(
  ${XR_RUNNER_NAME}
  PUBLIC cxxopts::cxxopts
  scene
  settings
  render
  logger
  OpenXR::openxr_loader
  ${Vulkan_LIBRARIES}
)
