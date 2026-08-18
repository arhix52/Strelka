cmake_minimum_required(VERSION 3.22)

find_package(imgui REQUIRED)
find_package(imguizmo REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

set(DISPLAY_SOURCES_COMMON
    ${CMAKE_CURRENT_SOURCE_DIR}/include/strelka/display/display.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/strelka/display/output_policy.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/strelka/display/output_probe.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/display.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/display_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/output_probe.cpp
)

if(WIN32 OR LINUX)
    find_package(CUDAToolkit REQUIRED)
    find_package(VulkanLoader REQUIRED)
    find_program(GLSLC_EXECUTABLE glslc REQUIRED)

    set(VULKAN_SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders)
    set(VULKAN_SHADER_OUTPUT_DIR ${OUTPUT_DIRECTORY}/vulkan/shaders)
    set(VULKAN_SHADER_OUTPUTS)
    foreach(shader IN ITEMS imgui_linear.frag final.vert final.frag)
        set(shader_source ${VULKAN_SHADER_SOURCE_DIR}/${shader})
        set(shader_output ${VULKAN_SHADER_OUTPUT_DIR}/${shader}.spv)
        add_custom_command(
            OUTPUT ${shader_output}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${VULKAN_SHADER_OUTPUT_DIR}
            COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.1
                    -o ${shader_output} ${shader_source}
            DEPENDS ${shader_source}
            COMMENT "Compiling Vulkan shader ${shader}"
            VERBATIM
        )
        list(APPEND VULKAN_SHADER_OUTPUTS ${shader_output})
    endforeach()
    add_custom_target(strelka_display_vulkan_shaders ALL
        DEPENDS ${VULKAN_SHADER_OUTPUTS})

    set(IMGUI_IMPL_SOURCES
        ${ROOT_HOME}/external/imgui/imgui_impl_glfw.cpp
        ${ROOT_HOME}/external/imgui/imgui_impl_vulkan.cpp
    )

    set(DISPLAY_SOURCES
        ${DISPLAY_SOURCES_COMMON}
        ${CMAKE_CURRENT_SOURCE_DIR}/vulkan/cuda_vulkan_interop.h
        ${CMAKE_CURRENT_SOURCE_DIR}/vulkan/cuda_vulkan_interop.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/vulkan/glfw_display.h
        ${CMAKE_CURRENT_SOURCE_DIR}/vulkan/glfw_display.cpp
    )

    add_library(strelka_vendor_imgui OBJECT ${IMGUI_IMPL_SOURCES})
    strelka_apply_vendor_warnings(strelka_vendor_imgui)
    target_include_directories(strelka_vendor_imgui PRIVATE ${ROOT_HOME}/external/imgui/)
    target_link_libraries(strelka_vendor_imgui PRIVATE imgui::imgui glfw Vulkan::Loader)

    add_library(strelka_display STATIC ${DISPLAY_SOURCES})
    add_dependencies(strelka_display strelka_display_vulkan_shaders)
    strelka_apply_warnings(strelka_display)
    target_compile_definitions(strelka_display PRIVATE STRELKA_MATERIAL_CUDA_HOST)

    target_include_directories(strelka_display
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${ROOT_HOME}/external/imgui/
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src/shaders/optix
    )

    target_link_libraries(strelka_display
        PRIVATE
            strelka_vendor_imgui
        PUBLIC
            strelka_foundation
            strelka_render
            imgui::imgui
            imguizmo::imguizmo
            glfw
            glm::glm
            Vulkan::Loader
            CUDA::cudart
    )
endif()

if(APPLE)
    set(IMGUI_IMPL_SOURCES
        ${ROOT_HOME}/external/imgui/imgui_impl_glfw.cpp
        ${ROOT_HOME}/external/imgui/imgui_impl_metal.mm
    )

    set(DISPLAY_SOURCES
        ${DISPLAY_SOURCES_COMMON}
        ${CMAKE_CURRENT_SOURCE_DIR}/metal/glfw_display.h
        ${CMAKE_CURRENT_SOURCE_DIR}/metal/glfw_display.mm
    )

    add_library(strelka_vendor_imgui OBJECT ${IMGUI_IMPL_SOURCES})
    strelka_apply_vendor_warnings(strelka_vendor_imgui)
    target_include_directories(strelka_vendor_imgui PRIVATE ${ROOT_HOME}/external/imgui/)
    target_link_libraries(strelka_vendor_imgui PRIVATE imgui::imgui glfw)

    add_library(strelka_display STATIC ${DISPLAY_SOURCES})
    strelka_apply_warnings(strelka_display)

    target_include_directories(strelka_display
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${ROOT_HOME}/external/imgui/
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_include_directories(strelka_display
        SYSTEM PUBLIC
            ${ROOT_HOME}/third_party/metal_cpp/
    )

    target_link_libraries(strelka_display
        PRIVATE
            strelka_vendor_imgui
            "-framework Foundation"
            "-framework QuartzCore"
            "-framework CoreGraphics"
            "-framework Metal"
            "-framework MetalKit"
    )
    target_link_libraries(strelka_display
        PUBLIC
            strelka_foundation
            strelka_render
            imgui::imgui
            glfw
            glm::glm
    )
endif()
