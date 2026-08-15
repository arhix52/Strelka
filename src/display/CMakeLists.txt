cmake_minimum_required(VERSION 3.22)

find_package(imgui REQUIRED)
find_package(imguizmo REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

set(DISPLAY_SOURCES_COMMON
    ${CMAKE_CURRENT_SOURCE_DIR}/include/strelka/display/display.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/display.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/display_factory.cpp
)

if(WIN32 OR LINUX)
    find_package(glad REQUIRED)
    find_package(CUDAToolkit REQUIRED)

    set(IMGUI_IMPL_SOURCES
        ${ROOT_HOME}/external/imgui/imgui_impl_glfw.cpp
        ${ROOT_HOME}/external/imgui/imgui_impl_opengl3.cpp
    )

    set(DISPLAY_SOURCES
        ${DISPLAY_SOURCES_COMMON}
        ${CMAKE_CURRENT_SOURCE_DIR}/opengl/glfw_display.h
        ${CMAKE_CURRENT_SOURCE_DIR}/opengl/glfw_display.cpp
    )

    add_library(strelka_vendor_imgui OBJECT ${IMGUI_IMPL_SOURCES})
    strelka_apply_vendor_warnings(strelka_vendor_imgui)
    target_include_directories(strelka_vendor_imgui PRIVATE ${ROOT_HOME}/external/imgui/)
    target_link_libraries(strelka_vendor_imgui PRIVATE imgui::imgui glfw)

    add_library(strelka_display STATIC ${DISPLAY_SOURCES})
    strelka_apply_warnings(strelka_display)

    # The OpenGL display does CUDA graphics interop, so its translation units see
    # CUDA's vector types alongside the shared material headers. The two have to
    # agree on which float3 they mean -- see the note in material_params.h. PRIVATE
    # so it does not leak to unit_tests, which links this library but wants GLM.
    target_compile_definitions(strelka_display PRIVATE STRELKA_MATERIAL_CUDA_HOST)

    target_include_directories(strelka_display
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${ROOT_HOME}/external/imgui/
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(strelka_display
        PRIVATE
            strelka_vendor_imgui
        PUBLIC
            ${OPENGL_LIBRARIES}
            strelka_foundation
            strelka_render
            imgui::imgui
            imguizmo::imguizmo
            glfw
            glm::glm
            glad::glad
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
