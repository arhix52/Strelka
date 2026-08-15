cmake_minimum_required(VERSION 3.22)

find_package(doctest REQUIRED)
find_package(glm REQUIRED)
find_package(imgui REQUIRED)
find_package(imguizmo REQUIRED)

add_executable(unit_tests
    ${CMAKE_CURRENT_SOURCE_DIR}/foundation/test_env.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_bsdf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_energy_conservation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_anisotropy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_dielectric_exit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_thin_walled.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_hair_chiang.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_ior_stack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_diffuse_transmission.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_sample_eval_consistency.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_sheen.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_clearcoat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_iridescence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_vertex_packing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_light_units.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_scene.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sceneloader/test_light_json.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sceneloader/test_curve_sidecar.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sceneloader/test_lod_filter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_pick_ray.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_skinned_bounds.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_camera_animation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_camera_basis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_camera_ortho_zoom.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_transform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_lights.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_rect_light_sampling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_punctual_lights.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_gizmo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_camera_takeover.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_viewport_input.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_viewport_layout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_screenshot_semantics.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_metal_fx_ui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_frame_budget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_editor_document.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_editor_camera_exposure.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_editor_camera_framing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_editor_overlay.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_frame_jitter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_integrator_features.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_texture_cache_key.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_texture_compress.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_ibl_alias_table.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_env_alias_sampling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_integrator_buffer_sizes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_render_resolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_residency_set_diff.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_wavefront_chunk_plan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_wavefront_stage_diagnostic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_optix_denoise_plan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_tonemappers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_scene_stream.cpp
)
strelka_apply_warnings(unit_tests)

target_include_directories(unit_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/render/metal
        # Deliberately CUDA-free headers only, so the logic they hold can be
        # tested on a machine without a GPU: optix_denoise_plan.h (the denoiser's
        # decisions) and env_alias_sampling.h (compiled here for the host and by
        # nvcc for the device, so both run one copy). Including this directory
        # pulls no OptiX header in.
        ${CMAKE_SOURCE_DIR}/src/render/optix
        ${CMAKE_SOURCE_DIR}/src/editor
)

target_link_libraries(unit_tests
    PRIVATE
        doctest::doctest
        strelka_material
        strelka_scene
        strelka_sceneloader
        strelka_display
        glm::glm
        imgui::imgui
        imguizmo::imguizmo
)

# scenes/validation is the asset root the skinned-bounds regression reads when
# STRELKA_SKINNED_GLTF is unset. Compile it in so the test does not depend on
# which directory ctest was launched from.
target_compile_definitions(unit_tests
    PRIVATE
        STRELKA_TEST_ASSETS_DIR="${CMAKE_SOURCE_DIR}/scenes/validation"
)

add_test(NAME unit_tests COMMAND unit_tests)

# The curve sidecar has a writer as well as a reader, and the writer is Python
# that Blender runs -- which is where the diameter/radius convention actually
# went wrong. The C++ suite cannot reach it, so the other end of that contract is
# a stdlib unittest module with no Blender and no pytest dependency. Optional
# only because a build host without a Python interpreter should still build.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME curve_sidecar_writer
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_SOURCE_DIR}/tools/iso_bathroom/test_curve_sidecar.py)
else()
    message(STATUS "No Python3 interpreter: skipping the curve sidecar writer test")
endif()
