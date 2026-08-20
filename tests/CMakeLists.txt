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
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_opaque_back_face.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_valid_reflection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_thin_walled.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_hair_chiang.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_ior_stack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_diffuse_transmission.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_sample_eval_consistency.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_smooth_lobe.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_sheen.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_clearcoat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_iridescence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_vertex_packing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_light_units.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_scene.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sceneloader/test_light_json.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sceneloader/test_ies_loader.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_camera_smoothing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_viewport_input.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_viewport_layout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_screenshot_semantics.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_denoiser_ui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_frame_budget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_editor_document.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_editor_camera_exposure.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_editor_camera_framing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_editor_overlay.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_camera_dump.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/display/test_output_policy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/display/test_output_probe.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_frame_jitter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_integrator_features.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_texture_cache_key.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_texture_compress.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_ibl_alias_table.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_env_alias_sampling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_env_map_math.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_integrator_buffer_sizes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_render_resolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_residency_set_diff.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_wavefront_chunk_plan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_wavefront_stage_diagnostic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_optix_denoise_plan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_tonemappers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_hdr_output.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_scene_stream.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_fibre_chord.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_nee_pairing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_light_pdf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_projector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_texture_transform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_texture_upload_plan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_accel_build_policy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_ies_pack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_ies_math.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_opacity_micromap_policy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_sharc_grid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_curve_layout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_medium_walk.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_optix_scene_preparation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_gpu_stage_breadcrumb.cpp
)
strelka_apply_warnings(unit_tests)

# GCC 15.2 miscompiles `float x = fabsf(<a glm::dot prvalue>)` to a literal 0.0f.
#
# It is the unqualified C spelling only -- `std::fabs` is right, and binding the
# dot product to a named float before taking its magnitude is right, which is
# exactly why nobody saw it: the wrong form appears inside one expression. Only
# `fabsf` of the C math builtins misbehaves; fmaxf, sqrtf, powf and the rest all
# return the correct value on the same argument. clang 12 is correct at every
# optimisation level, and so is GCC with this flag, so it is the compiler and not
# the model. scratch-free reproduction, no Strelka headers involved:
#
#     glm::vec3 a(0,-1,0), b(0.063,-0.995,0.082);
#     float u = fabsf(glm::dot(a, b));   // GCC 15.2: 0.0   clang: 0.994658
#
# The BSDF headers spell it `fabsf(dot(...))` in dozens of places -- the rough
# transmission pdf is built out of three of them -- so on this compiler the host
# build of the material model was returning a zero pdf for directions its own
# sampler had just produced, and `tests/material/test_dielectric_exit.cpp` and
# `test_diffuse_transmission.cpp` were reporting that as a model defect.
#
# Scoped to this target because unit_tests is the only host consumer of the BSDF
# headers; the device builds are unaffected, since neither the CUDA nor the Metal
# branch of material_math.h uses GLM's vector types. Guarded on GNU so an
# AppleClang build of the same tests is untouched.
target_compile_options(unit_tests PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-fno-builtin-fabsf>)

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
        # <shading/...> -- the two OptiX shading headers that carry no CUDA at
        # all, so the closest-hit program and these tests compile the same code.
        ${CMAKE_SOURCE_DIR}/src/shaders/optix
)

# strelka_sceneloader PUBLIC-links strelka_scene and strelka_display PUBLIC-links
# imgui and imguizmo, so naming any of them again here only put the archive on
# the link line twice.
target_link_libraries(unit_tests
    PRIVATE
        doctest::doctest
        strelka_material
        strelka_sceneloader
        strelka_display
        glm::glm
)

# scenes/validation is the asset root the skinned-bounds regression reads when
# STRELKA_SKINNED_GLTF is unset. Compile it in so the test does not depend on
# which directory ctest was launched from.
target_compile_definitions(unit_tests
    PRIVATE
        STRELKA_TEST_ASSETS_DIR="${CMAKE_SOURCE_DIR}/scenes/validation"
)

add_test(NAME unit_tests COMMAND unit_tests)

if(LINUX AND TARGET strelka_display)
    add_executable(vulkan_cuda_interop_smoke
        ${CMAKE_CURRENT_SOURCE_DIR}/display/vulkan_cuda_interop_smoke.cpp
    )
    strelka_apply_warnings(vulkan_cuda_interop_smoke)
    target_include_directories(vulkan_cuda_interop_smoke
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/display/vulkan
            ${CMAKE_SOURCE_DIR}/src/shaders/optix
    )
    target_link_libraries(vulkan_cuda_interop_smoke
        PRIVATE
            strelka_display
            strelka_shaders
    )
    add_test(NAME vulkan_cuda_interop_smoke
             COMMAND vulkan_cuda_interop_smoke)
    set_tests_properties(vulkan_cuda_interop_smoke
        PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 60
    )
endif()

# One rendered frame, put through the display transform.
#
# Separate from unit_tests because it needs a device and a built StrelkaCLI, and
# a suite that cannot run without a GPU is a suite that stops being run. It skips
# rather than fails when either is missing.
if(TARGET StrelkaCLI)
    find_package(tinyexr REQUIRED)
    add_executable(cornell_display_regression
        ${CMAKE_CURRENT_SOURCE_DIR}/render/cornell_display_regression.cpp
    )
    strelka_apply_warnings(cornell_display_regression)
    target_compile_definitions(cornell_display_regression
        PRIVATE
            STRELKA_TEST_ASSETS_DIR="${CMAKE_SOURCE_DIR}/scenes/validation"
            STRELKA_TEST_BINARY_DIR="${OUTPUT_DIRECTORY}"
    )
    target_include_directories(cornell_display_regression
        PRIVATE ${CMAKE_SOURCE_DIR}/src/shaders/common
    )
    target_link_libraries(cornell_display_regression PRIVATE tinyexr::tinyexr glm::glm)
    add_dependencies(cornell_display_regression StrelkaCLI)
    add_test(NAME cornell_display_regression COMMAND cornell_display_regression)
    set_tests_properties(cornell_display_regression
        PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 300
    )
endif()

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
