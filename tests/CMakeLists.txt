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
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_pick_ray.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_skinned_bounds.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_camera_animation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_transform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_lights.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_punctual_lights.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_gizmo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_camera_takeover.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_viewport_input.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_editor_document.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_frame_jitter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_integrator_features.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_texture_cache_key.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_ibl_alias_table.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/render/test_integrator_buffer_sizes.cpp
)

target_include_directories(unit_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/render/metal
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

add_test(NAME unit_tests COMMAND unit_tests)
