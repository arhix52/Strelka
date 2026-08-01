cmake_minimum_required(VERSION 3.22)

find_package(doctest REQUIRED)
find_package(glm REQUIRED)
find_package(imgui REQUIRED)
find_package(imguizmo REQUIRED)

add_executable(unit_tests
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_bsdf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_scene.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sceneloader/test_light_json.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_pick_ray.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/test_skinned_bounds.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editor/test_gizmo.cpp
)

target_link_libraries(unit_tests
    PRIVATE
        doctest::doctest
        strelka_material
        strelka_scene
        strelka_sceneloader
        glm::glm
        imgui::imgui
        imguizmo::imguizmo
)

add_test(NAME unit_tests COMMAND unit_tests)
