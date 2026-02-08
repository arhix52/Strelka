cmake_minimum_required(VERSION 3.22)

find_package(doctest REQUIRED)
find_package(glm REQUIRED)

add_executable(unit_tests
    ${CMAKE_CURRENT_SOURCE_DIR}/material/test_bsdf.cpp
)

target_link_libraries(unit_tests
    PRIVATE
        doctest::doctest
        strelka_material
        glm::glm
)

add_test(NAME unit_tests COMMAND unit_tests)
