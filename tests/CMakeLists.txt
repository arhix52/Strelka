cmake_minimum_required(VERSION 3.20)

include(CTest)
find_package(doctest REQUIRED)

set(TESTFILES
        tests_main.cpp
        materialmanager/test_materialmanager.cpp
        )

set(TEST_MAIN unit_tests)
set(TEST_RUNNER_PARAMS "")

find_package(USD REQUIRED HINTS ${USD_DIR} NAMES pxr)
message(STATUS "USD LIBRARY: ${USD_DIR}")
find_package(MaterialX REQUIRED)

add_executable(${TEST_MAIN} ${TESTFILES})
target_link_libraries(${TEST_MAIN} 
    PUBLIC
    doctest::doctest
    materialmanager
    )

target_include_directories(${TEST_MAIN} PRIVATE ${ROOT_HOME}/include/)

set_target_properties(${TEST_MAIN} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        )

add_test(
        NAME ${RENDERLIB_NAME}.${TEST_MAIN}
        COMMAND ${TEST_MAIN} ${TEST_RUNNER_PARAMS})

