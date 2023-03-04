cmake_minimum_required(VERSION 3.24)
set(LOGLIB_NAME logger)
set(LOGLIB_SOURCES
    ${ROOT_HOME}/include/log/log.h 
    ${ROOT_HOME}/include/log/logmanager.h
    ${ROOT_HOME}/src/log/logmanager.cpp)

find_package(spdlog REQUIRED)

add_library(${LOGLIB_NAME} STATIC ${LOGLIB_SOURCES})
target_include_directories(${LOGLIB_NAME} PUBLIC ${ROOT_HOME}/include/log)
# target_include_directories(${LOGLIB_NAME} PUBLIC spdlog::spdlog)

target_link_libraries(${LOGLIB_NAME} PUBLIC spdlog::spdlog)
