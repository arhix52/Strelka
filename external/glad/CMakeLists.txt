cmake_minimum_required(VERSION 3.16)
project(glad LANGUAGES C)
enable_language(C)

find_package(OpenGL REQUIRED)

add_library(glad STATIC
     #glad.h
     #glad.c
   include/KHR/khrplatform.h
   include/glad/gl.h
   src/gl.c
)
set_target_properties(glad PROPERTIES LINKER_LANGUAGE C)

set_property( TARGET glad PROPERTY C_STANDARD 99 )

target_compile_definitions(glad
  PRIVATE GLAD_GLAPI_EXPORT_BUILD
  PUBLIC  GLAD_GLAPI_EXPORT)
target_include_directories(glad PUBLIC .)
target_include_directories(glad PUBLIC ./include)
target_link_libraries(glad PUBLIC ${OPENGL_LIBRARIES})
