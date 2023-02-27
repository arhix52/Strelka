if (NOT RELEASE)
  find_program(CLANGTIDY clang-tidy)
  if (CLANGTIDY)
    message(STATUS "Using clang-tidy")
    set(CMAKE_CXX_CLANG_TIDY ${CLANGTIDY})
  else ()
    message(SEND_ERROR "clang-tidy requested but executable not found")
  endif ()

#   message(STATUS "Using address sanitizer")
#   set(CMAKE_CXX_FLAGS
#     "${CMAKE_CXX_FLAGS} -O0 -fsanitize=address -g")
endif ()
