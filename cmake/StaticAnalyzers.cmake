# clang-tidy as part of the build.
#
# Off by default, deliberately. Running the analyzer on every translation unit
# roughly doubles compile time, and on macOS clang-tidy is not on PATH at all --
# Homebrew's llvm is keg-only and Xcode's toolchain does not ship it -- so a
# default-on switch would turn `./build.sh` into a configure error for anyone who
# has not installed it. Turn it on where the cost is worth paying:
#
#     cmake .. -DSTRELKA_ENABLE_CLANG_TIDY=ON
#
# The checks themselves live in .clang-tidy, with two narrower configs for the
# headers that three different compilers read; see the Conventions section of
# CLAUDE.md. Only bugprone-*, clang-analyzer-*, performance-*, concurrency-* and
# cert-* are errors, so an advisory diagnostic will not stop a build.
option(STRELKA_ENABLE_CLANG_TIDY "Run clang-tidy as part of the build" OFF)

if(STRELKA_ENABLE_CLANG_TIDY)
    # HINTS rather than PATHS: a clang-tidy already on PATH wins, and these are
    # only where Homebrew puts the one it will not link into /usr/local/bin.
    find_program(STRELKA_CLANG_TIDY
        NAMES clang-tidy
        HINTS /opt/homebrew/opt/llvm/bin /usr/local/opt/llvm/bin)

    if(NOT STRELKA_CLANG_TIDY)
        message(FATAL_ERROR
            "STRELKA_ENABLE_CLANG_TIDY=ON but clang-tidy was not found.\n"
            "  macOS:  brew install llvm   (it lands in /opt/homebrew/opt/llvm/bin)\n"
            "  Linux:  apt install clang-tidy\n"
            "Or configure with -DSTRELKA_ENABLE_CLANG_TIDY=OFF to build without it.")
    endif()

    message(STATUS "clang-tidy: ${STRELKA_CLANG_TIDY}")
    # .mm files are compiled as CXX here -- the project enables only that
    # language -- so this one variable covers the Metal backend too.
    # No PARENT_SCOPE: include() does not open a scope, so this lands in the
    # root directory scope and is inherited by every add_subdirectory below it.
    set(CMAKE_CXX_CLANG_TIDY "${STRELKA_CLANG_TIDY}")
endif()

# Call after the subdirectories are added: clears the property on targets built
# from vendored sources.
#
# Not an optimisation -- it is required. Those directories carry a .clang-tidy of
# Checks: '-*', and clang-tidy treats "no checks enabled" as a usage error and
# exits non-zero, which fails the build on a file nobody wanted analysed.
macro(strelka_skip_clang_tidy_on_vendored_targets)
    foreach(vendored_target strelka_vendor_imgui strelka_vendor_file_dialog)
        if(TARGET ${vendored_target})
            set_target_properties(${vendored_target} PROPERTIES CXX_CLANG_TIDY "")
        endif()
    endforeach()
endmacro()
