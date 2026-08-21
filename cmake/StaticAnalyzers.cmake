# clang-tidy as part of the build.
#
# On by default, by explicit choice, and the cost is not small: measured on this
# tree, a clean build goes from 20s to 112s -- 5.6x, not the "roughly doubles"
# that gets said about clang-tidy. Incremental builds pay it per changed file.
#
# It is worth it: the run that first turned this on caught three leaked MetalFX
# descriptors and an out-of-bounds read. Only bugprone-*, clang-analyzer-*,
# performance-*, concurrency-* and cert-* are errors, so advisory diagnostics
# still do not stop a build.
#
# A missing clang-tidy is a warning and not an error, and that is on purpose: the
# tool is not installed everywhere -- Xcode does not ship it and Homebrew's llvm
# is keg-only -- and a default-on switch that turns `./build.sh` into a configure
# failure on a fresh machine would get itself turned off permanently within a
# day. Where it is present it runs; where it is not, the build says so loudly and
# carries on.
#
#     cmake .. -DSTRELKA_ENABLE_CLANG_TIDY=OFF     # to skip it deliberately
#
# The checks themselves live in .clang-tidy, with two narrower configs for the
# headers that three different compilers read; see the Conventions section of
# CLAUDE.md.
option(STRELKA_ENABLE_CLANG_TIDY "Run clang-tidy as part of the build" ON)

if(STRELKA_ENABLE_CLANG_TIDY)
    # HINTS rather than PATHS: a clang-tidy already on PATH wins, and these are
    # only where Homebrew puts the one it will not link into /usr/local/bin.
    find_program(STRELKA_CLANG_TIDY
        NAMES clang-tidy
        HINTS /opt/homebrew/opt/llvm/bin /usr/local/opt/llvm/bin)

    if(NOT STRELKA_CLANG_TIDY)
        message(WARNING
            "clang-tidy was not found, so this build has NO static analysis.\n"
            "Installing it is recommended -- it is what guards the tree against the bug,\n"
            "UB and performance classes that are build errors here:\n"
            "  macOS:  brew install llvm     (lands in /opt/homebrew/opt/llvm/bin, which\n"
            "                                 this file searches; no PATH change needed)\n"
            "  Linux:  apt install clang-tidy\n"
            "To build without it deliberately, and without this warning, configure with\n"
            "-DSTRELKA_ENABLE_CLANG_TIDY=OFF.")
    endif()
endif()

if(STRELKA_ENABLE_CLANG_TIDY AND STRELKA_CLANG_TIDY)
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
    # MaterialX brings its own targets and its own .clang-tidy, and running ours
    # over them fails outright rather than merely reporting: the two configs
    # disagree about which checks exist, and clang-tidy exits non-zero on "no
    # checks enabled". Vendored code is not ours to analyse either way.
    foreach(vendored_target
            strelka_vendor_imgui
            strelka_vendor_file_dialog
            MaterialXCore
            MaterialXFormat
            MaterialXGenShader
            MaterialXRender)
        if(TARGET ${vendored_target})
            set_target_properties(${vendored_target} PROPERTIES CXX_CLANG_TIDY "")
        endif()
    endforeach()
endmacro()
