# clang-tidy as part of the build.
#
# Off by default, and the reason is the cost rather than the value. Measured on
# this tree: EditorApp.cpp compiles in 7.4 s and takes 61 s with the analysis in
# front of it, so 88% of what an incremental build costs is clang-tidy re-reading
# a file whose diagnostics nobody is waiting for. A clean build goes from 20 s to
# 112 s the same way. That is paid on every edit of the write-build-test loop,
# which is most of what working on this tree consists of.
#
# Nothing is given up by not paying it there, because it is not the gate. The
# gate is .githooks/pre-commit, which runs clang-tidy over the staged sources --
# the same checks, on exactly the files being committed, at the one moment the
# answer has to be right. tools/run_clang_tidy.sh does the whole tree when that
# is what is wanted. It was worth turning on: the run that first did caught three
# leaked MetalFX descriptors and an out-of-bounds read. It is still on, one step
# later.
#
#     cmake .. -DSTRELKA_ENABLE_CLANG_TIDY=ON      # analyse every build
#
# Only bugprone-*, clang-analyzer-*, performance-*, concurrency-* and cert-* are
# errors, so advisory diagnostics do not stop a build either way. The checks
# themselves live in .clang-tidy, with two narrower configs for the headers that
# three different compilers read; see the Conventions section of CLAUDE.md.
option(STRELKA_ENABLE_CLANG_TIDY "Run clang-tidy as part of the build" OFF)

if(STRELKA_ENABLE_CLANG_TIDY)
    # HINTS rather than PATHS: a clang-tidy already on PATH wins, and these are
    # only where Homebrew puts the one it will not link into /usr/local/bin.
    find_program(STRELKA_CLANG_TIDY
        NAMES clang-tidy
        HINTS /opt/homebrew/opt/llvm/bin /usr/local/opt/llvm/bin)

    if(NOT STRELKA_CLANG_TIDY)
        message(WARNING
            "clang-tidy was asked for and not found, so this build has no static analysis.\n"
            "  macOS:  brew install llvm     (lands in /opt/homebrew/opt/llvm/bin, which\n"
            "                                 this file searches; no PATH change needed)\n"
            "  Linux:  apt install clang-tidy\n"
            "The pre-commit hook needs it too, and will refuse to pass without it.")
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
