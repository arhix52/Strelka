# Compiler cache, when there is one.
#
# The loop this is for is edit, build, test, and often revert and build again:
# an A/B measurement compiles one tree, the other, then the first one back. The
# third of those three is free with a cache and is a full rebuild without one,
# and the files this tree is built from are not small -- EditorApp.cpp and
# OptixRender.cpp are each around a minute on their own.
#
# Found or not found, never required: a machine without ccache builds exactly as
# it did before, which is also what keeps this from becoming the reason a fresh
# checkout does not configure.
#
# Note what it does *not* cover. clang-tidy runs as a separate process per
# translation unit (see cmake/StaticAnalyzers.cmake) and its result is not
# cached, so a cached rebuild still pays for the analysis. Turn that off with
# -DSTRELKA_ENABLE_CLANG_TIDY=OFF when iterating, and let the pre-commit hook
# run it over what is actually being committed.
option(STRELKA_ENABLE_CCACHE "Use ccache for C++ compilation when it is installed" ON)

if(STRELKA_ENABLE_CCACHE)
    find_program(STRELKA_CCACHE NAMES ccache)
    if(STRELKA_CCACHE)
        message(STATUS "ccache: ${STRELKA_CCACHE}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${STRELKA_CCACHE}")
        set(CMAKE_C_COMPILER_LAUNCHER "${STRELKA_CCACHE}")
    endif()
endif()
