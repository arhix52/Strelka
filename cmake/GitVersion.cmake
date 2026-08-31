# Sets STRELKA_VERSION to `git describe` (falls back to the short commit hash
# when there are no tags, and to "unknown" outside a git checkout -- a source
# tarball or a shallow CI clone with no history). Read once at configure time:
# good enough for an About dialog, not meant to track uncommitted edits made
# after the last cmake run without a reconfigure.
find_package(Git QUIET)

set(STRELKA_VERSION "unknown")
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE STRELKA_GIT_DESCRIBE
        RESULT_VARIABLE STRELKA_GIT_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(STRELKA_GIT_RESULT EQUAL 0 AND STRELKA_GIT_DESCRIBE)
        set(STRELKA_VERSION "${STRELKA_GIT_DESCRIBE}")
    endif()
endif()

message(STATUS "Strelka version: ${STRELKA_VERSION}")
