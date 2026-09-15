option(STRELKA_WARNINGS_AS_ERRORS "Treat project compiler warnings as errors" ON)

function(strelka_apply_warnings target)
    if(MSVC)
        # The tree is developed on clang; MSVC's /WX on /W4 surfaces many
        # benign size_t/uint32_t conversions that are not worth fixing per-platform.
        target_compile_options(${target} PRIVATE /W3 /utf-8)
    else()
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wshadow
            -Wnon-virtual-dtor
            -Wunused
            -Wpedantic
        )
        if(STRELKA_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endfunction()

function(strelka_apply_vendor_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W3 /utf-8)
    else()
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wno-error
            -Wno-deprecated-enum-enum-conversion
            -Wno-arc-bridge-casts-disallowed-in-nonarc
        )
    endif()
endfunction()
