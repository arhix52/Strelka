option(STRELKA_WARNINGS_AS_ERRORS "Treat project compiler warnings as errors" ON)

function(strelka_apply_warnings target)
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
endfunction()

function(strelka_apply_vendor_warnings target)
    target_compile_options(${target} PRIVATE
        -Wall
        -Wextra
        -Wno-error
        -Wno-deprecated-enum-enum-conversion
        -Wno-arc-bridge-casts-disallowed-in-nonarc
    )
endfunction()
