#pragma once

#include <log/log.h>
#include <log/logmanager.h>

#include <openxr/openxr_reflection.h>

// original: https://github.com/KhronosGroup/OpenXR-SDK-Source/blob/main/src/tests/hello_xr/common.h
// Macro to generate stringify functions for OpenXR enumerations based data provided in openxr_reflection.h
// clang-format off
#define ENUM_CASE_STR(name, val) case name: return #name;
#define MAKE_TO_STRING_FUNC(enumType)                  \
    inline const char* to_string(enumType e) {         \
        switch (e) {                                   \
            XR_LIST_ENUM_##enumType(ENUM_CASE_STR)     \
            default: return "Unknown " #enumType;      \
        }                                              \
    }
// clang-format on

MAKE_TO_STRING_FUNC(XrReferenceSpaceType);
MAKE_TO_STRING_FUNC(XrViewConfigurationType);
MAKE_TO_STRING_FUNC(XrEnvironmentBlendMode);
MAKE_TO_STRING_FUNC(XrSessionState);
MAKE_TO_STRING_FUNC(XrResult);
MAKE_TO_STRING_FUNC(XrFormFactor);

inline void OpenXRDebugBreak()
{
    STRELKA_ERROR("Breakpoint here to debug.");
    return;
}

inline const char* GetXRErrorString(XrInstance xrInstance, XrResult result)
{
    static char string[XR_MAX_RESULT_STRING_SIZE];
    xrResultToString(xrInstance, result, string);
    return string;
}

#define OPENXR_CHECK(x, y)                                                                                             \
    {                                                                                                                  \
        XrResult result = (x);                                                                                         \
        if (!XR_SUCCEEDED(result))                                                                                     \
        {                                                                                                              \
            STRELKA_ERROR("ERROR: OPENXR: {} {}", int(result), y);                                                     \
            OpenXRDebugBreak();                                                                                        \
        }                                                                                                              \
    }
