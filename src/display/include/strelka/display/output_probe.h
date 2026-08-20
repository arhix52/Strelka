#pragma once

#include <strelka/display/output_policy.h>

#include <string_view>


namespace oka::display_output
{

struct PlatformDisplayState
{
    VrrStatus vrrStatus = VrrStatus::Unknown;
    float minRefreshRateHz = 0.0f;
    float maxRefreshRateHz = 0.0f;
    float currentRefreshRateHz = 0.0f;
};

PlatformDisplayState parseGdctlOutput(std::string_view output,
                                     std::string_view monitorName);

// nativeWindow is HWND on Windows and ignored on other platforms. The probe is
// read-only and returns Unknown/zero values when its platform API is unavailable.
PlatformDisplayState probePlatformDisplay(std::string_view monitorName,
                                          void *nativeWindow);

} // namespace oka::display_output

