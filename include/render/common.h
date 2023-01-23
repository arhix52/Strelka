#pragma once

#include <settings/settings.h>

namespace oka
{
static constexpr int MAX_FRAMES_IN_FLIGHT = 3;

struct float3;
struct float4;

class Render;

struct SharedContext
{
    size_t mFrameNumber = 0;
    SettingsManager* mSettingsManager = nullptr;
    Render* mRender = nullptr;
};

enum class Result : uint32_t
{
    eOk,
    eFail,
    eOutOfMemory,
};

} // namespace oka
