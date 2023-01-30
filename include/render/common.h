#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/compatibility.hpp>

#include <settings/settings.h>

namespace oka
{
static constexpr int MAX_FRAMES_IN_FLIGHT = 3;

#ifdef __APPLE__
struct float3;
struct float4;
#else
using Float3 = glm::float3;
using Float4 = glm::float4;
#endif

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
