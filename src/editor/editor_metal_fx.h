#pragma once

namespace oka
{
namespace editor_metal_fx
{

enum class Mode
{
    Off = 0,
    Spatial = 1,
    TemporalDenoise = 2,
};

inline Mode modeFromSettings(bool denoise, bool enableUpscale)
{
    return denoise ? Mode::TemporalDenoise : (enableUpscale ? Mode::Spatial : Mode::Off);
}

inline bool shouldUpscale(Mode mode, float scale)
{
    return mode != Mode::Off && scale < 1.0f;
}

} // namespace editor_metal_fx
} // namespace oka
