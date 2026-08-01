#pragma once
// Light type constants shared between CPU scene code and GPU shader code.
// Keep this header free of CUDA/Metal/platform-specific includes.

enum LightType : int
{
    LIGHT_TYPE_RECT    = 0,
    LIGHT_TYPE_DISC    = 1,
    LIGHT_TYPE_SPHERE  = 2,
    LIGHT_TYPE_DISTANT = 3,
    LIGHT_TYPE_DOME    = 4,
};
