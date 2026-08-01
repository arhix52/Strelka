#pragma once
#include <stdint.h>

// Build 2D CDF for importance-sampled environment map.
// d_envData: device pointer to float4 env map pixel data (RGBA, row-major)
// w, h: env map dimensions
// d_cdfX: output conditional CDF buffer, size w * h
// d_cdfY: output marginal CDF buffer, size h
// totalPower: output total luminance power (host pointer, single float)
extern "C" void buildEnvMapCdf(
    const float* d_envData,
    int w, int h,
    float* d_cdfX,
    float* d_cdfY,
    float* totalPower);
