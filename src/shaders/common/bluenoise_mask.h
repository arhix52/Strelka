#pragma once

// Frozen 128x128 void-and-cluster rank mask shared byte-for-byte with Metal.
__device__ const unsigned short kBlueNoiseRank[16384] = {
#include "bluenoise_rank.inc"
};
